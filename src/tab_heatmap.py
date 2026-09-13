import gradio as gr
import cv2
import numpy as np
import os
from utils import list_models as get_model_files, get_yolo_model

# ==========================================
# 核心演算法：滑動遮擋熱力圖
# ==========================================
def generate_occlusion_heatmap(img_file, model_path, patch_size, stride, progress=gr.Progress()):
    if not img_file: return None,"請上傳圖片"
    if not os.path.exists(model_path) or "找不到" in model_path: return None,"請選擇正確的 YOLO 模型"
    
    # 強制將 Gradio 的小數點滑桿值轉為整數 (防當機)
    patch_size = int(patch_size)
    stride = int(stride)
    
    # 強大讀圖法：防止 Windows 中文路徑報錯
    img_path = img_file if isinstance(img_file, str) else img_file.name
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return None,"圖片讀取失敗，請確認檔案格式"
    
    try:
        model = get_yolo_model(model_path)
    except Exception as e:
        return None, f"模型載入失敗: {str(e)}"

    # 1. 取得原始圖片的最高信心度
    orig_res = model(img, verbose=False)[0]
    if len(orig_res.boxes) == 0:
        return img[:, :, ::-1],"這張圖裡 AI 找不到任何目標，無法生成熱力圖。"
    
    # 取信心度最高的第一個物件當作基準
    best_box = orig_res.boxes[0]
    orig_conf = float(best_box.conf[0])
    orig_cls = int(best_box.cls[0])
    
    h, w = img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # 2. 開始滑動遮擋 (Sliding Window Occlusion)
    y_steps = list(range(0, h, stride))
    x_steps = list(range(0, w, stride))
    total_steps = len(y_steps) * len(x_steps)
    step_count = 0
    
    for y in y_steps:
        for x in x_steps:
            step_count += 1
            if step_count % 10 == 0:
                progress(step_count / total_steps, desc=f"運算敏感度矩陣中... ({step_count}/{total_steps})")
                
            # 複製原圖並畫上黑色遮擋塊
            occ_img = img.copy()
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            occ_img[y:y2, x:x2] = 0
            
            # 重新預測
            occ_res = model(occ_img, verbose=False)[0]
            
            # 找看看同一個類別還在不在，信心度掉多少
            new_conf = 0.0
            for box in occ_res.boxes:
                if int(box.cls[0]) == orig_cls:
                    new_conf = max(new_conf, float(box.conf[0]))
            
            # 如果擋住這裡導致信心度下降，代表這裡很重要 (熱度增加)
            drop = max(0, orig_conf - new_conf)
            heatmap[y:y2, x:x2] += drop
            
    # 3. 渲染成彩色熱力圖
    heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 把熱力圖疊加回原圖
    overlay = cv2.addWeighted(img, 0.5, colormap, 0.5, 0)
    
    # 轉回 RGB 給 Gradio 顯示
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    msg = f"熱力圖生成完畢！\n原始最高信心度: {orig_conf:.2f}\n紅色區域代表 AI 判定物件的核心特徵。"
    return overlay_rgb, msg

# ==========================================
# UI 介面渲染
# ==========================================
def render_ui():
    gr.Markdown("## 實驗室 V3：AI 視覺遮擋熱力圖 (Occlusion Saliency Map)")
    gr.Markdown("用來解釋 AI 的「思考過程」。我們透過程式邏輯在圖片上滑動黑色方塊，如果擋住某個區域導致 AI 的信心度大幅下降，就代表那是核心特徵（紅色高溫區）。這被稱為**模型可解釋性 (Explainable AI)**。")
    
    with gr.Row():
        with gr.Column(scale=1):
            upload_img = gr.File(label="上傳要產生熱力圖的圖片", file_count="single")
            
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=get_model_files(), label="選擇 YOLO 模型", value=get_model_files()[0], interactive=True)
                refresh_model_btn = gr.Button("", scale=0)
            
            gr.Markdown("*(注意：這個演算法需要進行數百次矩陣運算，預計需要 5~15 秒)*")
            patch_size = gr.Slider(minimum=20, maximum=150, value=60, step=10, label="遮擋方塊大小 (越大算越快，但熱圖越粗糙)")
            stride = gr.Slider(minimum=10, maximum=100, value=30, step=10, label="滑動步長 (越小算越久，但邊緣越精細)")
            
            run_btn = gr.Button("開始解剖 AI 視覺", variant="primary")
            status_txt = gr.Textbox(label="運算報告")
            
        with gr.Column(scale=2):
            result_img = gr.Image(label="AI 特徵熱力圖", interactive=False)
            
    # 按鈕事件綁定
    refresh_model_btn.click(fn=lambda: gr.update(choices=get_model_files(), value=get_model_files()[0]), inputs=None, outputs=[model_dropdown])
    run_btn.click(fn=generate_occlusion_heatmap, inputs=[upload_img, model_dropdown, patch_size, stride], outputs=[result_img, status_txt])