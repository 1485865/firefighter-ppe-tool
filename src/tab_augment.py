import gradio as gr
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

def parse_xml(xml_path):
    """讀取 XML 找出所有標註框"""
    if not os.path.exists(xml_path): return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        boxes.append({"name": name,"box": [xmin, ymin, xmax, ymax]})
    return boxes

def apply_augmentation(img_file, brightness, noise_level, do_flip):
    if img_file is None:
        return None,"請先上傳一張帶有 XML 的圖片"
        
    img_path = img_file.name
    xml_path = img_path.rsplit('.', 1)[0] +'.xml'
    
    # 讀取影像與標註框
    img = cv2.imread(img_path)
    if img is None: return None,"影像讀取失敗"
    h, w = img.shape[:2]
    boxes = parse_xml(xml_path)

    # ========================================
    # 核心演算法 1：亮度調整 (模擬夜間/逆光)
    # ========================================
    if brightness != 0:
        # 利用 cv2.convertScaleAbs 進行矩陣線性轉換
        img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)

    # ========================================
    # 核心演算法 2：高斯雜訊 (模擬煙霧/火場懸浮微粒)
    # ========================================
    if noise_level > 0:
        # 生成與影像同大小的常態分佈雜訊矩陣
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        noisy_img = cv2.add(img.astype(np.float32), noise)
        img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # ========================================
    # 核心演算法 3：水平翻轉 (包含 BBox 座標重新計算)
    # ========================================
    if do_flip:
        img = cv2.flip(img, 1)
        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            # 數學公式：新的 x1 = 圖片寬度 - 舊的 x2
            new_x1 = w - x2
            new_x2 = w - x1
            b["box"] = [new_x1, y1, new_x2, y2]

    # ========================================
    # 繪製預覽框
    # ========================================
    preview_img = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        # 畫橘色框
        cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        # 畫標籤底色與文字
        cv2.rectangle(preview_img, (x1, y1-20), (x1+100, y1), (0, 165, 255), -1)
        cv2.putText(preview_img, b["name"], (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
    msg = f"處理完成！\n亮度: {brightness}\n雜訊: {noise_level}\n翻轉: {'是' if do_flip else '否'}\n偵測到 {len(boxes)} 個標籤。"
    return preview_img, msg

def render_ui():
    gr.Markdown("## 實驗室 V1：火場環境增強模擬器")
    gr.Markdown("在這裡，你可以透過矩陣運算，模擬火場的**低光源**與**煙霧微粒干擾**，並動態重新計算 Bounding Box 座標。這是驗證 AI 模型魯棒性 (Robustness) 的關鍵步驟。")
    
    with gr.Row():
        with gr.Column(scale=1):
            upload_img = gr.File(label="上傳測試圖片 (請確認同資料夾有同名 .xml 檔)", file_types=["image"])
            
            gr.Markdown("### 環境參數控制器")
            brightness_slider = gr.Slider(minimum=-100, maximum=100, value=0, step=5, label="亮度調整 (負值模擬夜間，正值模擬火光)")
            noise_slider = gr.Slider(minimum=0, maximum=100, value=0, step=5, label="煙霧微粒濃度 (Gaussian Noise)")
            flip_checkbox = gr.Checkbox(label="水平翻轉 (自動重新計算 XML 座標)", value=False)
            
        with gr.Column(scale=2):
            result_img = gr.Image(label="即時增強預覽", interactive=False)
            status_text = gr.Textbox(label="運算狀態")

    # 綁定即時互動事件：只要滑桿一動，馬上重新計算！
    inputs = [upload_img, brightness_slider, noise_slider, flip_checkbox]
    
    brightness_slider.change(fn=apply_augmentation, inputs=inputs, outputs=[result_img, status_text])
    noise_slider.change(fn=apply_augmentation, inputs=inputs, outputs=[result_img, status_text])
    flip_checkbox.change(fn=apply_augmentation, inputs=inputs, outputs=[result_img, status_text])
    upload_img.change(fn=apply_augmentation, inputs=inputs, outputs=[result_img, status_text])