import gradio as gr
import cv2
import os
import shutil
import tempfile
import time
import zipfile
import glob
import random
from notifier import send_alert
from utils import (list_models as get_model_files, device_banner, get_yolo_model,
                    upload_to_roboflow as upload_batch_to_roboflow)
from annotate_core import annotate, draw_overlay, build_eda_figures

# 固定放在專案目錄下,不用系統暫存區(會被 Windows 暫存清理機制清掉,
# 導致長時間執行中 BATCH_TEMP_DIR 的父目錄憑空消失、mkdtemp 找不到路徑而炸掉)
BATCH_TEMP_DIR = os.path.join(os.getcwd(), "temp", "batch_annotate")
BATCH_TEMP_MAX_AGE_DAYS = 7


def _cleanup_old_batches(max_age_days=BATCH_TEMP_MAX_AGE_DAYS):
    """程式啟動時清掉超過 max_age_days 天的舊批次資料夾,避免無限累積。"""
    if not os.path.isdir(BATCH_TEMP_DIR):
        return
    cutoff = time.time() - max_age_days * 86400
    for name in os.listdir(BATCH_TEMP_DIR):
        path = os.path.join(BATCH_TEMP_DIR, name)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


os.makedirs(BATCH_TEMP_DIR, exist_ok=True)
_cleanup_old_batches()


def _ensemble_log_line(filename, st):
    """根據單張圖的 ensemble 戰況,產生一段「有溫度」的對話日誌。"""
    main_box_count = st["main_box_count"]
    duplicate_count = st["duplicate_count"]
    added_by_aux = st["added_by_aux"]
    aux_details = st["aux_details"]
    if main_box_count == 0:
        main_quotes = [
            "初步掃描，未發現明顯目標。",
            "畫面似乎很乾淨，沒有達到門檻的物件。",
            "無發現。是不是空鏡頭？"
        ]
        aux_quotes = [
            f"提出異議！角落有微弱特徵，我補上了 {added_by_aux} 個 {aux_details[0].split('(')[0]}。",
            f"等等，切換檢測權重後，我看到了盲區目標：{','.join(aux_details[:2])}。",
            f"不能漏掉這個。強制追加 {added_by_aux} 個困難樣本。"
        ]
    else:
        main_quotes = [
            f"我鎖定了 {main_box_count} 個主要特徵。",
            f"初步標記完畢，共 {main_box_count} 個物件。",
            f"高信心度目標確認，共 {main_box_count} 個。"
        ]
        aux_quotes = [
            f"同意，我們的判定重疊了 {duplicate_count} 個。但我另外擴充了邊緣細節：{','.join(aux_details[:2])}。",
            f"主體沒錯。但我幫你把剩下的 {added_by_aux} 個 {aux_details[0].split('(')[0]} 補齊了。",
            f"複檢完成。除了有共識的 {duplicate_count} 個，我再追加 {added_by_aux} 個輔助框。"
        ]
    return (f"[{filename}]\n 主模型:「{random.choice(main_quotes)}」\n"
            f"輔助:「{random.choice(aux_quotes)}」\n")


# 批次處理核心邏輯 (共用 annotate_core:聯合診斷 + EDA 偏見分析)
def process_batch_images(files, model_path, use_ensemble, model2_path, conf_threshold, hard_threshold, padding_val, progress=gr.Progress()):
    if not files: return [], None,"請拖曳上傳圖片", None, None, None, None
    if not model_path or "找不到" in model_path: return [], None,"請選擇有效的主模型", None, None, None, None

    try:
        model1 = get_yolo_model(model_path)
    except Exception as e:
        return [], None, f"主模型載入失敗: {str(e)}", None, None, None, None

    # 判斷是否啟用第二模型
    model2 = None
    if use_ensemble and model2_path and "找不到" not in model2_path:
        try:
            model2 = get_yolo_model(model2_path)
        except Exception:
            model2 = None
    use_ensemble = model2 is not None

    processed_images = []
    preview_images = []
    class_counts = {}
    eda_cx, eda_cy, eda_ar = [], [], [] # EDA 偏見分析數據
    os.makedirs(BATCH_TEMP_DIR, exist_ok=True)  # 每次用前都重建,防止資料夾在執行期間被外部清掉
    job_dir = tempfile.mkdtemp(dir=BATCH_TEMP_DIR)

    total_main_boxes = 0
    total_aux_added = 0
    leak_logs = []

    for i, file_obj in enumerate(files):
        original_path = file_obj.name
        filename = os.path.basename(original_path)
        new_img_path = os.path.join(job_dir, filename)
        shutil.copy(original_path, new_img_path)

        # 共用核心:主模型 +(可選)輔助模型 → XML / 繪框 / EDA / 戰況
        r = annotate(new_img_path, filename, model1, model2,
                     conf_threshold, hard_threshold, padding_val)

        # 彙整標籤統計與 EDA
        for k, v in r["class_counts"].items():
            class_counts[k] = class_counts.get(k, 0) + v
        eda_cx.extend(r["eda_cx"])
        eda_cy.extend(r["eda_cy"])
        eda_ar.extend(r["eda_ar"])
        total_main_boxes += r["stats"]["main_box_count"]

        # 寫 XML
        with open(new_img_path.rsplit('.', 1)[0] +'.xml','w', encoding='utf-8') as f:
            f.write(r["xml"])

        # 繪製預覽圖(檔名浮水印 + 橘色框)
        preview = draw_overlay(cv2.imread(new_img_path), filename, r["draw_boxes"])
        preview_path = new_img_path.rsplit('.', 1)[0] +'_preview.jpg'
        cv2.imwrite(preview_path, preview)
        preview_images.append(preview_path)
        processed_images.append(new_img_path)

        # 聯合診斷對話日誌
        if use_ensemble and r["stats"]["added_by_aux"] > 0:
            total_aux_added += r["stats"]["added_by_aux"]
            if len(leak_logs) < 50:
                leak_logs.append(_ensemble_log_line(filename, r["stats"]))

        progress((i + 1) / len(files), desc=f"AI 標註與診斷中... ({i+1}/{len(files)})")

    zip_path = os.path.join(job_dir,"batch_annotated.zip")
    with zipfile.ZipFile(zip_path,'w') as zf:
        for img_p in processed_images:
            zf.write(img_p, os.path.basename(img_p))
            xml_p = img_p.rsplit('.', 1)[0] +'.xml'
            if os.path.exists(xml_p):
                zf.write(xml_p, os.path.basename(xml_p))

    # 三張分析圖表(共用)
    fig_stats, fig_heat, fig_ar = build_eda_figures(class_counts, eda_cx, eda_cy, eda_ar)

    try:
        send_alert("批次標註完成", f"已成功處理 {len(processed_images)} 張影像！")
    except Exception:
        pass

    # 組裝系統對話流
    final_status = f"成功處理 {len(processed_images)} 張圖片！\n"
    if use_ensemble:
        final_status +="="* 40 +"\n"
        final_status +="【聯合診斷通訊頻道】已開啟\n"
        final_status += f"總結戰況：主模型標出 {total_main_boxes} 個，輔助模型成功救援 {total_aux_added} 個。\n"
        final_status +="-"* 40 +"\n"

        if total_aux_added > 0:
            final_status +="通訊日誌抽樣：\n"
            final_status +="\n".join(leak_logs)
        else:
            final_status +="通訊日誌：主模型表現完美，輔助模型表示無須補充。"

    return processed_images, zip_path, final_status, fig_stats, fig_heat, fig_ar, preview_images

# API 上傳函數改用 utils.upload_to_roboflow(已於頂層 import 為 upload_batch_to_roboflow)

# ==========================================
# UI 介面設計
# ==========================================
def render_ui(global_api, global_ws, global_proj):
    processed_imgs_state = gr.State([])

    gr.Markdown(f"### {device_banner()}")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("步驟一：匯入外部圖片", open=True):
                upload_files = gr.File(label="拖曳圖片到此處", file_count="multiple", file_types=["image"])
                receive_btn = gr.Button("接收來自第二頁的圖片", variant="secondary")
            
            gr.Markdown("### 步驟二：選擇自動標註引擎")
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=get_model_files(), label="主模型 (決定哪些標籤可以上桌)", value=get_model_files()[0], interactive=True)
                refresh_model_btn = gr.Button("", scale=0)
            
            # 新增：聯合診斷的勾勾與隱藏選單
            use_ensemble_cb = gr.Checkbox(label="啟用聯合診斷 (使用輔助模型幫忙抓漏)", value=False)
            model2_dropdown = gr.Dropdown(choices=get_model_files(), label="選擇輔助模型", visible=False, interactive=True)
            
            # 勾選時才顯示輔助模型選單
            use_ensemble_cb.change(fn=lambda x: gr.update(visible=x), inputs=[use_ensemble_cb], outputs=[model2_dropdown], queue=False)
            
            gr.Markdown("### 參數設定")
            conf_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.05, label="信心度過濾下限 (低於此值將被拋棄)")
            hard_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, step=0.05, label="困難樣本上限 (標記為 HARD)")
            padding_slider = gr.Slider(minimum=0, maximum=20, value=5, step=1, label="邊界框擴張比例 (Padding %)")
            
            process_btn = gr.Button("開始批次標註", variant="primary")
            # 把原本單行的 Textbox 加上 lines=10 讓它長高，用來容納對話紀錄
            sys_msg = gr.Textbox(label="系統狀態與通訊日誌", interactive=False, lines=10)

        with gr.Column(scale=2):
            gr.Markdown("### 處理結果預覽")
            gallery = gr.Gallery(label="已處理的圖片", columns=4, height=300, allow_preview=True)
            
            ## 新增：健康度與偏見分析儀表板
            gr.Markdown("### 資料集體檢與偏見分析 (EDA)")
            stats_plot = gr.Plot(label="標籤分佈統計")
            with gr.Row():
                heatmap_plot = gr.Plot(label="空間分佈熱力圖")
                ar_plot = gr.Plot(label="框型長寬比分佈")
            
            gr.Markdown("---")
            with gr.Row():
                download_out = gr.File(label="下載 ZIP (圖片 + .xml)")
                
            rf_upload_btn = gr.Button("將這批圖片直接射進 Roboflow", variant="secondary")

    def load_from_transfer():
        imgs = glob.glob("transfer_to_tab3/*.jpg")
        return imgs if imgs else None

    receive_btn.click(fn=load_from_transfer, inputs=None, outputs=[upload_files])

    # 按鈕邏輯
    refresh_model_btn.click(fn=lambda: gr.update(choices=get_model_files(), value=get_model_files()[0]), inputs=None, outputs=[model_dropdown])

    # 修改：將滑桿當作輸入，將圖表當作輸出
    process_btn.click(
        fn=process_batch_images,
        inputs=[
            upload_files,
            model_dropdown,
            use_ensemble_cb,
            model2_dropdown,
            conf_slider,
            hard_slider,
            padding_slider
        ],
        outputs=[
            processed_imgs_state,
            download_out,
            sys_msg,
            stats_plot,
            heatmap_plot,
            ar_plot,
            gallery
        ]
    )
    
    rf_upload_btn.click(
        fn=upload_batch_to_roboflow,
        inputs=[processed_imgs_state, global_api, global_ws, global_proj],
        outputs=[sys_msg]
    )