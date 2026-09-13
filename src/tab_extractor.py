import gradio as gr
import cv2
import os
import zipfile
import glob
import torch
import shutil
import queue
import threading
import multiprocessing as mp
from ultralytics import YOLO
from notifier import send_alert
from utils import list_models as get_model_files, device_banner, upload_to_roboflow
from dedup_core import hist_similarity, feature_dedup

TEMP_DIR ="extracted_frames"
os.makedirs(TEMP_DIR, exist_ok=True)

TRANSFER_DIR ="transfer_to_tab3"
os.makedirs(TRANSFER_DIR, exist_ok=True)

def get_downloaded_videos():
    os.makedirs("downloads", exist_ok=True)
    return [os.path.join("downloads", f) for f in os.listdir("downloads") if f.endswith(('.mp4','.webm','.avi'))]

# 新增：將選中的圖片丟進傳送門
def send_to_tab3(selected_images):
    if not selected_images: return "請先選擇圖片！"
    
    # 每次傳送前，先清空舊的傳送門，避免混到上一批的圖
    if os.path.exists(TRANSFER_DIR):
        shutil.rmtree(TRANSFER_DIR)
    os.makedirs(TRANSFER_DIR, exist_ok=True)
    
    # 把選中的圖片複製過去
    for img in selected_images:
        shutil.copy(img, os.path.join(TRANSFER_DIR, os.path.basename(img)))
        
    return f"成功傳送 {len(selected_images)} 張圖片！請切換到第三頁點擊「接收」。"

def process_video(dropdown_path, uploaded_video, start_min, start_sec, end_min, end_sec, interval, sim_threshold, use_yolo, model_path, deep_dedup, dedup_threshold, progress=gr.Progress()):

    video_path = uploaded_video if uploaded_video else dropdown_path
    
    if not video_path: return [],"請先選擇或上傳影片"

    if use_yolo and "找不到" in model_path:
        return [],"請先將 .pt 模型檔放進資料夾，並點擊重整按鈕！"
    
    progress(0, desc="正在清理上一輪的舊資料...")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    total_start_sec = (start_min * 60) + start_sec
    total_end_sec = (end_min * 60) + end_sec
    
    # ==========================================
    # 補回遺失的拼圖：先讀取影片資訊與載入模型
    # ==========================================
    cap_temp = cv2.VideoCapture(video_path)
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release() # 只是用來拿 fps，拿完就關掉
    
    start_frame = int(total_start_sec * fps)
    end_frame = int(total_end_sec * fps) if total_end_sec > 0 else total_video_frames
    
    model = None
    if use_yolo and model_path:
        try:
            model = YOLO(model_path)
        except Exception as e:
            return [], f"模型載入失敗: {str(e)}"
    
    # === 迴圈前的準備作業 ===
    interval_frames = int(fps * interval)
    total_frames_to_process = end_frame - start_frame + 1
    total_tasks = total_frames_to_process // interval_frames
    
    batch_size = 8
    
    # 建立一條輸送帶 (Queue)。
    # maxsize=3 很重要！這代表輸送帶上最多放 3 批圖。
    # 這是防止 CPU 解碼太快，把你的 32GB RAM 全部塞爆導致電腦死機。
    batch_queue = queue.Queue(maxsize=3)
    
    saved_images = []
    seen_track_ids = set()

    # ==========================================
    # 生產者：CPU 專屬的打工仔 (負責解碼與去重)
    # ==========================================
    def producer_cpu():
        cap_thread = cv2.VideoCapture(video_path)
        cap_thread.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr_frame = start_frame
        
        batch_imgs = []
        batch_img_paths = []
        saved_count = 0
        last_frame = None
        
        task_idx = 0 # 新增：用來記錄目前看過幾個抽幀點了

        while curr_frame <= end_frame:
            ret, frame = cap_thread.read()
            if not ret: break
            
            if curr_frame % interval_frames == 0:
                task_idx += 1
                is_dup = False
                
                if last_frame is not None and hist_similarity(frame, last_frame) > sim_threshold:
                    is_dup = True
                    
                if not is_dup:
                    img_path = os.path.join(TEMP_DIR, f"frame_{saved_count:04d}.jpg")
                    batch_imgs.append(frame)
                    batch_img_paths.append(img_path)
                    last_frame = frame.copy()
                    saved_count += 1
                    
                # 裝滿一車，推上輸送帶 (加入 task_idx 讓 GPU 知道進度)
                if len(batch_imgs) == batch_size:
                    batch_queue.put((batch_imgs, batch_img_paths, task_idx))
                    batch_imgs = []
                    batch_img_paths = []
                # 心跳包：如果一直遇到重複的圖導致車裝不滿，每 30 個階段還是要回報一次進度給 GPU
                elif task_idx % 30 == 0:
                    batch_queue.put(([], [], task_idx))
                    
            curr_frame += 1
            
        # 結尾：把剩下不滿一車的也推上去
        if len(batch_imgs) > 0:
            batch_queue.put((batch_imgs, batch_img_paths, task_idx))
            
        # 放一個「毒藥 (None)」告訴 GPU：影片讀完了，你可以下班了
        batch_queue.put(None)
        cap_thread.release()

    # 啟動 CPU 生產線 (放在背景執行)
    threading.Thread(target=producer_cpu, daemon=True).start()

    # ==========================================
    # 消費者：GPU 專屬 (負責無情推論)
    # ==========================================
    processed_count = 0
    
    while True:
        # 從輸送帶拿資料
        batch_data = batch_queue.get()
        
        if batch_data is None: # 吃到毒藥，跳出迴圈
            break
            
        # 解包時多拿一個 task_idx (目前階段)
        b_imgs, b_paths, task_idx = batch_data
        
        # 如果裡面有圖片，就進行 YOLO 運算
        if len(b_imgs) > 0:
            if model is not None:
                for b_idx, b_frame in enumerate(b_imgs):
                    b_img_path = b_paths[b_idx]
                    txt_path = b_img_path.replace('.jpg','.txt')
                    xml_path = b_img_path.replace('.jpg','.xml')
                    
                    current_res = model.track(b_frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                    res = current_res
                    
                    with open(txt_path,'w') as f:
                        for box in res[0].boxes:
                            f.write(f"{int(box.cls[0])} {box.xywhn[0][0]:.5f} {box.xywhn[0][1]:.5f} {box.xywhn[0][2]:.5f} {box.xywhn[0][3]:.5f}\n")
                    
                    h, w, _ = b_frame.shape
                    xml_content = f"<annotation><filename>{os.path.basename(b_img_path)}</filename><size><width>{w}</width><height>{h}</height></size>"
                    for box in res[0].boxes:
                        cls_name = model.names[int(box.cls[0])]
                        x1, y1, x2, y2 = box.xyxy[0]
                        xml_content += f"<object><name>{cls_name}</name><bndbox><xmin>{int(x1)}</xmin><ymin>{int(y1)}</ymin><xmax>{int(x2)}</xmax><ymax>{int(y2)}</ymax></bndbox></object>"
                    xml_content +="</annotation>"
                    
                    with open(xml_path,'w', encoding='utf-8') as f:
                        f.write(xml_content)
                        
                    cv2.imwrite(b_img_path, b_frame)
                    saved_images.append(b_img_path)
            else:
                for b_idx, b_frame in enumerate(b_imgs):
                    b_img_path = b_paths[b_idx]
                    cv2.imwrite(b_img_path, b_frame)
                    saved_images.append(b_img_path)
            
            # 統計實際產出的精華圖片數量
            processed_count += len(b_imgs)
            
        # 完美的進度條回歸！
        current_percent = min(task_idx / total_tasks, 1.0)
        progress(current_percent, desc=f"雙核狂飆中... 第 {task_idx}/{total_tasks} 階段 (已提煉 {processed_count} 張)")

    # ==========================================
    # 第二道精篩(選用):AI 特徵向量深度去重
    # 串流階段的直方圖去重只擋相鄰重複幀;這裡用 MobileNetV2 特徵
    # 兩兩比對,連位移/縮放的連拍廢片也剔除。被刪的圖連同 .txt/.xml 一起清掉。
    # ==========================================
    dedup_msg =""
    if deep_dedup and len(saved_images) > 1:
        progress(1.0, desc="AI 特徵向量深度去重中...")
        unique, removed = feature_dedup(saved_images, int(dedup_threshold), progress)
        unique_set = set(unique)
        for p in saved_images:
            if p not in unique_set:
                for f in (p, p.replace('.jpg','.txt'), p.replace('.jpg','.xml')):
                    if os.path.exists(f):
                        os.remove(f)
        saved_images = unique
        dedup_msg = f"\n AI 特徵向量再精篩,剔除了 {removed} 張連拍廢片。"

    send_alert("抽幀清洗完畢", f"成功從影片中提取 {len(saved_images)} 張精華圖片！")
    return saved_images, f"處理完成！共抽出 {len(saved_images)} 張圖片，已存入 {TEMP_DIR} 資料夾。{dedup_msg}"

# ==========================================
# UI 介面
# ==========================================
FILMSTRIP_CSS = """
<style>
/* ---- 未開預覽時的 grid 排列(縮圖列橫向捲動) ---- */
#filmstrip_gallery .grid-wrap {
    overflow-x: auto !important;
    overflow-y: hidden !important;
}
#filmstrip_gallery .grid-container {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 6px;
}
#filmstrip_gallery .thumbnail-item.thumbnail-lg {
    flex: 0 0 auto !important;
    width: 140px !important;
    height: 100px !important;
}
#filmstrip_gallery .thumbnail-item.thumbnail-lg img {
    width: 140px !important;
    height: 100px !important;
    object-fit: cover !important;
}
#filmstrip_gallery .grid-wrap::-webkit-scrollbar {
    height: 12px;
}
#filmstrip_gallery .grid-wrap::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 6px;
}
#filmstrip_gallery .ff-focus {
    outline: 3px solid #ff9800 !important;
    outline-offset: -3px;
}

/* ---- 開啟預覽(lightbox)後,底部的 .thumbnails 縮圖列 ----
   注意:這是跟上面 grid 完全不同的元素(Gallery 元件在 DOM 裡
   grid 和 preview 是並存的,只是預覽開啟時 grid 被蓋住),
   class 名稱經瀏覽器 DevTools 實測為 Gradio 6.9 的
   `.thumbnails` > `.thumbnail-item.thumbnail-small`,
   不是 grid 用的 `.thumbnail-lg`,選擇器不能共用。
   捲軸可見 + 滾輪/拖曳橫向捲動已經收斂到 gallery_scroll.py,
   對全站所有 Gallery 生效,這裡只留這一頁專屬的縮圖放大。 */
#filmstrip_gallery .thumbnails .thumbnail-item.thumbnail-small {
    flex: 0 0 auto !important;
    width: 76px !important;
    height: 56px !important;
}
#filmstrip_gallery .thumbnails .thumbnail-item.thumbnail-small img {
    width: 76px !important;
    height: 56px !important;
    object-fit: cover !important;
}
</style>
"""

# 注意:JS 不能透過 gr.HTML() 注入!Gradio 的 HTML 元件是用
# element.innerHTML = ... 把這段字串塞進 DOM,瀏覽器規範中透過
# innerHTML 插入的 <script> 標籤天生就是「惰性」的、根本不會執行
# (實測驗證:window 上的旗標從未被設起來)。真正會執行的位置是
# demo.launch(head=...) —— 那段字串會被寫進真正的 <head>,走瀏覽器
# 正常的文件解析流程,<script> 才會跑。所以這段只回傳 <script> 字串,
# 由 app.py 收集起來,一起傳給 demo.launch(head=...)。
FILMSTRIP_HEAD_JS = """
<script>
(function() {
    // ---- 未開預覽 grid 的左右鍵/空白鍵導覽(沿用既有邏輯,只把選擇器
    //      收斂到 .thumbnail-lg,避免預覽開啟時跟 .thumbnail-small 混算) ----
    function setupGridKeyboardNav() {
        let focusIdx = 0;
        function items() {
            var wrap = document.getElementById('filmstrip_gallery');
            return wrap ? wrap.querySelectorAll('.thumbnail-item.thumbnail-lg') : [];
        }
        function highlight() {
            var its = items();
            its.forEach(function(el, i) { el.classList.toggle('ff-focus', i === focusIdx); });
        }
        function scrollIntoView() {
            var its = items();
            if (its[focusIdx]) its[focusIdx].scrollIntoView({behavior: 'smooth', inline: 'center', block: 'nearest'});
        }
        document.addEventListener('keydown', function(e) {
            var tag = (document.activeElement && document.activeElement.tagName) || '';
            if (tag === 'INPUT' || tag === 'TEXTAREA') return;
            var its = items();
            if (its.length === 0) return;
            if (e.key === 'ArrowLeft') {
                focusIdx = Math.max(0, focusIdx - 1);
                highlight(); scrollIntoView(); e.preventDefault();
            } else if (e.key === 'ArrowRight') {
                focusIdx = Math.min(its.length - 1, focusIdx + 1);
                highlight(); scrollIntoView(); e.preventDefault();
            } else if (e.code === 'Space') {
                if (its[focusIdx]) its[focusIdx].click();
                e.preventDefault();
            }
        });
    }
    // Gradio 是動態渲染,DOM 掛載時機不固定,輪詢等到 filmstrip 出現為止
    var t = setInterval(function() {
        if (document.getElementById('filmstrip_gallery')) { setupGridKeyboardNav(); clearInterval(t); }
    }, 500);

    // 預覽模式底部 .thumbnails 列的滾輪/拖曳橫向捲動已經收斂到
    // gallery_scroll.py(對全站所有 Gallery 生效),不在這裡重複。
})();
</script>
"""


def render_ui(global_api, global_ws, global_proj):
    current_images = gr.State([])
    displayed_images = gr.State([])
    selected_images = gr.State([])

    # 硬體狀態燈偵測(共用)
    gr.Markdown(f"### {device_banner()}")
    gr.Markdown("---")
    gr.HTML(FILMSTRIP_CSS)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row(equal_height=True):
                video_dropdown = gr.Dropdown(choices=get_downloaded_videos(), label="選擇第一頁下載的影片", interactive=True, scale=1)
                refresh_btn = gr.Button("", scale=0, min_width=60)
                
            video_upload = gr.Video(label="或手動上傳影片")
            
            with gr.Row():
                with gr.Column(min_width=100):
                    gr.Markdown("**開始時間**")
                    with gr.Row():
                        start_min = gr.Number(value=0, label="分", precision=0)
                        start_sec = gr.Number(value=0, label="秒", precision=0)
                with gr.Column(min_width=100):
                    gr.Markdown("**結束時間 [都填0=到底]**")
                    with gr.Row():
                        end_min = gr.Number(value=0, label="分", precision=0)
                        end_sec = gr.Number(value=0, label="秒", precision=0)
                
            interval_slider = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="抽幀間隔(秒)")
            sim_slider = gr.Slider(0.5, 0.99, value=0.90, step=0.01, label="去重嚴格度")
            
            use_yolo = gr.Checkbox(label="啟用 YOLO 預先標註 (.txt)", value=True)
            
            with gr.Row():
                model_path = gr.Dropdown(choices=get_model_files(), label="選擇 AI 模型", interactive=True)
                refresh_model_btn = gr.Button("", scale=0)

            deep_dedup_cb = gr.Checkbox(label="抽完再用 AI 特徵向量深度去重 (較慢但更乾淨，連位移/縮放的連拍廢片也剔除)", value=False)
            dedup_thr_slider = gr.Slider(80, 100, value=95, step=1, label="特徵去重嚴格度 (%) (越高越嚴格，越容易判定為不同圖)")

            process_btn = gr.Button("開始抽幀", variant="primary")

        with gr.Column(scale=2):
            sys_msg = gr.Textbox(label="狀態")
            
            # 新增：全選按鈕與動態計數器
            with gr.Row():
                select_all_btn = gr.Button("全部選取", variant="secondary", size="sm")
                clear_all_btn = gr.Button("全部清除", variant="stop", size="sm")
            
            # 標題留空，我們等一下用程式動態把數字塞進去
            selected_dd = gr.Dropdown(label="已選圖片 (0 / 0)", choices=[], multiselect=True, interactive=True)
            
            with gr.Row():
                pack_btn = gr.Button("下載勾選 ZIP", variant="secondary")
                # 新增這顆傳送按鈕
                send_tab3_btn = gr.Button("傳送至第三頁", variant="primary")
                # 把 Roboflow 按鈕降級成灰色 (因為以後上傳都在第三頁做了)
                rf_upload_btn = gr.Button("導入 Roboflow", variant="secondary")
            
            download_out = gr.File(label="下載區 (供 ZIP 使用)")

            step_slider = gr.Slider(
                1, 20, value=1, step=1,
                label="每 N 幀顯示一張(找連續片段用,不影響實際抽出的幀數；定位到大概位置後再切回 1)",
            )
            gallery = gr.Gallery(
                label="結果展示 (點圖加入選單，左右鍵移動、空白鍵選取/取消)",
                elem_id="filmstrip_gallery",
                columns=100, height=140, allow_preview=True,
            )

    # --- 互動邏輯綁定 ---

    def auto_fill_video(dd_path): return dd_path
    video_dropdown.change(fn=auto_fill_video, inputs=[video_dropdown], outputs=[video_upload])
    
    refresh_btn.click(fn=lambda: gr.update(choices=get_downloaded_videos()), inputs=None, outputs=[video_dropdown])
    
    refresh_model_btn.click(
        fn=lambda: gr.update(choices=get_model_files()),
        inputs=None,
        outputs=[model_path]
    )
    
    def apply_step(all_imgs, step):
        """依「每 N 幀顯示一張」重新抽稀 gallery 顯示內容(不影響實際抽出的幀數)。"""
        step = max(1, int(step))
        subset = all_imgs[::step]
        return [(p, os.path.basename(p)) for p in subset], subset

    process_btn.click(
        fn=process_video,
        inputs=[video_dropdown, video_upload, start_min, start_sec, end_min, end_sec, interval_slider, sim_slider, use_yolo, model_path, deep_dedup_cb, dedup_thr_slider],
        outputs=[current_images, sys_msg]
    ).then(
        fn=lambda paths, step: apply_step(paths, step),
        inputs=[current_images, step_slider],
        outputs=[gallery, displayed_images]
    ).then(
        fn=lambda: ([], gr.update(choices=[], value=[])), inputs=None, outputs=[selected_images, selected_dd]
    )

    step_slider.change(
        fn=apply_step,
        inputs=[current_images, step_slider],
        outputs=[gallery, displayed_images],
    )

    def select_image(curr_selected, shown_imgs, all_imgs, evt: gr.SelectData):
        idx = evt.index
        if idx >= len(shown_imgs):
            filenames = [os.path.basename(p) for p in curr_selected]
            return curr_selected, gr.update(choices=filenames, value=filenames, label=f"已選圖片 ({len(curr_selected)} / {len(all_imgs)})")

        img_path = shown_imgs[idx]
        # 再選一次同一張 = 取消選取(切換式多選)
        if img_path in curr_selected:
            curr_selected = [p for p in curr_selected if p != img_path]
        else:
            curr_selected = curr_selected + [img_path]

        filenames = [os.path.basename(p) for p in curr_selected]
        # 更新：把數量寫進 label
        return curr_selected, gr.update(choices=filenames, value=filenames, label=f"已選圖片 ({len(curr_selected)} / {len(all_imgs)})")

    gallery.select(fn=select_image, inputs=[selected_images, displayed_images, current_images], outputs=[selected_images, selected_dd])
    
    def sync_dropdown(dd_values, curr_selected, all_imgs):
        new_selected = [p for p in curr_selected if os.path.basename(p) in dd_values]
        # 更新：同步更新數量標題
        return new_selected, gr.update(label=f"已選圖片 ({len(new_selected)} / {len(all_imgs)})")

    # 注意這裡的 inputs 多加了 current_images，用來計算總數
    selected_dd.change(fn=sync_dropdown, inputs=[selected_dd, selected_images, current_images], outputs=[selected_images, selected_dd])
    
    def pack_zip(imgs):
        if not imgs: return None,"請先選擇圖片！"
        zp = os.path.join(TEMP_DIR,"dataset.zip")
        with zipfile.ZipFile(zp,'w') as zf:
            for i in imgs:
                zf.write(i, os.path.basename(i))
                txt_p = i.replace('.jpg','.txt')
                if os.path.exists(txt_p): zf.write(txt_p, os.path.basename(txt_p))
        return zp,"打包完成！"

    # 新增：全部選取與全部清除的功能
    def select_all_imgs(all_imgs):
        if not all_imgs:
            return [], gr.update(value=[], label="已選圖片 (0 / 0)")
        filenames = [os.path.basename(p) for p in all_imgs]
        # 回傳所有圖片，並更新 Dropdown 標題顯示數量
        return all_imgs.copy(), gr.update(choices=filenames, value=filenames, label=f"已選圖片 ({len(all_imgs)} / {len(all_imgs)})")

    select_all_btn.click(
        fn=select_all_imgs,
        inputs=[current_images],
        outputs=[selected_images, selected_dd]
    )

    def clear_all_imgs(all_imgs):
        filenames = [os.path.basename(p) for p in all_imgs] if all_imgs else []
        # 清空已選清單，並歸零計數器
        return [], gr.update(choices=filenames, value=[], label=f"已選圖片 (0 / {len(all_imgs) if all_imgs else 0})")

    clear_all_btn.click(
        fn=clear_all_imgs,
        inputs=[current_images],
        outputs=[selected_images, selected_dd]
    )
        
    pack_btn.click(fn=pack_zip, inputs=[selected_images], outputs=[download_out, sys_msg])
    
    rf_upload_btn.click(
        fn=upload_to_roboflow,
        inputs=[selected_images, global_api, global_ws, global_proj],
        outputs=[sys_msg],
    )
    send_tab3_btn.click(fn=send_to_tab3, inputs=[selected_images], outputs=[sys_msg])