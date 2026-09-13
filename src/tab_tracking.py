import gradio as gr
import cv2
import os
import numpy as np
from ultralytics import YOLO
import tempfile
from collections import defaultdict
import subprocess
from utils import list_models as get_model_files, ffmpeg_path

def process_video_tracking(video_file, model_path, progress=gr.Progress()):
    if not video_file: return None,"請上傳影片"
    if "找不到" in model_path: return None,"請選擇正確的模型"
    
    video_path = video_file.name
    try:
        model = YOLO(model_path)
    except Exception as e:
        return None, f"模型載入失敗: {e}"

    # 準備讀取影片
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 建立輸出影片的暫存檔 (使用 mp4v 編碼)
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out_path = temp_out.name
    temp_out.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    # 用來記錄每個 ID 的歷史軌跡座標
    track_history = defaultdict(lambda: [])
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        if frame_count % 10 == 0:
            progress(frame_count / total_frames, desc=f"AI 軌跡追蹤中... ({frame_count}/{total_frames})")
            
        # 核心：啟動追蹤模式 (persist=True 代表記住上一幀的狀態)
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # 取得框、類別、與追蹤 ID
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                cls_name = model.names[cls_id]
                
                # 計算中心點並記錄軌跡
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track = track_history[track_id]
                track.append((cx, cy))
                if len(track) > 60: # 軌跡最長保留 60 幀 (約 2 秒)
                    track.pop(0)
                    
                # 畫出 Bounding Box 與 ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 畫出小尾巴 (軌跡線)
                for i in range(1, len(track)):
                    # 越近的軌跡線越粗
                    thickness = int(np.sqrt(64 / float(len(track) - i + 1)) * 2)
                    cv2.line(frame, track[i - 1], track[i], (0, 165, 255), thickness)
                    
        out.write(frame)

    cap.release()
    out.release()
    
    # ==========================================
    # 聲音縫合術：用 FFmpeg 把原影片的聲音貼回新影片
    # ==========================================
    final_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    final_out_path = final_out.name
    final_out.close()
    
    # 尋找 ffmpeg(共用定位:../bin → 本資料夾 → 系統 PATH)
    ffmpeg_cmd = ffmpeg_path()
    
    # FFmpeg 指令：-map 0:v:0 拿新影片畫面，-map 1:a:0? 拿原影片聲音
    cmd = [
        ffmpeg_cmd,'-y',
        '-i', out_path, # 輸入 1：無聲追蹤影片
        '-i', video_path, # 輸入 2：有聲原始影片
        '-c:v','copy', # 畫面直接複製
        '-c:a','aac', # 聲音重新編碼確保相容
        '-map','0:v:0', # 取輸入 1 的影像
        '-map','1:a:0?', # 取輸入 2 的聲音 (?代表如果沒聲音就算了)
        '-shortest', # 以最短的長度為準
        final_out_path
    ]
    
    try:
        # 靜默執行合併
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return_path = final_out_path
        msg = f"軌跡追蹤與音效合併完成！共處理 {total_frames} 幀。"
    except Exception as e:
        # 如果合併失敗，至少退回無聲版本
        return_path = out_path
        msg = f"軌跡追蹤完成，但音效合併失敗 (僅輸出無聲畫面)。"

    return return_path, msg

def render_ui():
    gr.Markdown("## 實驗室 V6：影片多目標軌跡追蹤 (BoT-SORT Tracking)")
    gr.Markdown("超越靜態圖片！利用 YOLO 內建的追蹤演算法，賦予畫面上每個消防員專屬的 ID，並計算他們的移動軌跡 (小尾巴)。這能應用於火場指揮系統的人員計算與動線分析。")
    
    with gr.Row():
        with gr.Column(scale=1):
            upload_vid = gr.File(label="上傳測試影片 (建議 10~30 秒)", file_types=["video"])
            
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=get_model_files(), label="選擇 YOLO 模型", value=get_model_files()[0], interactive=True)
                refresh_btn = gr.Button("", scale=0)
                
            run_btn = gr.Button("啟動軌跡追蹤", variant="primary")
            status_txt = gr.Textbox(label="處理報告")
            
        with gr.Column(scale=2):
            result_vid = gr.Video(label="動態軌跡分析結果")
            
    refresh_btn.click(fn=lambda: gr.update(choices=get_model_files(), value=get_model_files()[0]), inputs=None, outputs=[model_dropdown])
    run_btn.click(fn=process_video_tracking, inputs=[upload_vid, model_dropdown], outputs=[result_vid, status_txt])