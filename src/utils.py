"""共用工具集中地。

原本散落在各分頁(extractor / batch_annotate / heatmap / tracking / distill /
youtube)的重複邏輯,統一收斂到這裡,改一處即可。功能與原本完全一致,
只是去掉重複。
"""
import os
import glob
import torch

# gradio 只用在 upload_to_roboflow() 的 progress=gr.Progress() 預設值
# (讓 Gradio 版 .click(fn=upload_to_roboflow) 能自動注入即時進度條)。
# Qt 桌面版呼叫這個函式時一律自己傳 progress callback,不吃這個預設值,
# 所以放寬成可選匯入——沒裝 gradio(例如瘦身過的桌面版打包)也能正常匯入
# 這個模組,只是拿不到 gr.Progress() 那個預設物件。
try:
    import gradio as gr
except ImportError:
    gr = None

MODEL_WARN ="找不到 .pt 模型檔"


# ==========================================================
# 模型掃描
# ==========================================================
def list_models(with_placeholder=True):
    """掃描 models/ 底下的 .pt 檔。

    with_placeholder=True : 找不到時回傳 [MODEL_WARN](給 Dropdown 當提示用)。
    with_placeholder=False: 回傳乾淨的 list(distill 要再串接官方底模,不能混進警告字)。
    """
    os.makedirs("models", exist_ok=True)
    models = glob.glob("models/*.pt")
    if with_placeholder:
        return models if models else [MODEL_WARN]
    return models


# ==========================================================
# GPU 狀態橫幅
# ==========================================================
def device_banner():
    """回傳一行運算核心狀態的 Markdown 字串。"""
    if torch.cuda.is_available():
        return f"運算核心狀態: {torch.cuda.get_device_name(0)} (CUDA 硬體加速啟用中)"
    return "運算核心狀態: CPU (建議配置 NVIDIA GPU 以大幅提升處理速度)"


# ==========================================================
# YOLO 模型快取(batch_annotate / heatmap / distill 的老師模型共用)
# ==========================================================
_MODEL_CACHE = {}


def get_yolo_model(path):
    """依路徑快取已載入的 YOLO 模型,避免每次按鈕都重新從硬碟讀檔+暖機。

    只給「純推論、每次呼叫互不影響狀態」的用途使用(batch_annotate 的
    主/輔助模型、heatmap、distill 的老師模型皆是如此)。凡是會被訓練改權重
    (student 模型)、或用 .track(persist=True) 保留追蹤狀態跨影片使用
    (extractor / tracking 分頁)的呼叫端,絕對不能走這個快取,必須每次都
    重新 YOLO(path),否則會把上一次的訓練結果或追蹤 ID 狀態帶進下一次。
    """
    from ultralytics import YOLO
    key = os.path.abspath(path)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = YOLO(path)
        _MODEL_CACHE[key] = model
    return model


# ==========================================================
# FFmpeg / FFprobe 定位
# 優先序:系統 PATH → 專案 bin/(找不到就自動下載,見 ffmpeg_setup.py)。
# 這兩個函式本身只做單純查找、不下載、不拋例外——真正的「找不到就自動
# 下載」在程式啟動時由 ffmpeg_setup.ensure_ffmpeg() 處理一次,這裡找不到
# 就回傳裸命令名稱交給系統 PATH 做最後嘗試,跟呼叫端既有的 subprocess
# 例外處理相容(找不到執行檔會在 subprocess 呼叫當下自然噴錯,而不是在
# 這裡就把整條呼叫鏈炸掉)。
# ==========================================================
def ffmpeg_path():
    from ffmpeg_setup import find_ffmpeg
    ffmpeg, _ = find_ffmpeg()
    return ffmpeg or "ffmpeg"


def ffprobe_path():
    from ffmpeg_setup import find_ffmpeg
    _, ffprobe = find_ffmpeg()
    return ffprobe or "ffprobe"


# ==========================================================
# Roboflow 上傳(extractor 與 batch_annotate 共用)
# 每張圖找同名 .xml 當標註一起上傳。
# ==========================================================
def upload_to_roboflow(image_paths, api_key, workspace, project,
                        progress=gr.Progress() if gr else None):
    if not image_paths:
        return "沒有可上傳的圖片！請先加入清單。"
    if not api_key or not workspace or not project:
        return "請填寫完整的 Roboflow API 資訊！"

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)

        success_count = 0
        total = len(image_paths)
        for i, img_path in enumerate(image_paths):
            xml_path = img_path.rsplit('.', 1)[0] +'.xml'
            ann_path = xml_path if os.path.exists(xml_path) else None

            proj.upload(img_path, annotation_path=ann_path, num_retry_uploads=3)
            success_count += 1
            if progress is not None:
                progress((i + 1) / total, desc=f"上傳中... ({success_count}/{total})")

        return f"神速上傳完畢！成功將 {success_count} 張圖片匯入 Roboflow！"
    except Exception as e:
        return f"上傳失敗: {str(e)}"
