"""跨模型共識標註 UI:v11s / v26s / VLM 至少勾 2 個來源投票產生 pseudo-label。"""
import os
import time

import gradio as gr

import consensus_core as cc
from utils import list_models as get_model_files, get_yolo_model

TEMP_DIR = os.path.join(os.getcwd(), "temp", "consensus")
os.makedirs(TEMP_DIR, exist_ok=True)


def _api_key_present():
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def run_batch(folder, use_m1, model1_path, use_m2, model2_path, use_vlm,
              iou_threshold, vote_rule, merge_mode, progress=gr.Progress()):
    if not folder or not os.path.isdir(folder):
        return "找不到資料夾,請確認路徑正確。", []

    enabled_count = sum([use_m1, use_m2, use_vlm and _api_key_present()])
    if enabled_count < 2:
        return "至少要勾選 2 個來源才能進行共識投票。", []

    models = {}
    try:
        if use_m1:
            if not model1_path or "找不到" in model1_path:
                return "請選擇有效的模型一。", []
            models["v11s"] = get_yolo_model(model1_path)
        if use_m2:
            if not model2_path or "找不到" in model2_path:
                return "請選擇有效的模型二。", []
            models["v26s"] = get_yolo_model(model2_path)
    except Exception as e:
        return f"模型載入失敗: {e}", []

    api_key = os.environ.get("OPENAI_API_KEY", "").strip() if use_vlm else None
    if use_vlm and not api_key:
        return "已勾選 VLM,但環境變數 OPENAI_API_KEY 未設定,請先設定後重新啟動本工具。", []

    out_dir = os.path.join(TEMP_DIR, str(int(time.time())))
    summary, preview_items, failed = cc.run_consensus_batch(
        folder, models, api_key, iou_threshold, vote_rule, merge_mode,
        out_dir, progress,
    )

    unique_lines = "\n".join(
        f"  {src}: {cnt}" for src, cnt in summary["unique_by_source"].items())
    msg = (
        f"共識比對完成,輸出至: {out_dir}\n\n"
        f"全數同意: {summary['unanimous']} 個物件\n"
        f"多數同意(已標 _HARD): {summary['majority']} 個物件\n"
        f"僅單一來源(已標 _HARD): {summary['single']} 個物件\n"
        f"類別不一致(已標 _HARD): {summary['conflict']} 個物件\n\n"
        f"各來源獨有偵測數:\n{unique_lines}"
    )
    if failed:
        msg += "\n\n失敗清單:\n" + "\n".join(
            f"{os.path.basename(p)}: {err}" for p, err in failed[:30])

    return msg, preview_items


def render_ui():
    gr.Markdown("## 跨模型共識標註(Ensemble Consensus)")
    gr.Markdown(
        "多個模型/來源對同一張圖各跑一次推論,用 IoU 空間聚類做多數決過濾單一模型的誤判。"
        "至少要勾選 2 個來源。"
    )

    key_present = _api_key_present()

    with gr.Row():
        with gr.Column(scale=1):
            folder_input = gr.Textbox(
                label="圖片資料夾路徑", value="extracted_frames",
            )

            with gr.Row():
                use_m1 = gr.Checkbox(label="v11s 本機模型", value=True)
                model1_dd = gr.Dropdown(choices=get_model_files(), label="模型一",
                                        value=get_model_files()[0])
            with gr.Row():
                use_m2 = gr.Checkbox(label="v26s 本機模型", value=True)
                model2_dd = gr.Dropdown(choices=get_model_files(), label="模型二",
                                        value=get_model_files()[-1])
            use_vlm = gr.Checkbox(
                label="VLM(OpenAI Vision)" + ("" if key_present else " — 未偵測到 OPENAI_API_KEY"),
                value=False, interactive=key_present,
            )

            iou_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="IoU 配對門檻")
            vote_dd = gr.Dropdown(choices=cc.VOTE_RULES, value=cc.VOTE_STRICT, label="投票規則")
            merge_dd = gr.Dropdown(choices=cc.MERGE_MODES, value=cc.MERGE_HIGHEST_CONF,
                                   label="座標合併方式")

            run_btn = gr.Button("開始比對", variant="primary")
            status_txt = gr.Textbox(label="統計報告", lines=12)

        with gr.Column(scale=2):
            gallery = gr.Gallery(label="共識標註預覽(紅框=_HARD 待複核,綠框=正常)",
                                 columns=3, height=500)

    run_btn.click(
        fn=run_batch,
        inputs=[folder_input, use_m1, model1_dd, use_m2, model2_dd, use_vlm,
                iou_slider, vote_dd, merge_dd],
        outputs=[status_txt, gallery],
    )
