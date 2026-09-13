"""VLM 自動標註分頁: 用 OpenAI Vision API 對圖片資料夾做預標註,
輸出 YOLO .txt(+ 附帶 _HARD 複核用 .xml,慣例同 annotate_core)。

API key 一律從環境變數 OPENAI_API_KEY 讀取,絕不寫死/存進 config.json。
"""
import os
import glob

import gradio as gr

from vlm_annotate_core import annotate_batch, VLM_MODEL

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_images(folder):
    if not folder or not os.path.isdir(folder):
        return []
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(paths))


def _run_annotation(folder, conf_threshold, max_workers, limit, cumulative_cost, progress):
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return ("尚未偵測到環境變數 OPENAI_API_KEY,請先設定後重新啟動本工具"
                 "(Key 絕不會寫進程式碼或設定檔)。",
                "", cumulative_cost, f"US$ {cumulative_cost:.4f}")

    images = _list_images(folder)
    if not images:
        return (f"資料夾內找不到圖片: {folder}", "", cumulative_cost,
                f"US$ {cumulative_cost:.4f}")

    if limit:
        images = images[:limit]

    total = len(images)
    progress(0, desc=f"準備標註 {total} 張圖片...")

    def on_progress(done, tot, name):
        progress(done / tot, desc=f"VLM 標註中... ({done}/{tot}) {name}")

    results, failed = annotate_batch(
        images, api_key, conf_threshold,
        max_workers=int(max_workers), progress_cb=on_progress,
    )

    total_boxes = sum(r["box_count"] for r in results)
    total_hard = sum(r["hard_count"] for r in results)
    run_cost = sum(r["cost"] for r in results)
    new_cumulative = cumulative_cost + run_cost

    summary = (
        f"成功: {len(results)} / {total}　失敗: {len(failed)}\n"
        f"總框數: {total_boxes}　_HARD 待複查框數: {total_hard}\n"
        f"本次估算花費: 約 US$ {run_cost:.4f}(model={VLM_MODEL})\n"
        f"已寫出 .txt(YOLO,可直接訓練)與 .xml(含 _HARD 標記,供人工複核)"
    )
    failed_list = ("\n".join(f"{os.path.basename(p)}: {err}" for p, err in failed)
                   if failed else "(無失敗)")

    return summary, failed_list, new_cumulative, f"US$ {new_cumulative:.4f}"


def render_ui():
    cumulative_cost_state = gr.State(0.0)

    gr.Markdown("### VLM 自動標註(OpenAI Vision 預標註 → YOLO .txt)")

    key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    gr.Markdown(
        "已偵測到環境變數 `OPENAI_API_KEY`,可以開始標註。" if key_present else
        "**尚未偵測到環境變數 `OPENAI_API_KEY`**。"
        "請在系統環境變數設定好之後,重新啟動本工具再使用這個分頁"
        "(Key 絕對不會被寫進程式碼或 config.json)。"
    )
    gr.Markdown(
        "先找手、找頭,再分類「戴/沒戴」的兩段式標註流程"
        "(VLM 對『找出所有的手』很強,但對直接找『沒戴手套的手』這種缺席類別很弱,"
        "所以固定用這套 prompt,不要改)。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            folder_input = gr.Textbox(
                label="圖片資料夾路徑",
                placeholder="例如: extracted_frames",
                value="extracted_frames",
            )
            conf_slider = gr.Slider(
                0.0, 1.0, value=0.5, step=0.05,
                label="信心門檻(低於此值的框,XML 類別加上 _HARD 待人工複核)",
            )
            workers_slider = gr.Slider(
                1, 16, value=4, step=1,
                label="並發數(ThreadPoolExecutor max_workers)",
            )
            with gr.Row():
                trial_btn = gr.Button("試跑 5 張", variant="secondary")
                full_btn = gr.Button("全部標註", variant="primary")
            reset_cost_btn = gr.Button("重設累計花費估算", size="sm", variant="stop")

        with gr.Column(scale=1):
            summary_box = gr.Textbox(label="結果摘要", lines=6, interactive=False)
            failed_box = gr.Textbox(label="失敗檔名清單", lines=8, interactive=False)
            cumulative_box = gr.Textbox(
                label="累計 API 花費估算(本次工具啟動以來,粗估僅供參考)",
                value="US$ 0.0000", interactive=False,
            )

    def run_trial(folder, conf_thr, workers, cum_cost, progress=gr.Progress()):
        return _run_annotation(folder, conf_thr, workers, 5, cum_cost, progress)

    def run_full(folder, conf_thr, workers, cum_cost, progress=gr.Progress()):
        return _run_annotation(folder, conf_thr, workers, None, cum_cost, progress)

    trial_btn.click(
        fn=run_trial,
        inputs=[folder_input, conf_slider, workers_slider, cumulative_cost_state],
        outputs=[summary_box, failed_box, cumulative_cost_state, cumulative_box],
    )

    full_btn.click(
        fn=run_full,
        inputs=[folder_input, conf_slider, workers_slider, cumulative_cost_state],
        outputs=[summary_box, failed_box, cumulative_cost_state, cumulative_box],
    )

    reset_cost_btn.click(
        fn=lambda: (0.0, "US$ 0.0000"),
        inputs=None,
        outputs=[cumulative_cost_state, cumulative_box],
    )
