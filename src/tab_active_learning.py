"""主動學習排序:對未標註圖片池評分排序,把最值得標的圖挑到前面。"""
import os
import time

import gradio as gr

from active_learning_core import rank_images, export_filelist
from utils import list_models as get_model_files, get_yolo_model

TEMP_DIR = os.path.join(os.getcwd(), "temp", "active_learning")
_MAX_AGE_DAYS = 7


def _cleanup_old_exports():
    if not os.path.isdir(TEMP_DIR):
        return
    cutoff = time.time() - _MAX_AGE_DAYS * 86400
    for name in os.listdir(TEMP_DIR):
        path = os.path.join(TEMP_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            continue


os.makedirs(TEMP_DIR, exist_ok=True)
_cleanup_old_exports()


def run_ranking(folder, model_path, w_unc, w_cls, w_den, use_overlap, w_ovl,
                progress=gr.Progress()):
    if not folder or not os.path.isdir(folder):
        return [], [], "找不到資料夾,請確認路徑正確。", gr.update(choices=[], value=[])
    if not model_path or "找不到" in model_path:
        return [], [], "請選擇有效的模型。", gr.update(choices=[], value=[])

    try:
        model = get_yolo_model(model_path)
    except Exception as e:
        return [], [], f"模型載入失敗: {e}", gr.update(choices=[], value=[])

    weights = {
        "uncertainty": w_unc,
        "class_weight": w_cls,
        "density": w_den,
        "overlap": w_ovl if use_overlap else 0.0,
    }
    ranked, failed = rank_images(folder, model, weights, progress)

    gallery_items = [
        (r["path"], f"#{i+1} 分數 {r['score']:.1f} | 框數 {r['box_count']} "
                    f"(不確定 {r['uncertainty']} / 手部 {r['class_weight']})")
        for i, r in enumerate(ranked)
    ]
    paths = [r["path"] for r in ranked]

    msg = f"評分完成!共排序 {len(ranked)} 張圖(失敗 {len(failed)} 張)。"
    if failed:
        msg += "\n\n失敗清單:\n" + "\n".join(
            f"{os.path.basename(p)}: {err}" for p, err in failed[:30])

    return gallery_items, paths, msg, gr.update(choices=[], value=[])


def select_top_n(ranked_paths, n):
    if not ranked_paths:
        return gr.update(value=[])
    n = max(0, int(n))
    names = [os.path.basename(p) for p in ranked_paths[:n]]
    return gr.update(value=names)


def export_selected(selected_names, ranked_paths):
    if not selected_names:
        return None, "請先勾選要匯出的圖片。"
    name_to_path = {os.path.basename(p): p for p in ranked_paths}
    chosen = [name_to_path[n] for n in selected_names if n in name_to_path]
    if not chosen:
        return None, "選取的圖片清單已經失效,請重新計算排序。"
    os.makedirs(TEMP_DIR, exist_ok=True)
    out_path = os.path.join(TEMP_DIR, f"active_learning_{int(time.time())}.txt")
    export_filelist(chosen, out_path)
    return out_path, f"已匯出 {len(chosen)} 個檔名到清單。"


def render_ui():
    ranked_paths_state = gr.State([])

    gr.Markdown("## 主動學習排序(Active Learning)")
    gr.Markdown("對未標註圖片池評分,把「標了最有價值」的圖排到前面,取代目前隨機選圖的做法。")

    with gr.Row():
        with gr.Column(scale=1):
            folder_input = gr.Textbox(
                label="未標註圖片資料夾路徑", value="extracted_frames",
                placeholder="例如: extracted_frames",
            )
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=get_model_files(), label="選擇 YOLO 模型",
                    value=get_model_files()[0], interactive=True,
                )
                refresh_btn = gr.Button("", scale=0)

            gr.Markdown("### 分數權重")
            w_uncertainty = gr.Slider(0, 5, value=1.0, step=0.1,
                                      label="不確定性權重(信心 0.3~0.6 的框數)")
            w_class = gr.Slider(0, 5, value=1.0, step=0.1,
                                label="類別加權(Gloves/No_Gloves 短板類別)")
            w_density = gr.Slider(0, 5, value=1.0, step=0.1,
                                  label="密度權重(單張圖手部框總數)")
            use_overlap = gr.Checkbox(label="啟用重疊異常加權(可選)", value=False)
            w_overlap = gr.Slider(0, 5, value=1.0, step=0.1,
                                  label="重疊異常權重(不同類別框重疊代表模型混亂)",
                                  visible=False)
            use_overlap.change(fn=lambda x: gr.update(visible=x),
                               inputs=[use_overlap], outputs=[w_overlap], queue=False)

            run_btn = gr.Button("計算排序", variant="primary")
            status_txt = gr.Textbox(label="運算報告", lines=6)

        with gr.Column(scale=2):
            gallery = gr.Gallery(label="依分數排序的結果(由高到低)", columns=4, height=420)

            with gr.Row():
                top_n = gr.Number(value=20, label="快速全選前 N 張", precision=0)
                top_n_btn = gr.Button("套用")

            selected_dd = gr.Dropdown(
                label="已選取要匯出的圖片", choices=[], multiselect=True, interactive=True,
            )
            export_btn = gr.Button("匯出勾選清單(txt)", variant="secondary")
            export_file = gr.File(label="下載檔名清單")

    refresh_btn.click(fn=lambda: gr.update(choices=get_model_files()),
                      inputs=None, outputs=[model_dropdown])

    run_btn.click(
        fn=run_ranking,
        inputs=[folder_input, model_dropdown, w_uncertainty, w_class, w_density,
                use_overlap, w_overlap],
        outputs=[gallery, ranked_paths_state, status_txt, selected_dd],
    )

    top_n_btn.click(fn=select_top_n, inputs=[ranked_paths_state, top_n],
                    outputs=[selected_dd])

    export_btn.click(fn=export_selected, inputs=[selected_dd, ranked_paths_state],
                     outputs=[export_file, status_txt])
