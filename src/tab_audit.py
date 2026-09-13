"""標註品質稽核 UI。"""
import os
import time

import cv2
import gradio as gr

import audit_core as ac
from vlm_annotate_core import CLASS_NAME_TO_ID

TEMP_DIR = os.path.join(os.getcwd(), "temp", "audit")
os.makedirs(TEMP_DIR, exist_ok=True)


def run_audit(folder, area_max, area_min, progress=gr.Progress()):
    if not folder or not os.path.isdir(folder):
        empty = gr.update(value=[])
        return ("找不到資料夾,請確認路徑正確。", empty, empty, None, None,
                [], [])

    result = ac.audit_dataset(folder, area_max_pct=area_max, area_min_pct=area_min,
                              progress=progress)

    missing_rows = [
        [os.path.basename(r["image"]), r["person"], r["hands"], r["expected_hands"],
         r["heads"], r["expected_heads"]]
        for r in result["missing_labels"]
    ]
    anomaly_rows = [
        [os.path.basename(img), line, reason]
        for img, line, reason in result["anomaly_boxes"]
    ]

    fig_counts, fig_areas = ac.build_stats_figures(
        result["class_counts"], result["class_areas"])

    avg_instances = (sum(result["per_image_counts"]) / len(result["per_image_counts"])
                     if result["per_image_counts"] else 0)
    summary = (
        f"共稽核 {result['total_images']} 張圖(讀取失敗 {len(result['failed'])} 張)。\n"
        f"疑似漏標: {len(missing_rows)} 張\n"
        f"異常框: {len(anomaly_rows)} 個\n"
        f"平均每張圖實例數: {avg_instances:.2f}"
    )
    if result["failed"]:
        summary += "\n\n讀取失敗清單:\n" + "\n".join(
            f"{os.path.basename(p)}: {e}" for p, e in result["failed"][:30])

    missing_paths = [r["image"] for r in result["missing_labels"]]
    anomaly_paths = [img for img, _line, _reason in result["anomaly_boxes"]]

    return summary, missing_rows, anomaly_rows, fig_counts, fig_areas, missing_paths, anomaly_paths


def export_missing(missing_paths):
    if not missing_paths:
        return None
    out_path = os.path.join(TEMP_DIR, f"missing_{int(time.time())}.txt")
    ac.export_filelist(missing_paths, out_path)
    return out_path


def export_anomaly(anomaly_paths):
    if not anomaly_paths:
        return None
    out_path = os.path.join(TEMP_DIR, f"anomaly_{int(time.time())}.txt")
    ac.export_filelist(anomaly_paths, out_path)
    return out_path


def run_sample(folder, cls_name, n):
    crops = ac.sample_class_crops(folder, cls_name, int(n))
    return [(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption) for crop, caption in crops]


def render_ui():
    missing_paths_state = gr.State([])
    anomaly_paths_state = gr.State([])

    gr.Markdown("## 標註品質稽核")
    gr.Markdown("找出標註錯誤和漏標,支援 images/+labels/ 標準版型或同資料夾同檔名版型。")

    with gr.Row():
        folder_input = gr.Textbox(label="資料集資料夾路徑", value="extracted_frames")
        area_max_slider = gr.Slider(10, 100, value=60, step=5, label="框過大門檻(佔畫面 %)")
        area_min_slider = gr.Slider(0.01, 5, value=0.1, step=0.01, label="框過小門檻(佔畫面 %)")
    run_btn = gr.Button("執行稽核", variant="primary")

    with gr.Tabs():
        with gr.TabItem("總覽"):
            summary_txt = gr.Textbox(label="稽核摘要", lines=8)

        with gr.TabItem("漏標清單"):
            missing_df = gr.Dataframe(
                headers=["圖片", "person 數", "手部框數", "預期手部框數",
                        "頭部框數", "預期頭部框數"],
                label="疑似漏標(依缺口大小排序)",
            )
            export_missing_btn = gr.Button("匯出檔名清單(txt)")
            export_missing_file = gr.File(label="下載")

        with gr.TabItem("異常框清單"):
            anomaly_df = gr.Dataframe(
                headers=["圖片", "第幾個框", "異常原因"],
                label="異常框清單",
            )
            export_anomaly_btn = gr.Button("匯出檔名清單(txt)")
            export_anomaly_file = gr.File(label="下載")

        with gr.TabItem("統計圖表"):
            with gr.Row():
                counts_plot = gr.Plot(label="各類別實例數")
                areas_plot = gr.Plot(label="各類別框面積分布")

        with gr.TabItem("一致性抽樣比對"):
            with gr.Row():
                sample_cls_dd = gr.Dropdown(
                    choices=list(CLASS_NAME_TO_ID.keys()), value="Gloves", label="類別")
                sample_n_slider = gr.Slider(4, 40, value=12, step=1, label="抽樣數量")
                sample_btn = gr.Button("抽樣")
            sample_gallery = gr.Gallery(label="抽樣裁切圖(目視比對標註標準是否一致)",
                                        columns=6, height=400)

    run_btn.click(
        fn=run_audit,
        inputs=[folder_input, area_max_slider, area_min_slider],
        outputs=[summary_txt, missing_df, anomaly_df, counts_plot, areas_plot,
                missing_paths_state, anomaly_paths_state],
    )

    export_missing_btn.click(fn=export_missing, inputs=[missing_paths_state],
                             outputs=[export_missing_file])
    export_anomaly_btn.click(fn=export_anomaly, inputs=[anomaly_paths_state],
                             outputs=[export_anomaly_file])

    sample_btn.click(fn=run_sample, inputs=[folder_input, sample_cls_dd, sample_n_slider],
                     outputs=[sample_gallery])
