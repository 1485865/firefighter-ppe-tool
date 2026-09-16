"""標註品質稽核 UI。"""
import os
import time

import cv2
import gradio as gr

import audit_core as ac
import audit_excel_export as aee
from vlm_annotate_core import CLASS_NAME_TO_ID

TEMP_DIR = os.path.join(os.getcwd(), "temp", "audit")
os.makedirs(TEMP_DIR, exist_ok=True)


def run_audit(folder, area_max, area_min, progress=gr.Progress()):
    if not folder or not os.path.isdir(folder):
        empty = gr.update(value=[])
        return ("找不到資料夾,請確認路徑正確。", empty, empty, None, None,
                [], [], None)

    result = ac.audit_dataset(folder, area_max_pct=area_max, area_min_pct=area_min,
                              progress=progress)

    missing_rows = [
        [os.path.basename(r["image"]), r["person"], r["hands"], r["expected_hands"],
         r["heads"], r["expected_heads"]]
        for r in result["missing_labels"]
    ]
    anomaly_rows = [
        [os.path.basename(a["image"]), a["box_index"], a["detail"]]
        for a in result["anomaly_boxes"]
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
    anomaly_paths = [a["image"] for a in result["anomaly_boxes"]]

    return (summary, missing_rows, anomaly_rows, fig_counts, fig_areas,
             missing_paths, anomaly_paths, result)


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


def export_excel(result, embed_limit, batch_export, progress=gr.Progress()):
    if not result:
        return None, "請先執行稽核。"

    embed_limit = int(embed_limit) if embed_limit else aee.DEFAULT_EMBED_LIMIT
    base_name = os.path.join(TEMP_DIR, f"audit_{time.strftime('%Y%m%d_%H%M%S')}")

    def cb(frac, desc=""):
        progress(frac, desc=desc)

    if batch_export:
        stats = aee.export_excel_batches(result, TEMP_DIR, base_name,
                                         embed_limit=embed_limit, progress_cb=cb)
        msg = (
            f"已分成 {stats['batch_count']} 個批次(每批最多 200 筆),打包成 zip。\n"
            f"漏標清單縮圖:嵌入 {stats['missing_embedded']} 張,讀取失敗 {stats['missing_failed']} 張\n"
            f"異常框清單縮圖:嵌入 {stats['anomaly_embedded']} 張,讀取失敗 {stats['anomaly_failed']} 張\n"
            f"檔案大小:{stats['file_size'] / 1024 / 1024:.2f} MB"
        )
    else:
        out_path = base_name + ".xlsx"
        stats = aee.export_excel(result, out_path, embed_limit=embed_limit, progress_cb=cb)
        msg = (
            f"漏標清單縮圖:嵌入 {stats['missing_embedded']} 張,讀取失敗 {stats['missing_failed']} 張\n"
            f"異常框清單縮圖:嵌入 {stats['anomaly_embedded']} 張,讀取失敗 {stats['anomaly_failed']} 張\n"
            f"檔案大小:{stats['file_size'] / 1024 / 1024:.2f} MB"
        )

    return stats["path"], msg


def run_sample(folder, cls_name, n):
    crops = ac.sample_class_crops(folder, cls_name, int(n))
    return [(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption) for crop, caption in crops]


def render_ui():
    missing_paths_state = gr.State([])
    anomaly_paths_state = gr.State([])
    audit_result_state = gr.State(None)

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
            with gr.Row():
                embed_limit_input = gr.Number(
                    value=aee.DEFAULT_EMBED_LIMIT, precision=0, minimum=0,
                    label="匯出筆數上限(每個清單最多嵌幾張縮圖,依嚴重度取前 N 筆)")
                batch_export_checkbox = gr.Checkbox(
                    value=False, label="分批匯出(每 200 筆一個檔案,打包成 zip,方便分給不同人審)")
            export_excel_btn = gr.Button("匯出 Excel", variant="primary")
            export_excel_status = gr.Textbox(label="匯出結果", lines=4, interactive=False)
            export_excel_file = gr.File(label="下載 Excel 報表")

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
                missing_paths_state, anomaly_paths_state, audit_result_state],
    )

    export_missing_btn.click(fn=export_missing, inputs=[missing_paths_state],
                             outputs=[export_missing_file])
    export_anomaly_btn.click(fn=export_anomaly, inputs=[anomaly_paths_state],
                             outputs=[export_anomaly_file])
    export_excel_btn.click(fn=export_excel,
                           inputs=[audit_result_state, embed_limit_input, batch_export_checkbox],
                           outputs=[export_excel_file, export_excel_status])

    sample_btn.click(fn=run_sample, inputs=[folder_input, sample_cls_dd, sample_n_slider],
                     outputs=[sample_gallery])
