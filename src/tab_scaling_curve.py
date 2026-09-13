"""Data Scaling Curve UI:量化「資料量 vs 模型效能」的邊際效益。"""
import time

import gradio as gr

import scaling_curve_core as scc
from tab_distill import student_choices


def parse_fractions(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            f = float(part)
        except ValueError:
            continue
        if 0 < f <= 1.0:
            out.append(f)
    return sorted(set(out))


def run_curve(data_yaml, model_base, epochs, imgsz, batch, fractions_text,
              target_map, progress=gr.Progress()):
    if not data_yaml:
        yield "請填 data.yaml 路徑。", [], None, None, ""
        return

    fractions = parse_fractions(fractions_text)
    if not fractions:
        yield "fraction 清單無效,請填類似 0.25,0.5,0.75,1.0 的格式。", [], None, None, ""
        return

    run_name = f"run_{int(time.time())}"
    results = []
    for log_msg, results in scc.run_scaling_curve(
            data_yaml, model_base, fractions, epochs, imgsz, batch, run_name,
            progress=progress):
        rows = [[r["fraction"], f"{r['overall_map50']:.4f}"] for r in results]
        fig_overall, fig_per_class = scc.build_curve_figures(results)
        yield log_msg, rows, fig_overall, fig_per_class, ""

    total_images = scc.estimate_train_image_count(data_yaml)
    extrapolation = scc.extrapolate_target(results, target_map, total_images)
    rows = [[r["fraction"], f"{r['overall_map50']:.4f}"] for r in results]
    fig_overall, fig_per_class = scc.build_curve_figures(results)
    yield "全部訓練完成。", rows, fig_overall, fig_per_class, extrapolation


def render_ui():
    gr.Markdown("## Data Scaling Curve")
    gr.Markdown(
        "同一份資料集用不同資料量比例各訓練一輪,量化「資料量 vs 模型效能」的邊際效益,"
        "回答「還需要多少資料」。這會跑很久,請耐心等候;中斷後重跑相同參數會自動跳過已完成的 fraction。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            data_yaml_input = gr.Textbox(
                label="資料集 data.yaml 路徑",
                placeholder="例如 runs_distill/student_distilled/data.yaml",
            )
            model_dd = gr.Dropdown(
                choices=student_choices(), value="yolo11n.pt", label="訓練底模",
            )
            with gr.Row():
                epochs_num = gr.Number(value=50, label="epochs", precision=0)
                imgsz_num = gr.Number(value=640, label="imgsz", precision=0)
                batch_num = gr.Number(value=16, label="batch", precision=0)
            gr.Markdown("workers 固定為 0(避免 pin memory 的 CUDA 錯誤),不開放調整。")
            fractions_input = gr.Textbox(
                value="0.25,0.5,0.75,1.0", label="fraction 清單(逗號分隔)")
            target_map_slider = gr.Slider(0.5, 0.99, value=0.8, step=0.01,
                                          label="目標 mAP50(外推估算用)")
            run_btn = gr.Button("開始", variant="primary")

        with gr.Column(scale=2):
            log_txt = gr.Textbox(label="進度紀錄", lines=6)
            results_df = gr.Dataframe(headers=["fraction", "overall mAP50"],
                                      label="各輪結果")
            with gr.Row():
                overall_plot = gr.Plot(label="整體 mAP50 對資料量")
                per_class_plot = gr.Plot(label="逐類別 mAP50 對資料量")
            extrapolation_txt = gr.Textbox(label="外推估算(還需要多少資料)", lines=6)

    run_btn.click(
        fn=run_curve,
        inputs=[data_yaml_input, model_dd, epochs_num, imgsz_num, batch_num,
                fractions_input, target_map_slider],
        outputs=[log_txt, results_df, overall_plot, per_class_plot, extrapolation_txt],
    )
