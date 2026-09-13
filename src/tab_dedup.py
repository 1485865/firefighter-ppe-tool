import gradio as gr
import os
from dedup_core import feature_dedup

def process_dedup(files, threshold, progress=gr.Progress()):
    if not files: return [],"請先上傳圖片"

    img_paths = [f.name for f in files]
    unique_images, removed = feature_dedup(img_paths, threshold, progress)

    msg = f"極速去重完成！\n原始數量: {len(files)} 張\n過濾後剩下: {len(unique_images)} 張 (排除了 {removed} 張廢片)。"

    gallery_format = [(p, os.path.basename(p)) for p in unique_images]

    return gallery_format, msg

def render_ui():
    gr.Markdown("## 實驗室 V2：AI 特徵向量去重 (Embedding Deduplication)")
    gr.Markdown("使用 MobileNetV2 提取高維度特徵，並透過 **餘弦相似度 (Cosine Similarity)** 計算差異。這比傳統的比對像素更強大，連輕微位移、縮放的連拍廢片都能精準剔除！")
    
    with gr.Row():
        with gr.Accordion("展開/折疊上傳的圖片清單", open=False):
            upload_imgs = gr.File(label="丟入一整批疑似重複的連拍圖片", file_count="multiple")

            sim_threshold = gr.Slider(minimum=80, maximum=100, value=95, step=1, label="相似度判定閥值 (%) (越高越嚴格，越容易判定為不同圖片)")
            run_btn = gr.Button("開始 AI 深度去重", variant="primary")
            status_txt = gr.Textbox(label="運算報告")
            
        with gr.Column(scale=2):
            result_gallery = gr.Gallery(label="淬鍊後的唯一精華圖片", columns=4, height=400)
            
    run_btn.click(fn=process_dedup, inputs=[upload_imgs, sim_threshold], outputs=[result_gallery, status_txt])