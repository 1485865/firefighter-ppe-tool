import gradio as gr
import os
import shutil
import datetime
import torch
from PIL import Image
import requests
from io import BytesIO
# diffusers / openai / google-genai 改成「用到才載入」(延遲匯入)。
# 這些是重型/選用套件,沒裝的話不該害整個工作站開不了機。

# ==========================================
# 系統偵測 & 路徑設定
# ==========================================
GEN_DIR ="generated_images"# 生成圖自動存這裡
TRANSFER_DIR ="transfer_to_tab3"# 與抽幀頁共用的傳送門 → 標註/蒸餾頁可「接收」

# 本地已下載、可直接用的 SD1.5 底模(從 HF cache 偵測到的)
LOCAL_MODELS = {
    "Realistic Vision V5.1（寫實・推薦）":"SG161222/Realistic_Vision_V5.1_noVAE",
    "Stable Diffusion v1.5（通用）":"runwayml/stable-diffusion-v1-5",
}

# PPE 資料集情境 preset:幫你快速生稀有類別的合成圖
PROMPT_PRESETS = {
    "(自訂)":"",
    "消防員戴歐式消防帽+手套(全身)":
        "a firefighter in full turnout gear wearing a white European fire helmet "
        "(Gallet F1 style) and protective gloves, full body, standing, realistic "
        "photo, high detail, daylight",
    "消防員沒戴消防帽":
        "a firefighter without helmet, bare head visible, wearing turnout jacket, "
        "fireground background, realistic photo, high detail",
    "人員沒戴手套(手部特寫)":
        "a worker's bare hands without gloves, close up on hands, realistic photo, "
        "high detail",
    "多名消防員現場":
        "several firefighters at an emergency scene, each wearing fire helmets and "
        "gloves, realistic photo, high detail",
}

DEFAULT_NEG = ("lowres, bad anatomy, bad hands, extra fingers, blurry, deformed, "
               "watermark, text, cartoon, 3d render")


def get_local_device():
    if torch.cuda.is_available():
        return "cuda", f"本地端運算 (硬體加速: {torch.cuda.get_device_name(0)})"
    return "cpu","本地端運算 (CPU 模式)"


DEVICE, LOCAL_ENGINE_NAME = get_local_device()

# pipe 依「模型 id」快取,換模型不用重載已載過的
_PIPES = {}


# ==========================================
# 共用工具
# ==========================================
def _save_image(img):
    os.makedirs(GEN_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(GEN_DIR, f"gen_{ts}.png")
    img.save(path)
    return path


def send_to_pipeline(paths):
    """把這批生成圖送進傳送門,讓「批次預標註 / 模型蒸餾」頁可以接收。"""
    if not paths:
        return "沒有可傳送的圖片,請先生成。"
    shutil.rmtree(TRANSFER_DIR, ignore_errors=True)
    os.makedirs(TRANSFER_DIR, exist_ok=True)
    for p in paths:
        if os.path.exists(p):
            shutil.copy(p, os.path.join(TRANSFER_DIR, os.path.basename(p)))
    return (f"已傳送 {len(paths)} 張到傳送門。請到「批次標註與模型蒸餾」頁,"
            f"按「接收」即可用雙老師標註後拿去訓練。")


# ==========================================
# 各引擎:皆回傳 list[PIL.Image]
# ==========================================
def local_generate(prompt, neg, n, model_id, steps, guidance, size, seed, progress):
    from diffusers import StableDiffusionPipeline # 延遲匯入
    pipe = _PIPES.get(model_id)
    if pipe is None:
        if progress:
            progress(0.0, desc=f"載入本地模型 {model_id} ...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if DEVICE =="cuda" else torch.float32,
            safety_checker=None,
        ).to(DEVICE)
        pipe.set_progress_bar_config(disable=True)
        _PIPES[model_id] = pipe

    w, h = (int(x) for x in size.lower().split("x"))
    imgs = []
    for i in range(int(n)):
        if progress:
            progress(i / max(1, n), desc=f"本地 GPU 生成中 {i + 1}/{n} ...")
        gen = None
        if seed is not None and int(seed) >= 0:
            gen = torch.Generator(device=DEVICE).manual_seed(int(seed) + i)
        img = pipe(prompt, negative_prompt=neg or None,
                   num_inference_steps=int(steps), guidance_scale=float(guidance),
                   width=w, height=h, generator=gen).images[0]
        imgs.append(img)
    return imgs


def openai_generate(prompt, key, n, size):
    from openai import OpenAI # 延遲匯入
    client = OpenAI(api_key=key)
    # DALL-E 3 每次只能 n=1,且尺寸有限制,這裡固定 1024x1024
    imgs = []
    for _ in range(int(n)):
        resp = client.images.generate(model="dall-e-3", prompt=prompt,
                                       size="1024x1024", quality="standard", n=1)
        r = requests.get(resp.data[0].url, timeout=60)
        imgs.append(Image.open(BytesIO(r.content)).convert("RGB"))
    return imgs


def google_generate(prompt, key, n):
    from google import genai # 延遲匯入
    client = genai.Client(api_key=key)
    result = client.models.generate_images(
        model="imagen-3.0-generate-002", prompt=prompt,
        config=dict(number_of_images=min(int(n), 4), aspect_ratio="1:1",
                    output_mime_type="image/jpeg"))
    return [Image.open(BytesIO(g.image.image_bytes)).convert("RGB")
            for g in result.generated_images]


# ==========================================
# 算力路由器(generator:邊生邊回報 + 自動存檔)
# ==========================================
def route_generation(prompt, neg, engine, api_key, model_label, num_images,
                     steps, guidance, size, seed, progress=gr.Progress()):
    gallery, paths = [], []
    if not prompt or not prompt.strip():
        yield "請先輸入正面咒語 (Prompt)。", gallery, paths
        return
    try:
        if engine == LOCAL_ENGINE_NAME:
            model_id = LOCAL_MODELS.get(model_label, list(LOCAL_MODELS.values())[0])
            imgs = local_generate(prompt, neg, num_images, model_id,
                                  steps, guidance, size, seed, progress)
            tag ="本地 SD"
        elif "OpenAI" in engine:
            if not api_key:
                yield "請輸入 OpenAI API Key。", gallery, paths
                return
            progress(0.3, desc="連線 OpenAI DALL-E 3 ...")
            imgs = openai_generate(prompt, api_key, num_images, size)
            tag ="DALL-E 3"
        elif "Google" in engine:
            if not api_key:
                yield "請輸入 Google API Key。", gallery, paths
                return
            progress(0.3, desc="連線 Google Imagen 3 ...")
            imgs = google_generate(prompt, api_key, num_images)
            tag ="Imagen 3"
        else:
            yield "未知引擎。", gallery, paths
            return

        for img in imgs:
            p = _save_image(img)
            paths.append(p)
            gallery.append((p, os.path.basename(p)))
        progress(1.0, desc="完成")
        yield (f"{tag} 生成 {len(paths)} 張完成,已自動存到 {GEN_DIR}/。"
               f"按「傳送」即可拿去標註訓練。"), gallery, paths
    except Exception as e:
        yield f"{engine} 生成出錯: {e}", gallery, paths


# ==========================================
# 前端 UI
# ==========================================
def render_ui():
    gr.Markdown("### AI 合成圖生成(餵回 PPE 資料流)")
    gr.Markdown("用 AI 生成消防/PPE 情境圖補稀有類別 → 自動存檔 → 一鍵送進"
                "「批次標註與模型蒸餾」頁,讓雙老師自動標註後拿去訓練。\n\n"
                "合成圖與真實火場仍有落差,建議**少量混入**真實資料、別整批取代。")

    gen_paths = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            engine_choice = gr.Dropdown(
                choices=[LOCAL_ENGINE_NAME,"OpenAI (DALL-E 3)","Google Gemini (Imagen 3)"],
                value=LOCAL_ENGINE_NAME, label="選擇算力來源")
            api_key_input = gr.Textbox(label="API Key", placeholder="雲端引擎才需要...",
                                       visible=False, type="password")
            local_model = gr.Dropdown(choices=list(LOCAL_MODELS.keys()),
                                      value=list(LOCAL_MODELS.keys())[0],
                                      label="本地模型", visible=True)

            preset = gr.Dropdown(choices=list(PROMPT_PRESETS.keys()), value="(自訂)",
                                 label="PPE 情境範本(選了會帶入下方咒語)")
            prompt_input = gr.Textbox(label="正面咒語 (Positive Prompt)", lines=4)
            neg_prompt_input = gr.Textbox(label="負面咒語 (Negative Prompt)",
                                          value=DEFAULT_NEG, lines=2)

            with gr.Row():
                num_images = gr.Slider(1, 12, value=4, step=1, label="生成張數")
                seed_input = gr.Number(value=-1, label="種子 (-1=隨機)", precision=0)
            with gr.Row():
                steps = gr.Slider(10, 50, value=25, step=1, label="步數 steps", visible=True)
                guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5,
                                     label="引導 guidance", visible=True)
            size = gr.Dropdown(choices=["512x512","640x640","768x768"],
                               value="640x640", label="尺寸(本地)", visible=True)

            gen_btn = gr.Button("執行生成", variant="primary")
            send_btn = gr.Button("傳送這批到標註/蒸餾頁", variant="secondary")

        with gr.Column(scale=1):
            status_box = gr.Textbox(label="系統訊息", lines=2)
            result_gallery = gr.Gallery(label="生成結果(已自動存檔)",
                                        columns=3, height=460, allow_preview=True)

    # 範本帶入咒語
    preset.change(fn=lambda k: PROMPT_PRESETS.get(k,""), inputs=preset, outputs=prompt_input)

    # 動態顯示:雲端顯示 Key、隱藏本地專屬參數;本地反之
    def toggle_fields(choice):
        is_cloud = ("OpenAI" in choice) or ("Google" in choice)
        return (gr.update(visible=is_cloud), # api key
                gr.update(visible=not is_cloud), # local model
                gr.update(visible=not is_cloud), # neg prompt
                gr.update(visible=not is_cloud), # steps
                gr.update(visible=not is_cloud), # guidance
                gr.update(visible=not is_cloud)) # size
    engine_choice.change(
        fn=toggle_fields, inputs=engine_choice,
        outputs=[api_key_input, local_model, neg_prompt_input, steps, guidance, size])

    gen_btn.click(
        fn=route_generation,
        inputs=[prompt_input, neg_prompt_input, engine_choice, api_key_input,
                local_model, num_images, steps, guidance, size, seed_input],
        outputs=[status_box, result_gallery, gen_paths])

    send_btn.click(fn=send_to_pipeline, inputs=[gen_paths], outputs=[status_box])
