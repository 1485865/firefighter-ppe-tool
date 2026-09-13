import gradio as gr
import os
import shutil
import tempfile
import tab_youtube
import tab_extractor
import tab_batch_annotate
import tab_augment
import tab_dedup
import tab_heatmap
import tab_tracking
import tab_distill
import tab_vlm_annotate
import dlc_image_gen
import gallery_scroll
import tab_active_learning
import tab_consensus
import tab_audit
import tab_scaling_curve
import updater

from config_manager import load_config, save_config

os.makedirs("downloads", exist_ok=True)
cfg = load_config()

# ==========================================
# 系統啟動前置作業：自動清理 Gradio 幽靈快取
# ==========================================
def clear_gradio_cache():
    try:
        # 抓出 Windows 的 Temp 暫存資料夾路徑
        temp_dir = tempfile.gettempdir()
        gradio_temp_path = os.path.join(temp_dir,'gradio')
        
        # 如果存在這個資料夾，就無情地把它整個刪掉
        if os.path.exists(gradio_temp_path):
            shutil.rmtree(gradio_temp_path, ignore_errors=True)
            print("[System] 啟動成功：已自動清除上次殘留的 Gradio 幽靈快取檔！")
        else:
            print("[System] 啟動成功：暫存區很乾淨，無需清理。")
    except Exception as e:
        print(f"[System] 警告：無法清理快取 ({e})，但系統仍可運行。")

# 在畫 UI 之前，強制執行清道夫！
clear_gradio_cache()

# 修正點：Gradio 6.0 規定這裡必須保持乾淨，不能放 theme
with gr.Blocks() as demo:
    gr.Markdown("#AI 數據工作站")
    # 全站通用:讓所有 Gallery 開啟預覽後,底部縮圖列可以捲軸拖曳/滾輪/按住拖曳
    gr.HTML(gallery_scroll.GALLERY_SCROLL_CSS)

    # ==========================================
    # 自動更新橫幅:頁面載入完才檢查(demo.load),不卡住主要功能的初始渲染。
    # 有更新只顯示提示,絕不自動更新,由使用者按按鈕確認才真的拉取。
    # ==========================================
    with gr.Row(visible=False) as update_banner:
        update_msg = gr.Markdown()
        update_btn = gr.Button("立即更新", variant="primary", scale=0)
    update_status = gr.Markdown()

    def _check_update_on_load():
        r = updater.check_for_update()
        if r["status"] == "update_available":
            commits = r.get("commits", "")
            return gr.update(visible=True), f"**發現新版本！**\n\n{commits}", ""
        return gr.update(visible=False), "", ""

    def _do_update_click():
        r = updater.do_update()
        return r["message"]

    demo.load(fn=_check_update_on_load, inputs=None,
              outputs=[update_banner, update_msg, update_status])
    update_btn.click(fn=_do_update_click, inputs=None, outputs=[update_status])

    with gr.Accordion("Roboflow API 設定", open=False):
            with gr.Row():
                global_api = gr.Textbox(label="API Key", value=cfg.get("api_key",""))
                global_ws = gr.Textbox(label="Workspace", value=cfg.get("workspace",""))
                global_proj = gr.Textbox(label="Project", value=cfg.get("project",""))
            
            # 新增：加一顆手動儲存按鈕，寫入 config.local.json(本機專用，不進版控)
            save_btn = gr.Button("儲存 API 設定", variant="secondary")
            
            # 綁定儲存按鈕的動作 (存檔成功後會在網頁右上角彈出綠色勾勾提示)
            save_btn.click(
                fn=save_config,
                inputs=[global_api, global_ws, global_proj],
                outputs=None
            ).then(fn=lambda: gr.Info("API 設定已永久儲存！"), inputs=None, outputs=None)
            
    with gr.Tabs():
        with gr.TabItem("YouTube 影片下載"):
            tab_youtube.render_ui()
            
        with gr.TabItem("影片抽幀與清洗"):
            tab_extractor.render_ui(global_api, global_ws, global_proj)
            
        with gr.TabItem("批次標註與模型蒸餾"):
            with gr.Tabs():
                with gr.TabItem("大量圖片批次預標註"):
                    tab_batch_annotate.render_ui(global_api, global_ws, global_proj)
                with gr.TabItem("YOLO 模型蒸餾"):
                    tab_distill.render_ui(global_api, global_ws, global_proj)
                with gr.TabItem("VLM 自動標註(OpenAI)"):
                    # 防護罩:openai 屬選用相依,萬一環境沒裝或載入失敗,
                    # 只讓這一頁顯示提示,不拖垮其他分頁。
                    try:
                        tab_vlm_annotate.render_ui()
                    except Exception as e:
                        gr.Markdown(
                            "### 此功能暫時無法載入\n"
                            "VLM 自動標註需要 `openai` 套件與環境變數 `OPENAI_API_KEY`,"
                            "目前偵測到載入失敗。\n\n"
                            f"**錯誤訊息：** `{e}`\n\n"
                            "其他分頁不受影響,可正常使用。"
                        )

        with gr.Tab("AI 實驗室"):
            with gr.Tabs():
                with gr.TabItem("火場環境增強模擬(V1)"):
                    tab_augment.render_ui()
                with gr.TabItem("模型解剖・熱力圖(V3)"):
                    tab_heatmap.render_ui()

        with gr.Tab("智慧去重V2(特徵向量)"):
            tab_dedup.render_ui()

        with gr.Tab("動態軌跡追蹤"):
            tab_tracking.render_ui()

        with gr.TabItem("AI 合成圖生成"):
            # 防護罩:這頁用到的生圖套件(diffusers/openai)屬選用且笨重,
            # 就算它整個壞掉,也只讓這一頁顯示提示,不該拖垮其他分頁。
            try:
                dlc_image_gen.render_ui()
            except Exception as e:
                gr.Markdown(
                    "### 此功能暫時無法載入\n"
                    "AI 生成圖片需要 `diffusers` / `openai` 等選用套件,目前偵測到載入失敗。\n\n"
                    f"**錯誤訊息：** `{e}`\n\n"
                    "其他分頁不受影響,可正常使用。"
                )

        with gr.Tab("資料集品質工具"):
            with gr.Tabs():
                with gr.TabItem("主動學習排序"):
                    tab_active_learning.render_ui()
                with gr.TabItem("跨模型共識標註"):
                    tab_consensus.render_ui()
                with gr.TabItem("標註品質稽核"):
                    tab_audit.render_ui()
                with gr.TabItem("Data Scaling Curve"):
                    tab_scaling_curve.render_ui()

def pick_free_port():
    """挑一個真的綁得上的埠。
    Windows 裝了 Hyper-V/WSL/Docker 後會「保留」一大段埠（含 Gradio 預設的
    7860~7959），綁這些埠會噴 WinError 10013。這裡逐一試綁，回傳第一個成功的；
    都失敗就讓作業系統自動配一個（port 0，系統不會配到保留範圍）。"""
    import socket
    for port in (8000, 8501, 9000, 9527, 10000):
        try:
            s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port)); s.close()
            return port
        except OSError:
            continue
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port

if __name__ =="__main__":
    port = pick_free_port()
    print(f"[System] 啟動於 http://127.0.0.1:{port}")
    # 注意:filmstrip 的 JS 不能用 gr.HTML() 注入(innerHTML 插入的 <script>
    # 瀏覽器天生不會執行),要放進真正的 <head> 才會跑,所以在這裡集中傳入。
    demo.launch(
        inbrowser=True, server_name="127.0.0.1", server_port=port,
        head=tab_extractor.FILMSTRIP_HEAD_JS + gallery_scroll.GALLERY_SCROLL_HEAD_JS,
    )