import gradio as gr
import yt_dlp
import subprocess
import os
import re

from notifier import send_alert
from utils import ffmpeg_path, ffprobe_path


# 更新版：加入 UTF-8 強制解碼與忽略亂碼防護
def ensure_h264_for_browser(video_path):
    if not video_path or not os.path.exists(video_path):
        return video_path,"找不到檔案"
        
    ffmpeg_cmd = ffmpeg_path()
    ffprobe_cmd = ffprobe_path()

    try:
        cmd_probe = [
            ffprobe_cmd,'-v','error',
            '-select_streams','v:0',
            '-show_entries','stream=codec_name',
            '-of','default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        # 關鍵改動 1：把 text=True 換成 encoding='utf-8', errors='ignore'
        result = subprocess.run(cmd_probe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', errors='ignore', creationflags=subprocess.CREATE_NO_WINDOW)
        codec = result.stdout.strip()
        
        # 網頁支援的格式，直接放行
        if codec in ['h264','avc1']:
            return video_path, f"原生支援網頁播放 ({codec})"
            
        temp_path = video_path.rsplit('.', 1)[0] +'_h264_converted.mp4'
        cmd_convert = [
            ffmpeg_cmd,'-i', video_path,
            '-c:v','libx264','-preset','fast','-crf','23',
            '-c:a','aac','-b:a','128k',
            '-y', temp_path
        ]
        
        # 關鍵改動 2：同樣換成 encoding='utf-8', errors='ignore'
        conv_result = subprocess.run(cmd_convert, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='ignore', creationflags=subprocess.CREATE_NO_WINDOW)
        
        # 檢查是否成功產出檔案
        if os.path.exists(temp_path):
            os.remove(video_path)
            os.rename(temp_path, video_path)
            return video_path, f"轉碼大成功！已將 {codec} 強制轉為 H.264，預覽已解鎖！"
        else:
            error_log = conv_result.stdout.strip()[-300:] if conv_result.stdout else "未知錯誤(無輸出，可能FFmpeg路徑錯誤或閃退)"
            return video_path, f"轉碼失敗 ({codec})。報錯原因: {error_log}"
            
    except Exception as e:
        return video_path, f"守衛發生系統錯誤: {str(e)}"

def download_yt(url, quality, progress=gr.Progress()):
    if not url: return "請輸入網址", None

    # 有些平台(B站/抖音等)分享時會把標題和網址黏在一起貼過來，這裡自動抽出純網址
    match = re.search(r'https?://\S+', url)
    if not match:
        return f"找不到有效的網址，請確認貼上的內容包含 http(s):// 開頭的連結。\n收到的內容: {url}", None
    url = match.group(0)

    os.makedirs("downloads", exist_ok=True)
    
    res_map = {
        "144p": 144,"240p": 240,"360p": 360,"480p": 480,
        "720p": 720,"1080p": 1080,"2K (1440p)": 1440,"4K (2160p)": 2160
    }
    
    if quality =="最高畫質 (自動)":
        fmt ='bestvideo[vcodec^=avc1]+bestaudio/bestvideo+bestaudio/best'
    else:
        h = res_map.get(quality, 1080)
        fmt = f'bestvideo[height<={h}][vcodec^=avc1]+bestaudio/bestvideo[height<={h}]+bestaudio/best'
        
   
    def my_hook(d):
        if d['status'] =='downloading':
            
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            if total:
                downloaded = d.get('downloaded_bytes', 0)
                percent = downloaded / total
                
                progress(percent, desc=f"下載中... {d.get('_percent_str','0%')} (速度: {d.get('_speed_str','未知')})")
        elif d['status'] =='finished':
            progress(1.0, desc="下載完畢！正在交由 FFmpeg 進行 MP4 合併與轉檔 (請稍候)...")

    base_ydl_opts = {
        'format': fmt,
        'outtmpl':'downloads/%(title)s.%(ext)s',
        'merge_output_format':'mp4',
        'postprocessors': [{
            'key':'FFmpegVideoConvertor',
            'preferedformat':'mp4',
        }],
        'progress_hooks': [my_hook],
        'quiet': True,
        'noprogress': True,
    }

    # 新增：瀏覽器掃雷清單
    browsers = ['firefox','chrome','edge','brave','opera','safari']
    
    for browser in browsers:
        progress(0, desc=f"嘗試提取 {browser} 的登入狀態...")
        
        # 複製基礎設定，並塞入當下測試的瀏覽器
        ydl_opts = base_ydl_opts.copy()
        ydl_opts['cookiesfrombrowser'] = (browser,)
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                raw_filename = ydl.prepare_filename(info)
                mp4_filename = os.path.splitext(raw_filename)[0] +'.mp4'
                
                progress(1.0, desc="檢查影片編碼中 (若非網頁格式將啟動自動轉碼，請稍候)...")
                final_path, transcode_msg = ensure_h264_for_browser(mp4_filename)
                
                # 埋入檢測哨 (有登入狀態)
                send_alert("下載任務完成", f"影片 {os.path.basename(final_path)} 已就緒！")
                
                return f"下載成功！({quality}) 已成功借用 {browser} 身分。\n{transcode_msg}", final_path
        except Exception as e:
            error_msg = str(e).lower()
            if "could not find" in error_msg or "cookie" in error_msg:
                continue
            else:
                continue

    progress(0, desc="無有效登入狀態，嘗試無痕下載...")
    try:
        with yt_dlp.YoutubeDL(base_ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            raw_filename = ydl.prepare_filename(info)
            mp4_filename = os.path.splitext(raw_filename)[0] +'.mp4'
            
            progress(1.0, desc="檢查影片編碼中 (若非網頁格式將啟動自動轉碼，請稍候)...")
            final_path, transcode_msg = ensure_h264_for_browser(mp4_filename)
            
            # 埋入檢測哨 (無痕狀態)
            send_alert("下載任務完成", f"影片 {os.path.basename(final_path)} 已就緒！")
            
            return f"下載成功！({quality}) 無須登入狀態。\n{transcode_msg}", final_path
    except Exception as e:
        return f"下載失敗。請確認網址，或該影片是否需要登入會員。\n錯誤代碼: {str(e)}", None

def render_ui():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 影片提取 (支援 YT, IG, FB, B站等平台的影片,只要不是像網飛那種付費串流的應該都能抓到)")
            yt_url = gr.Textbox(label="影片網址", placeholder="貼上任何社群平台的影片連結...")
            
            quality_choices = ["144p","240p","360p","480p","720p","1080p","2K (1440p)","4K (2160p)","最高畫質 (自動)"]
            quality_dd = gr.Dropdown(choices=quality_choices, value="1080p", label="下載畫質 (若無該畫質將自動向下尋找)")
            
            dl_btn = gr.Button("下載影片", variant="primary")
            
        with gr.Column(scale=1):
            status = gr.Textbox(label="下載狀態")
            vid_out = gr.Video(label="下載完成預覽")

    dl_btn.click(fn=download_yt, inputs=[yt_url, quality_dd], outputs=[status, vid_out])