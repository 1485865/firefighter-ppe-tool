"""ffmpeg/ffprobe 自動取得:啟動時檢查系統 PATH → 專案 bin/,都沒有就自動
下載並解壓到 bin/。執行檔本身不進版控(見 .gitignore 的 bin/*.exe)。

下載來源用 gyan.dev 的靜態建置——FFmpeg 官方(ffmpeg.org)本身不提供
Windows 執行檔,官網下載頁指定推薦 gyan.dev / BtbN 這兩個社群建置當作
事實上的標準來源,這裡選 gyan.dev(這台機器原本 winget 裝的 ffmpeg
就是同一個來源)。

沒有處理 ffplay——專案程式碼從頭到尾沒有呼叫過它(只用 ffmpeg 轉檔、
ffprobe 探測編碼),下載完整建置只是浪費頻寬跟磁碟空間。
"""
import os
import shutil
import urllib.error
import urllib.request
import zipfile

FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin")


def _bin_path(exe_name):
    p = os.path.join(BIN_DIR, exe_name)
    return p if os.path.exists(p) else None


def find_ffmpeg():
    """優先序:系統 PATH → 專案 bin/。回傳 (ffmpeg_path, ffprobe_path);
    任一個找不到就是 (None, None)。"""
    ffmpeg = shutil.which("ffmpeg") or _bin_path("ffmpeg.exe")
    ffprobe = shutil.which("ffprobe") or _bin_path("ffprobe.exe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    return None, None


def download_ffmpeg(progress_cb=None):
    """從官方推薦的靜態建置下載 ffmpeg/ffprobe 並解壓到 bin/。

    progress_cb(fraction, desc) 回報進度(相容 gr.Progress() 呼叫慣例)。
    成功回傳 (ffmpeg_path, ffprobe_path);失敗一律丟 RuntimeError,
    訊息附上手動安裝的說明,絕不靜默失敗。
    """
    os.makedirs(BIN_DIR, exist_ok=True)
    zip_path = os.path.join(BIN_DIR, "_ffmpeg_download.zip")

    def _p(frac, desc):
        if progress_cb is not None:
            progress_cb(frac, desc=desc)

    try:
        _p(0.0, "連線 gyan.dev 下載 ffmpeg 靜態建置...")
        req = urllib.request.Request(FFMPEG_URL, headers={"User-Agent": "firefighter-tool"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        frac = min(0.85, downloaded / total * 0.85)
                        _p(frac, f"下載中... {downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB")

        _p(0.9, "解壓縮中...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # essentials 版壓縮包內層資料夾名稱含版本號,用結尾比對而不是寫死路徑
            members = [m for m in zf.namelist()
                       if m.endswith("/bin/ffmpeg.exe") or m.endswith("/bin/ffprobe.exe")]
            if not members:
                raise RuntimeError("下載檔案內容不符預期,找不到 ffmpeg.exe/ffprobe.exe")
            for member in members:
                target_name = os.path.basename(member)
                with zf.open(member) as src, open(os.path.join(BIN_DIR, target_name), "wb") as dst:
                    shutil.copyfileobj(src, dst)

        ffmpeg_path = os.path.join(BIN_DIR, "ffmpeg.exe")
        ffprobe_path = os.path.join(BIN_DIR, "ffprobe.exe")
        if not (os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path)):
            raise RuntimeError("解壓後找不到 ffmpeg.exe/ffprobe.exe")

        _p(1.0, "完成")
        return ffmpeg_path, ffprobe_path

    except (urllib.error.URLError, TimeoutError, OSError, zipfile.BadZipFile) as e:
        raise RuntimeError(
            f"自動下載 ffmpeg 失敗({type(e).__name__}: {e})。\n"
            "請手動從 https://www.gyan.dev/ffmpeg/builds/ 下載"
            "「ffmpeg-release-essentials.zip」,解壓後把 ffmpeg.exe 和 ffprobe.exe "
            "放進專案的 bin/ 資料夾,或直接安裝 ffmpeg 並確保它在系統 PATH 裡。"
        ) from e
    finally:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass


def ensure_ffmpeg(progress_cb=None):
    """啟動時呼叫一次。找不到才下載;下載失敗會把 RuntimeError 往外拋,
    由呼叫端(app.py 的 demo.load 處理函式)接住並顯示訊息,不會讓程式當掉。
    """
    ffmpeg, ffprobe = find_ffmpeg()
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    return download_ffmpeg(progress_cb)
