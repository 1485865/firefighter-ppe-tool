# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the PySide6 desktop build(app_qt.py)。

用 --onedir(不是 --onefile):torch(cu121)+ultralytics+diffusers+
transformers+accelerate 加起來是數 GB 等級,onefile 每次啟動都要解壓縮到
暫存,對這種體積的 ML app 沒有好處,業界慣例一律用 onedir。
"""
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None
PROJECT_ROOT = os.path.abspath(".")

datas = []
binaries = []
hiddenimports = [
    "cv2", "PIL", "plyer.platforms.win.notification",
    "roboflow", "yt_dlp", "openai",
]

for pkg in ("torch", "torchvision", "ultralytics", "diffusers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

datas += collect_data_files("transformers")

# utils.py(annotate/extractor/distill/tracking/youtube 全部共用)頂層就
# `import gradio as gr`,即使 Qt 版完全不用 Gradio UI,這個 import 還是會
# 執行到,所以 gradio 本身、以及它會在 import 期間就讀資料檔的依賴
# (safehttpx 讀 version.txt)都要一起 collect_all,不然只抓到 .py 卻漏了
# package data,會在 import 階段就 FileNotFoundError 直接炸掉。
for pkg in ("gradio", "gradio_client", "safehttpx", "groovy"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

try:
    d, b, h = collect_all("google.genai")
    datas += d
    binaries += b
    hiddenimports += h
except Exception:
    pass

# 專案自己的相對路徑資源:bin/ffmpeg 系列、models/*.pt、config.json
datas += [
    (os.path.join(PROJECT_ROOT, "bin", "ffmpeg.exe"), "bin"),
    (os.path.join(PROJECT_ROOT, "bin", "ffplay.exe"), "bin"),
    (os.path.join(PROJECT_ROOT, "bin", "ffprobe.exe"), "bin"),
    (os.path.join(PROJECT_ROOT, "models"), "models"),
    (os.path.join(PROJECT_ROOT, "config.json"), "."),
]

a = Analysis(
    ["src/app_qt.py"],
    pathex=[os.path.join(PROJECT_ROOT, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # 注意:不能排除 gradio —— utils.py(annotate/extractor/distill/tracking/
    # youtube 全部共用)頂層就 `import gradio as gr`,拿掉會讓 exe 直接炸。
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FirefighterAI",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="FirefighterAI",
)
