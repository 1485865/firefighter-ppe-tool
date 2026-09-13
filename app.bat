@echo off
chcp 65001 >nul
setlocal
title Firefighter AI Tool - Ultimate GPU Version
color 0B

echo ===================================================
echo Firefighter Project - AI Data Workspace
echo ===================================================
echo.

:: 1. 鎖定工作目錄，並把 bin 資料夾加入系統路徑
cd /d "%~dp0"
set PATH=%~dp0bin;%PATH%

:: ==========================================
:: 階段零：檢查 C++ 運行庫
:: ==========================================
if exist "%SystemRoot%\System32\vcruntime140.dll" goto CHECK_PYTHON

echo [System] 偵測不到 Visual C++ Redistributable！
echo [System] 正在從微軟官方下載 C++ 核心運行庫...
curl -L -o vc_redist.x64.exe https://aka.ms/vs/16/release/vc_redist.x64.exe
echo [System] 正在背景靜默安裝 C++ 運行庫 (請稍候)...
start /wait vc_redist.x64.exe /install /quiet /norestart
echo [System] C++ 運行庫安裝完成！
echo ---------------------------------------------------

:: ==========================================
:: 階段一：檢查 Python 核心
:: ==========================================
:CHECK_PYTHON
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 goto CHECK_VENV

echo [System] 偵測不到 Python！準備進行安裝...
echo [System] 正在下載 Python 3.10.11 安裝檔...
curl -o python_installer.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe

echo ===================================================
echo 【極度重要提醒】
echo 稍後會彈出 Python 的安裝畫面。
echo 請務必在第一步的最下方，勾選 [v] Add Python 3.10 to PATH
echo.
echo 安裝完成後，請手動關閉這個黑視窗，再重新點擊一次本啟動檔！
echo ===================================================
start /wait python_installer.exe
pause
exit /b

:: ==========================================
:: 階段二：檢查 AI 虛擬環境 (venv)
:: ==========================================
:CHECK_VENV
if exist ".\venv\Scripts\python.exe" goto RUN_APP

echo [System] 偵測到 Python，但缺少 AI 虛擬環境。
echo [System] 準備自動建置專屬環境 (venv)...
python -m venv venv

echo [System] 啟動虛擬環境並升級核心套件...
call venv\Scripts\activate
python -m pip install --upgrade pip

echo [System] 正在安裝 AI 必備套件與 GPU 引擎...
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [System] 環境建置大功告成！準備啟動系統...
echo ---------------------------------------------------

:: ==========================================
:: 階段三：啟動主程式
:: ==========================================
:RUN_APP
echo [System] 正在檢查 yt-dlp 是否有新版本(YouTube 常改規則,舊版會下載失敗 403)...
".\venv\Scripts\python.exe" -m pip install --upgrade yt-dlp --quiet --disable-pip-version-check
if %errorlevel% neq 0 (
    echo [System] 無法連線更新 yt-dlp，可能是離線狀態，將使用現有版本繼續啟動。
) else (
    echo [System] yt-dlp 已是最新版本。
)
echo ---------------------------------------------------

set CUDA_VISIBLE_DEVICES=0
set KMP_DUPLICATE_LIB_OK=TRUE

:: 保安現在知道要去 src 資料夾裡面找人了！
if not exist "src\app.py" (
    echo [ERROR] Main script app.py not found in src folder!
    echo Please make sure you moved app.py to the src folder.
    pause
    exit /b
)

echo [Starting] Waking up YOLO Model on RTX GPU...
echo [Notice] This may take 10~30 seconds. Please wait...
echo ---------------------------------------------------

:: 啟動 src 裡面的 app.py
".\venv\Scripts\python.exe" "src\app.py"

if %errorlevel% neq 0 (
    echo.
    echo ===================================================
    echo [FATAL ERROR] The application crashed!
    echo Please screenshot this window and send it to the dev.
    echo ===================================================
    pause
)