@echo off
chcp 65001 >nul
setlocal
title Firefighter AI Tool - Desktop (Qt)

cd /d "%~dp0"
set PATH=%~dp0bin;%PATH%

if not exist ".\venv\Scripts\python.exe" (
    echo [ERROR] 找不到 venv,請先執行 app.bat 建置環境。
    pause
    exit /b
)

set CUDA_VISIBLE_DEVICES=0
set KMP_DUPLICATE_LIB_OK=TRUE

".\venv\Scripts\python.exe" "src\app_qt.py"

if %errorlevel% neq 0 (
    echo.
    echo [FATAL ERROR] 程式發生錯誤,請截圖回報。
    pause
)
