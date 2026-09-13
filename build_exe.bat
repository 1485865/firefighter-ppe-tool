@echo off
chcp 65001 >nul
setlocal
title Firefighter AI Tool - Build EXE

cd /d "%~dp0"

if not exist ".\venv\Scripts\python.exe" (
    echo [ERROR] 找不到 venv,請先執行 app.bat 建置環境。
    pause
    exit /b
)

echo [System] 開始打包(--onedir,體積龐大,請耐心等候,可能要 10~30 分鐘)...
".\venv\Scripts\python.exe" -m PyInstaller --noconfirm app_qt.spec

if %errorlevel% neq 0 (
    echo.
    echo [FATAL ERROR] 打包失敗,請往上捲看錯誤訊息。
    pause
    exit /b
)

echo.
echo [System] 打包完成!輸出在 dist\FirefighterAI\FirefighterAI.exe
pause
