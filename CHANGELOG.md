# Changelog

格式參考 [Keep a Changelog](https://keepachangelog.com/),分類:Added / Changed / Fixed / Removed。

## [0.1.0] - 2026-09-14

第一個掛版本號的版本,把目前已知的既有異動一併補進來。

### Added
- VLM 自動標註模組(OpenAI Vision API),沿用既有 `_HARD` 人工複核慣例輸出 YOLO txt + XML。
- 抽幀頁(影片抽幀與清洗)的 filmstrip 縮圖列支援橫向拖曳/滾輪捲動。
- 資料集品質工具四件套:主動學習排序、跨模型共識標註、標註品質稽核、Data Scaling Curve。
- PySide6 原生桌面版原型(`src/qt/`),不開瀏覽器,可用 `run_desktop.bat` 啟動、`build_exe.bat` 打包。
- 私有 GitHub repo 化 + 啟動時自動更新檢查(`src/updater.py`),有更新只顯示提示,使用者按按鈕才真的拉取。
- 版本號機制:`src/__version__.py` 單一來源、UI 顯示版本號+commit hash、更新提示顯示版本差異、未預期例外自動附上版本資訊。
- 版本切換與回溯 UI:列出所有 tag、切換到指定版本、一鍵回到最新版、相依套件差異偵測與同步。

### Changed
- ffmpeg/ffprobe 改成啟動時自動偵測(系統 PATH → 專案 `bin/`),偵測不到才從網路下載,不再假設執行檔一定存在。
- Roboflow 憑證改用 `config.local.json`(本機專用、不進版控)+ `config.example.json`(進 repo 的空白範本)分離。
- `app.bat` 改用 `pip install -r requirements.txt`,不再把套件清單寫死在 bat 檔裡。

### Fixed
- 批次預標註(`tab_batch_annotate.py`)的暫存目錄改用專案目錄下的固定路徑,修正系統 Temp 資料夾被清理工具清掉後導致的 `WinError 3`(找不到指定的路徑)。
- `tab_distill.py` 的 `run_feature_distill()` 驗證失敗訊息原本是 generator 裡直接 `return 字串`,不會真的送到 UI(靜默吞掉),改成正確的 `yield`。

### Removed
- `bin/` 底下的 ffmpeg 執行檔(約 290MB)已用 `git filter-repo` 從所有 commit 歷史徹底移除,不再進版控。
