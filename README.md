# firefighter-tool

消防 PPE(個人防護裝備)資料集建置與標註工具。用 Gradio 打造的一站式工作站:YouTube 影片下載、抽幀清洗、YOLO 批次預標註、VLM(OpenAI Vision)自動標註、模型蒸餾、主動學習排序、跨模型共識標註、標註品質稽核、Data Scaling Curve 等。

## 前置需求

- Windows + Python 3.10(`app.bat` 會自動檢查,沒裝會引導安裝)
- Git(用於 clone 本 repo,以及自動更新機制)
- (建議)NVIDIA GPU + 對應驅動,CPU 也能跑,但速度會慢很多

## 第一次 clone 後怎麼設定

1. `git clone` 這個 repo。
2. 雙擊 `app.bat`。它會自動:
   - 檢查/安裝 Python(沒有的話會引導你安裝)
   - 建立虛擬環境 `venv/`
   - `pip install -r requirements.txt`
   - 另外安裝 GPU 版 `torch`/`torchvision`/`torchaudio`(cu121)
   - 啟動 Gradio 網頁介面(自動開瀏覽器)
3. **設定檔**:第一次啟動時,程式會自動把 `config.example.json` 複製成 `config.local.json`(這個檔案不進版控,存你自己的 Roboflow API Key/Workspace/Project,在網頁上方「Roboflow API 設定」填好按儲存就會寫進這個檔案)。
4. **模型權重**:`models/*.pt` 跟根目錄的 `yolo11n.pt`/`yolo11x.pt`/`yolo26n.pt` **沒有進版控**(`.gitignore` 排除了所有 `*.pt`,避免大型二進位檔案塞爆 repo)。請跟團隊拿這些檔案,放進 `models/` 資料夾(根目錄的幾個 `.pt` 底模是 ultralytics 官方預訓練檔,沒有也會在第一次用到時自動下載,只是需要網路)。
5. **ffmpeg**:`bin/*.exe` 也沒有進版控。程式開啟時會自動檢查(先找系統 PATH,再找 `bin/`),兩個都沒有就自動從網路下載到 `bin/`(過程有進度顯示)。下載失敗會在頁面上顯示明確訊息,告訴你去哪裡手動下載安裝,不會擋住程式其他功能正常使用。

## 環境變數

- `OPENAI_API_KEY`(選用):VLM 自動標註(OpenAI Vision)、AI 合成圖生成(DALL·E)這兩個功能才需要,沒設就只有這兩頁會提示未偵測到,其他功能不受影響。**絕對不要把 Key 寫進程式碼或 config 檔**,只透過系統環境變數設定。

## 怎麼啟動

雙擊 `app.bat`(之後每次啟動只會做 yt-dlp 版本檢查,不會重新建置環境)。

## 自動更新機制

程式開啟時會自動檢查目前 repo 是否落後遠端(`git fetch` + 比對 HEAD),有更新會在網頁最上方出現藍色橫幅,附上新 commit 的訊息。**不會自動更新**——按「立即更新」按鈕才會真的 `git pull`。如果拉取前偵測到你本機有未提交的修改,會直接拒絕更新並提示你先 commit 或 stash,不會覆蓋你的東西。更新成功後需要手動重新啟動程式才會生效。

沒裝 Git、不是 git 目錄、沒有網路、GitHub 認證失敗等狀況都只會顯示訊息,不會擋住程式正常啟動與使用。

## ⚠️ 2026-09-13:repo 歷史被改寫過(移除 ffmpeg 執行檔)

`bin/` 底下的 ffmpeg 執行檔(約 290MB)原本不小心 commit 進 repo 歷史,已經用
`git filter-repo` 從**所有 commit 歷史**徹底移除,改成程式啟動時自動下載
(見上面「ffmpeg」那條)。這個操作會改寫 commit hash,**如果你已經 clone 過舊版**,
你的本機分支跟遠端會完全對不上,請照以下任一方式處理:

**方式一(推薦,最乾淨)**:直接刪掉本機資料夾,重新 `git clone`。

**方式二(想保留本機沒推上去的修改)**:
```bash
# 先確認有沒有還沒推上去的本機修改,有的話先自行備份/處理
git status
git log origin/main..HEAD   # 看有哪些 commit 只在本機、遠端沒有

# 確認沒問題後,同步到改寫後的歷史(這步會丟棄本機舊的、跟遠端衝突的歷史)
git fetch origin
git checkout main
git reset --hard origin/main
git gc --prune=now
```
**方式二會丟棄任何還沒推上遠端的本機 commit**,操作前務必先用 `git log origin/main..HEAD` 確認沒有東西會不見。

## 桌面版(選用,非主要交付物)

`src/qt/` 底下有一份用 PySide6 寫的原生桌面版原型,重用同一批核心邏輯模組,不開瀏覽器。開發模式用 `run_desktop.bat` 啟動;要打包成 exe 用 `build_exe.bat`(會跑 PyInstaller,輸出到 `dist/`,體積較大)。目前主要維護與交付的還是 `app.bat` 這條 Gradio 路線。
