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
6. **資料集不在工具目錄底下時**:Gradio 預設只允許顯示工作目錄以內的圖片,如果你的資料集放在別的磁碟位置(例如 `E:\訓練\`、`E:\ppe\`),抽樣比對、漏標預覽等需要顯示圖片的功能會被擋下,出現「Cannot move ... to the gradio cache dir」的錯誤。這時要在 `config.local.json` 加入 `allowed_paths`,列出要允許存取的資料夾(可以放多個,路徑用 `\\` 跳脫或直接用 `/`):
   ```json
   {
     "api_key": "",
     "workspace": "",
     "project": "",
     "allowed_paths": ["E:\\訓練", "E:\\ppe"]
   }
   ```
   路徑不存在或設成整個磁碟機根目錄(例如 `E:\`)會被自動略過並在終端機印警告,不會讓程式啟動失敗——但也因此**不能**用來一次開放整顆磁碟,請明確列出實際的資料夾。改完 `config.local.json` 後需要重新啟動程式才會生效。

## 環境變數

- `OPENAI_API_KEY`(選用):VLM 自動標註(OpenAI Vision)、AI 合成圖生成(DALL·E)這兩個功能才需要,沒設就只有這兩頁會提示未偵測到,其他功能不受影響。**絕對不要把 Key 寫進程式碼或 config 檔**,只透過系統環境變數設定。

## 怎麼啟動

雙擊 `app.bat`(之後每次啟動只會做 yt-dlp 版本檢查,不會重新建置環境)。

## 自動更新機制

程式開啟時會自動檢查目前 repo 是否落後遠端(`git fetch` + 比對 HEAD),有更新會在網頁最上方出現藍色橫幅,附上新 commit 的訊息跟版本號差異(例如「目前 v0.1.0 → 最新 v0.2.0」)。**不會自動更新**——按「立即更新」按鈕才會真的 `git pull`。如果拉取前偵測到你本機有未提交的修改,會直接拒絕更新並提示你先 commit 或 stash,不會覆蓋你的東西。更新成功後需要手動重新啟動程式才會生效。

沒裝 Git、不是 git 目錄、沒有網路、GitHub 認證失敗等狀況都只會顯示訊息,不會擋住程式正常啟動與使用。

## 版本號機制

- **單一來源**:`src/__version__.py` 裡的 `__version__` 字串,其他地方一律 `from __version__ import __version__` 引用,不要在別處寫死版本字串。
- **UI 顯示**:頁面上方會顯示 `v{版本號} ({commit 短 hash})`,例如 `v0.1.0 (a3f2c1d)`。**版本號可能忘了更新,但 commit hash 不會騙人**——回報問題時請兩個都截圖。不在 git 工作目錄(例如從 Release 下載的 zip)時 hash 會顯示 `unknown`,不會報錯。
- **錯誤回報輔助**:程式發生未預期例外時,錯誤訊息會自動附上 `[v0.1.0 (a3f2c1d)]`,組員截圖回報時可以一眼看出他跑的是哪一版,不用另外問。
- **CHANGELOG.md**:記錄每個版本改了什麼,分類 Added / Changed / Fixed / Removed。

### 發一個新版本要做哪些步驟

1. 改 `src/__version__.py` 的 `__version__` 字串。
2. 在 `CHANGELOG.md` 最上面新增一個 `## [新版本號] - 日期` 段落,依 Added / Changed / Fixed / Removed 分類寫這個版本改了什麼(版本切換頁面會直接抓這個段落給使用者看摘要,格式要對得上,`##` 後面接 `[版本號]`)。
3. `git add` + `git commit`。
4. `git tag v新版本號`(例如 `git tag v0.2.0`)——**tag 名稱要跟 `__version__.py`/CHANGELOG.md 的版本號一致**,不然「版本管理」頁面抓不到對應的 CHANGELOG 摘要。
5. `git push origin main --tags`。
6. 如果這個版本的 `models/*.pt` 有變動,記得也發一個新的 GitHub Release 附上新的權重檔。

## 版本管理(切換/回溯)

網頁上有一個「版本管理」分頁,列出所有 `git tag`,可以切換到舊版本檢查問題、或隨時「回到最新版」。

- 切換版本本質是 `git checkout <tag>`,會進入 **detached HEAD** 狀態(不在任何分支上)。這是正常的,「版本管理」頁面的狀態列會用顯眼的紅色底色提示你目前回溯到舊版本;**「🏠 回到最新版」按鈕**就是專門解決這個狀態的逃生出口(`git checkout main && git pull`),不熟 git 也不用擔心卡住。
- 切換前一定會檢查有沒有未提交的本機修改,有的話直接擋下來,**不會自動 stash 或丟棄任何東西**,你要自己先處理。
- 相依套件:切換版本後可以按「比對這個版本的 requirements.txt」,如果目前 venv 缺套件會列出來,按「同步相依套件」就會 `pip install -r requirements.txt`。
- 模型權重(`models/*.pt`)不在 repo 裡,切換版本不會幫你換權重檔,只會提醒你自行到對應的 GitHub Release 頁面確認。
- 沒有 `.git` 資料夾(直接下載 zip 的分發版本)時,這個分頁會自動隱藏切換功能、顯示提示文字,不會報錯。

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
