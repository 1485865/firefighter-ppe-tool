"""版本切換與回溯:列出 git tag、切換到指定版本、回到最新版、比對
requirements.txt 相依套件差異。

沒有 .git 資料夾時(zip 分發),呼叫端(tab_version.py)要自己用
git_utils.is_git_repo() 判斷並優雅隱藏這整個功能——這裡的函式一律不拋例外、
不自動做任何會丟資料的操作(不自動 stash、不自動 git reset --hard),
有未提交修改一律直接拒絕切換,讓使用者自己決定怎麼處理。
"""
import os
import re
import subprocess

from git_utils import run_git, has_uncommitted_changes


def list_tags(path="."):
    """git fetch --tags 後列出所有 tag,新到舊排序。
    回傳 [{"name","date","subject"}, ...];任何失敗都回傳空清單。"""
    try:
        run_git(["fetch", "--tags"], cwd=path, timeout=20)
        r = run_git(
            ["for-each-ref", "--sort=-creatordate", "refs/tags",
             "--format=%(refname:short)|%(creatordate:short)|%(subject)"],
            cwd=path, timeout=10,
        )
        if r.returncode != 0:
            return []
        tags = []
        for line in r.stdout.strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                tags.append({"name": parts[0], "date": parts[1], "subject": parts[2]})
        return tags
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def get_current_branch(path="."):
    """在一般分支上回傳分支名;detached HEAD 回傳 None。"""
    try:
        r = run_git(["symbolic-ref", "--short", "-q", "HEAD"], cwd=path, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_current_tag(path="."):
    """detached HEAD 狀態下,嘗試找出目前 HEAD 剛好對應哪個 tag;找不到回傳 None。"""
    try:
        r = run_git(["describe", "--tags", "--exact-match"], cwd=path, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def changelog_summary(tag, path="."):
    """從 CHANGELOG.md 找對應版本段落,回傳前幾行摘要;找不到回傳空字串。"""
    changelog_path = os.path.join(path, "CHANGELOG.md")
    if not os.path.exists(changelog_path):
        return ""
    version = tag.lstrip("v")
    try:
        with open(changelog_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return ""
    pattern = rf"^##\s*\[?{re.escape(version)}\]?.*?$(.*?)(?=^##\s|\Z)"
    m = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    if not m:
        return ""
    lines = [l for l in m.group(1).strip().splitlines() if l.strip()]
    return "\n".join(lines[:8])


def switch_to_tag(tag, path="."):
    """回傳 dict(status, message)。status: "dirty" | "success" | "error"。"""
    if has_uncommitted_changes(path):
        return {"status": "dirty",
                "message": f"偵測到尚未提交的本機修改,已取消切換到 {tag}。"
                           "請先自行 commit 或 stash 你的修改後再試一次(不會自動處理)。"}
    try:
        r = run_git(["checkout", tag], cwd=path, timeout=20)
        if r.returncode != 0:
            return {"status": "error", "message": f"切換失敗:\n{r.stderr.strip()}"}
        return {"status": "success",
                "message": f"已切換到 {tag}(現在是 detached HEAD 狀態)。"
                           "請重新啟動程式讓這個版本生效;要回到最新版請按「回到最新版」按鈕。"}
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"status": "error", "message": f"切換失敗: {e}"}


def switch_to_latest(path="."):
    """逃生出口:git checkout main && git pull,退出 detached HEAD 狀態回到最新版。"""
    if has_uncommitted_changes(path):
        return {"status": "dirty",
                "message": "偵測到尚未提交的本機修改,已取消切換。請先自行 commit 或 stash。"}
    try:
        r1 = run_git(["checkout", "main"], cwd=path, timeout=20)
        if r1.returncode != 0:
            return {"status": "error", "message": f"切回 main 失敗:\n{r1.stderr.strip()}"}
        r2 = run_git(["pull"], cwd=path, timeout=30)
        if r2.returncode != 0:
            return {"status": "error", "message": f"git pull 失敗:\n{r2.stderr.strip()}"}
        return {"status": "success", "message": "已回到最新版(main)。請重新啟動程式讓它生效。"}
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"status": "error", "message": f"切換失敗: {e}"}


def diff_requirements(ref, path="."):
    """比較指定 ref 的 requirements.txt 跟目前 venv 已安裝的套件,列出缺少的。

    這個專案的 requirements.txt 沒有釘版本號,所以只比對「套件有沒有裝」,
    不比對版本號是否一致(釘版本號的話這裡可以再加精細比對)。
    回傳 dict(missing=[...]) 或 dict(error=...) 表示比對失敗。
    """
    r = run_git(["show", f"{ref}:requirements.txt"], cwd=path, timeout=10)
    if r.returncode != 0:
        return {"error": f"這個版本讀不到 requirements.txt: {r.stderr.strip()[:200]}"}

    required = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[]", line, 1)[0].strip().lower()
        if name:
            required.append(name)

    venv_python = os.path.join(path, "venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        return {"error": "找不到 venv,請先執行 app.bat 建置環境。"}

    try:
        pip_freeze = subprocess.run(
            [venv_python, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"error": f"無法讀取目前已安裝套件: {e}"}

    installed = set()
    for line in pip_freeze.stdout.splitlines():
        name = re.split(r"[=@]", line, 1)[0].strip().lower()
        if name:
            installed.add(name)

    missing = [name for name in required if name not in installed]
    return {"missing": missing}


def sync_requirements(path="."):
    """pip install -r requirements.txt(用目前 checkout 出來、磁碟上的版本)。"""
    venv_python = os.path.join(path, "venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        return {"status": "error", "message": "找不到 venv,請先執行 app.bat 建置環境。"}
    try:
        r = subprocess.run(
            [venv_python, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=path, capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            return {"status": "error", "message": f"套件同步失敗:\n{r.stderr.strip()[:500]}"}
        return {"status": "success", "message": "套件已同步完成。"}
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"status": "error", "message": f"套件同步失敗: {e}"}
