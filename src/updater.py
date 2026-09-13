"""自動更新機制:啟動時檢查私有 GitHub repo 有沒有新版本,有的話在 UI
顯示提示,由使用者按按鈕確認才真的拉取,絕不自動更新。

存取權完全交給呼叫端已經設定好的 git 憑證(SSH key / credential manager)
處理,這個模組全程不儲存、不處理任何 GitHub token。

每個對外函式都保證回傳一個 dict(status, message, ...),不會把例外往外拋——
沒網路、沒裝 git、認證失敗都各自有清楚的中文訊息,絕不讓更新檢查擋住主要功能。
"""
import subprocess

from git_utils import run_git, is_git_repo, has_uncommitted_changes
from __version__ import __version__ as LOCAL_VERSION


def _read_version_at_ref(ref, path="."):
    """讀取指定 ref(例如 @{u})的 src/__version__.py 版本號;抓不到回傳 None。"""
    try:
        r = run_git(["show", f"{ref}:src/__version__.py"], cwd=path, timeout=5)
        if r.returncode != 0:
            return None
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("__version__"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    return parts[1].strip().strip("'\"")
    except Exception:
        return None
    return None


def check_for_update(path="."):
    """回傳 dict(status, message, commits=None, local_version, remote_version)。
    status: "not_git_repo" | "up_to_date" | "update_available" | "error"
    """
    if not is_git_repo(path):
        return {"status": "not_git_repo", "message": "不是 git 工作目錄,已跳過更新檢查。"}

    try:
        fetch = run_git(["fetch"], cwd=path, timeout=20)
        if fetch.returncode != 0:
            return {"status": "error",
                    "message": f"git fetch 失敗(可能沒有網路或認證失敗): {fetch.stderr.strip()[:300]}"}

        local = run_git(["rev-parse", "HEAD"], cwd=path, timeout=5)
        remote = run_git(["rev-parse", "@{u}"], cwd=path, timeout=5)
        if remote.returncode != 0:
            return {"status": "error",
                    "message": "目前分支沒有設定追蹤的遠端分支(upstream),已跳過更新檢查。"}

        local_hash = local.stdout.strip()
        remote_hash = remote.stdout.strip()
        if local_hash == remote_hash:
            return {"status": "up_to_date", "message": f"已是最新版本(v{LOCAL_VERSION})。"}

        log = run_git(
            ["log", f"{local_hash}..{remote_hash}", "--oneline", "-20"],
            cwd=path, timeout=5,
        )
        commits = log.stdout.strip()
        n = len(commits.splitlines()) if commits else 0

        remote_version = _read_version_at_ref("@{u}", path) or "unknown"
        version_line = f"目前 v{LOCAL_VERSION} → 最新 v{remote_version}"

        return {
            "status": "update_available",
            "message": f"發現新版本,有 {n} 個新 commit。{version_line}",
            "commits": commits,
            "local_version": LOCAL_VERSION,
            "remote_version": remote_version,
        }
    except FileNotFoundError:
        return {"status": "error", "message": "系統找不到 git,請先安裝 Git 才能使用自動更新功能。"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "連線逾時(可能沒有網路或 GitHub 認證失敗),已跳過本次更新檢查。"}
    except Exception as e:
        return {"status": "error", "message": f"更新檢查發生未預期錯誤: {e}"}


def do_update(path="."):
    """回傳 dict(status, message)。status: "dirty" | "success" | "error"。

    偵測到未提交的本機修改一律拒絕更新,不強制覆蓋使用者的東西。
    """
    try:
        if has_uncommitted_changes(path):
            return {"status": "dirty",
                    "message": "偵測到尚未提交的本機修改,為避免覆蓋你的變更,已取消更新。"
                               "請先自行 commit 或 stash 後再試一次。"}

        pull = run_git(["pull"], cwd=path, timeout=30)
        if pull.returncode != 0:
            return {"status": "error", "message": f"更新失敗: {pull.stderr.strip()[:300]}"}

        return {"status": "success", "message": "更新完成!請關閉並重新啟動本工具讓新版本生效。"}
    except FileNotFoundError:
        return {"status": "error", "message": "系統找不到 git,請先安裝 Git。"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "更新逾時,請檢查網路連線後再試一次。"}
    except Exception as e:
        return {"status": "error", "message": f"更新時發生未預期錯誤: {e}"}
