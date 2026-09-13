"""git subprocess 共用工具:updater.py(自動更新)與 version_switch.py
(版本切換與回溯)都靠這個。統一處理 timeout,永遠不拋例外,呼叫端自己看
subprocess 的 returncode/stderr——沒裝 git、不是 git 目錄、逾時都不會讓
呼叫端崩潰,只會拿到一個 returncode != 0 的結果或 False。
"""
import subprocess


def run_git(args, cwd=".", timeout=10):
    return subprocess.run(
        ["git"] + args, cwd=cwd, capture_output=True, text=True,
        timeout=timeout, encoding="utf-8", errors="replace",
    )


def is_git_repo(path="."):
    try:
        r = run_git(["rev-parse", "--is-inside-work-tree"], cwd=path, timeout=5)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        return False


def has_uncommitted_changes(path="."):
    try:
        r = run_git(["status", "--porcelain"], cwd=path, timeout=5)
        return bool(r.stdout.strip())
    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        return False
