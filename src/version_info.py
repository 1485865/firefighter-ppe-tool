"""版本號 + git commit hash 查詢,供 UI 顯示與錯誤回報使用。

版本號單一來源是 __version__.py,這裡不重複定義,只 import。
commit hash 抓不到(不在 git 工作目錄、沒裝 git、zip 分發)一律回傳
"unknown",不拋例外——版本號可能忘了更新,但 commit hash 不會騙人,
兩個一起看才是可靠的除錯依據。
"""
from __version__ import __version__

from git_utils import run_git


def get_commit_hash(path="."):
    """回傳目前 commit 短 hash(前 7 碼);抓不到回傳 "unknown"。"""
    try:
        r = run_git(["rev-parse", "--short=7", "HEAD"], cwd=path, timeout=5)
        if r.returncode == 0:
            h = r.stdout.strip()
            if h:
                return h
    except Exception:
        pass
    return "unknown"


def get_version_string(path="."):
    """回傳 "v0.1.0 (a3f2c1d)" 格式字串。"""
    return f"v{__version__} ({get_commit_hash(path)})"
