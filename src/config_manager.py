import json
import os
import shutil

CONFIG_LOCAL_FILE = "config.local.json"
CONFIG_EXAMPLE_FILE = "config.example.json"

_DEFAULT = {"api_key": "", "workspace": "", "project": "", "allowed_paths": []}


def save_config(api_key, workspace, project):
    data = {"api_key": api_key, "workspace": workspace, "project": project}
    with open(CONFIG_LOCAL_FILE, "w") as f:
        json.dump(data, f)


def load_config():
    if not os.path.exists(CONFIG_LOCAL_FILE):
        if os.path.exists(CONFIG_EXAMPLE_FILE):
            shutil.copy(CONFIG_EXAMPLE_FILE, CONFIG_LOCAL_FILE)
        else:
            return dict(_DEFAULT)
    with open(CONFIG_LOCAL_FILE, "r") as f:
        data = json.load(f)
    data.setdefault("allowed_paths", [])
    return data


def get_allowed_paths(cfg):
    """把 config 裡的 allowed_paths 轉成 Gradio demo.launch() 可用的絕對路徑清單。

    - 過濾掉不存在的路徑(印警告,不讓程式啟動失敗)
    - 過濾掉磁碟機根目錄(例如 "E:\\" 或 "/"),避免整顆磁碟被開放存取
    """
    result = []
    for raw_path in cfg.get("allowed_paths", []) or []:
        if not raw_path or not str(raw_path).strip():
            continue
        path = os.path.abspath(str(raw_path).strip())

        if not os.path.exists(path):
            print(f"[System] 警告:allowed_paths 設定的路徑不存在,已略過:{raw_path}")
            continue

        drive, tail = os.path.splitdrive(path)
        is_root = (tail in ("", os.sep, "/")) if drive else (path == os.sep)
        if is_root:
            print(f"[System] 警告:allowed_paths 不可設定整個磁碟機根目錄,已略過:{raw_path}")
            continue

        result.append(path)
    return result
