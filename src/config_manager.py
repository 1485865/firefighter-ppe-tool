import json
import os
import shutil

CONFIG_LOCAL_FILE = "config.local.json"
CONFIG_EXAMPLE_FILE = "config.example.json"

_DEFAULT = {"api_key": "", "workspace": "", "project": ""}


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
        return json.load(f)
