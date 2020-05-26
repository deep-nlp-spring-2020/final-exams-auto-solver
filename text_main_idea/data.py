import json


def read_config(config_path):
    if isinstance(config_path, str):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    return config
