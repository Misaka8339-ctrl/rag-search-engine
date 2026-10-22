import json

from .config import DATA_PATH

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["movies"]