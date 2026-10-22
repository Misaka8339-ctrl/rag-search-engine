import string
from functools import lru_cache
from pathlib import Path

from nltk.stem import PorterStemmer

from .config import STOPWORDS_PATH


stemmer = PorterStemmer()

@lru_cache(maxsize=1)

def tokenize(text: str) -> list[str]:
    translation_table = str.maketrans("", "", string.punctuation)
    text = text.lower().translate(translation_table)
    return text.split()

def load_stop_words(file_path: Path) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    stop_words = []

    for line in lines:
        stop_words.extend(tokenize(line))

    return stop_words

def get_stop_words() -> list[str]:
    return load_stop_words(STOPWORDS_PATH)

def remove_stop_words(
    tokens: list[str], stop_words: list[str]
) -> list[str]:
    return [token for token in tokens if token not in stop_words]

def preprocess_text(text: str, stop_words: list[str]) -> list[str]:
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens, stop_words)
    return [stemmer.stem(token) for token in tokens]

def tokenize_text(text: str) -> list[str]:
    return preprocess_text(text, get_stop_words())

def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)

    if len(tokens) != 1:
        raise ValueError(
            "Term must produce exactly one token after preprocessing."
        )

    return tokens[0]


tokenize_text("The Running Bears!")
# 返回预处理后的 token 列表

tokenize_term("running")
# 返回单个 token；数量不为 1 时抛出异常