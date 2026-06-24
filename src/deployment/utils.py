# src/deployment/utils.py
# RU: Вспомогательные функции / EN: Utility helpers

from pathlib import Path
import pickle


def load_pkl(path: Path):
    """RU: Загрузка pickle-файла / EN: Load pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)
