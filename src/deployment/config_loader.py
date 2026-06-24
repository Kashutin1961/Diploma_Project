# src/deployment/config_loader.py
# RU: Загрузка конфигурации и Stage 5 summary / EN: Config & Stage 5 summary loading

import json
from pathlib import Path


def load_config(project_root: Path) -> dict:
    """RU: Загрузка конфигурации проекта / EN: Load project config"""
    config_path = project_root / "project_structure_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stage5_summary(project_root: Path, config: dict):
    """RU: Загрузка Stage 5 summary / EN: Load Stage 5 summary"""
    stage5_output_dir = project_root / config["stages"]["stage5"]["output_path"]
    summary_candidates = list(stage5_output_dir.rglob("stage5_summary.json"))
    if not summary_candidates:
        raise FileNotFoundError(f"stage5_summary.json not found in {stage5_output_dir}")
    stage5_summary_path = summary_candidates[0]
    with open(stage5_summary_path, "r", encoding="utf-8") as f:
        stage5_summary = json.load(f)
    top_interactions = stage5_summary.get("top_interactions", [])
    nonlinear_feats = stage5_summary.get("nonlinear_features", [])
    print("Loaded top_interactions:", len(top_interactions))
    print("Loaded nonlinear_features:", len(nonlinear_feats))
    return top_interactions, nonlinear_feats
