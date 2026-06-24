# src/deployment/model_loader.py
# RU: Загрузка моделей и финальных признаков / EN: Load models and final features

import json
from pathlib import Path

import joblib


def load_models_and_features(project_root: Path):
    """RU: Загрузка моделей и финальных признаков / EN: Load models and final features"""
    ensemble_dir = project_root / "data" / "stage_outputs" / "stage8_optimization"
    print("Ensemble dir:", ensemble_dir)

    base_model_names = ["lasso", "lr", "ridge", "rf", "xgb"]
    models = {}

    for name in base_model_names:
        model_path = ensemble_dir / f"model_{name}_f205.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        models[name] = joblib.load(model_path)
        print(f"Loaded base model: {name} → {model_path.name}")

    meta_model_path = ensemble_dir / "meta_model_v3.pkl"
    if not meta_model_path.exists():
        raise FileNotFoundError(f"Meta-model not found: {meta_model_path}")
    meta_model = joblib.load(meta_model_path)
    print("Loaded meta-model:", meta_model_path.name)

    features_path = ensemble_dir / "final_selected_features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"final_selected_features.json not found: {features_path}")
    with open(features_path, "r", encoding="utf-8") as f:
        tmp = json.load(f)
    final_features = tmp.get("selected_features", None)
    if final_features is None:
        raise ValueError("JSON has no 'selected_features'.")
    print(f"Loaded final features list: {len(final_features)} features")

    return models, meta_model, final_features, base_model_names
