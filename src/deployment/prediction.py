# src/deployment/prediction.py
# RU: Выравнивание признаков и предсказание / EN: Feature alignment and prediction

from pathlib import Path

import numpy as np
import pandas as pd


def align_features(fe_final: pd.DataFrame, final_features: list) -> pd.DataFrame:
    """RU: Выравнивание признаков / EN: Align features"""
    missing_in_test = set(final_features) - set(fe_final.columns)
    if missing_in_test:
        print("WARNING: missing features in test:", missing_in_test)
        for col in missing_in_test:
            fe_final[col] = 0.0
    df_final = fe_final[final_features].copy()
    print("Aligned df_test_final:", df_final.shape)
    return df_final


def predict_ensemble(models: dict, meta_model, base_model_names: list, X) -> np.ndarray:
    """RU: Предсказание ансамблем / EN: Ensemble prediction"""
    base_preds = []
    for name in base_model_names:
        model = models[name]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            # RidgeClassifier fallback: convert decision_function to probability
            decision = model.decision_function(X)
            proba = 1 / (1 + np.exp(-decision))

        base_preds.append(proba)
        print(f"Base model {name} done.")

    stacked = np.vstack(base_preds).T
    meta_proba = meta_model.predict_proba(stacked)[:, 1]
    return meta_proba


def save_submission(project_root: Path, df_ids: pd.Series, proba, filename: str = "submission_stage10.csv"):
    """RU: Сохранение сабмита / EN: Save submission file"""
    submission = pd.DataFrame({
        "SK_ID_CURR": df_ids.values,
        "TARGET": proba,
    })
    print("Submission shape:", submission.shape)

    submission_dir = project_root / "data" / "stage_outputs" / "stage10_deployment"
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / filename
    submission.to_csv(submission_path, index=False)
    print("Submission saved to:", submission_path)
