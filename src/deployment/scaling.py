# src/deployment/scaling.py
# RU: Масштабирование признаков / EN: Feature scaling

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(fe_mv: pd.DataFrame, apply_scaling: bool = False) -> pd.DataFrame:
    """RU: Масштабирование (опционально) / EN: Scaling (optional)"""
    fe_final = fe_mv.copy()
    num_cols = fe_final.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print(f"Numeric cols for scaling (test): {len(num_cols)}")

    if apply_scaling:
        scaler = StandardScaler()
        fe_final[num_cols] = scaler.fit_transform(fe_final[num_cols])
        print("Scaling applied (StandardScaler).")
    else:
        print("Scaling skipped (tree-based models).")

    print("FE_v3_test_final:", fe_final.shape)
    return fe_final
