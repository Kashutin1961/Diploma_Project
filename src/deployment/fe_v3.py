# src/deployment/fe_v3.py
# RU: FE_v3 — нелинейные преобразования / EN: FE_v3 — nonlinear transforms

import numpy as np
import pandas as pd


def build_fe_v3(fe_v2: pd.DataFrame, nonlinear_feats: list) -> pd.DataFrame:
    """RU: FE_v3 — нелинейные преобразования / EN: FE_v3 — nonlinear transforms"""
    fe_v3 = fe_v2.copy()

    nonlinear_feats_test = [f for f in nonlinear_feats if f in fe_v3.columns]
    nonlinear_feats_test = nonlinear_feats_test[:6]
    print("Selected nonlinear features for test:", nonlinear_feats_test)

    if nonlinear_feats_test:
        for col in nonlinear_feats_test:
            fe_v3[f"log1p_{col}"] = np.log1p(fe_v3[col].clip(lower=0))
            fe_v3[f"sqrt_{col}"] = np.sqrt(fe_v3[col].clip(lower=0))
            try:
                fe_v3[f"bin_{col}"] = pd.qcut(
                    fe_v3[col],
                    q=10,
                    labels=False,
                    duplicates="drop",
                )
            except Exception:
                fe_v3[f"bin_{col}"] = fe_v3[col]
            lower = fe_v3[col].quantile(0.01)
            upper = fe_v3[col].quantile(0.99)
            fe_v3[f"winsor_{col}"] = fe_v3[col].clip(lower, upper)
        print("Nonlinear transforms applied.")
    else:
        print("No nonlinear features from Stage 5; FE_v3_test = FE_v2_test.")

    print("FE_v3_test:", fe_v3.shape)
    return fe_v3
