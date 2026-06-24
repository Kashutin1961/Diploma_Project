# src/deployment/fe_v2.py
# RU: FE_v2 — пропорции и SHAP-интеракции / EN: FE_v2 — ratios and SHAP interactions

import numpy as np
import pandas as pd


def build_fe_v2(fe_v1: pd.DataFrame, top_interactions: list) -> pd.DataFrame:
    """RU: FE_v2 — ratio + SHAP-интеракции / EN: FE_v2 — ratios + SHAP interactions"""
    fe_v2 = fe_v1.copy()

    fe_v2["ratio_annuity_income"] = fe_v2["AMT_ANNUITY"] / fe_v2["AMT_INCOME_TOTAL"].replace(0, np.nan)
    fe_v2["ratio_credit_income"] = fe_v2["AMT_CREDIT"] / fe_v2["AMT_INCOME_TOTAL"].replace(0, np.nan)
    fe_v2["ratio_credit_annuity"] = fe_v2["AMT_CREDIT"] / fe_v2["AMT_ANNUITY"].replace(0, np.nan)
    fe_v2["ratio_days_employed_age"] = fe_v2["DAYS_EMPLOYED"] / fe_v2["DAYS_BIRTH"].replace(0, np.nan)

    interaction_pairs = []
    for it in top_interactions:
        if isinstance(it, dict) and "feature_1" in it and "feature_2" in it:
            colA = it["feature_1"]
            colB = it["feature_2"]
            if colA in fe_v2.columns and colB in fe_v2.columns:
                interaction_pairs.append((colA, colB))
        if len(interaction_pairs) >= 2:
            break

    print("Using SHAP-informed interaction pairs:", interaction_pairs)

    for colA, colB in interaction_pairs:
        fe_v2[f"ratio_{colA}_{colB}"] = fe_v2[colA] / fe_v2[colB].replace(0, np.nan)
        fe_v2[f"ratio_{colB}_{colA}"] = fe_v2[colB] / fe_v2[colA].replace(0, np.nan)
        fe_v2[f"prod_{colA}_{colB}"] = fe_v2[colA] * fe_v2[colB]

    print("FE_v2_test:", fe_v2.shape)
    return fe_v2
