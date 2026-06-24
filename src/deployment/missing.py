# src/deployment/missing.py
# RU: Обработка пропусков / EN: Missing values handling

import numpy as np
import pandas as pd


def handle_missing(fe_enc: pd.DataFrame) -> pd.DataFrame:
    """RU: Полная обработка пропусков и бесконечностей
       EN: Full missing-value and infinity handling"""
    fe_mv = fe_enc.copy()

    # Replace infinities
    fe_mv = fe_mv.replace([np.inf, -np.inf], np.nan)

    # Fill ALL numeric NaNs with 0
    num_cols = fe_mv.select_dtypes(include=["int64", "float64"]).columns.tolist()
    fe_mv[num_cols] = fe_mv[num_cols].fillna(0)

    # Fill ALL categorical NaNs with "missing"
    cat_cols = fe_mv.select_dtypes(include=["object", "category"]).columns.tolist()
    fe_mv[cat_cols] = fe_mv[cat_cols].fillna("missing")

    print("Missing values fully handled (NaN → 0, inf → 0).")
    print("FE_v3_test_mv:", fe_mv.shape)
    return fe_mv
