# src/deployment/encoding.py
# RU: Кодирование признаков / EN: Feature encoding

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_features(fe_v3: pd.DataFrame) -> pd.DataFrame:
    """RU: Кодирование признаков / EN: Feature encoding"""
    fe_enc = fe_v3.copy()

    categorical_cols = fe_enc.select_dtypes(include=["object", "category"]).columns.tolist()
    print("Categorical cols (test):", categorical_cols)

    low_cardinality = [c for c in categorical_cols if fe_enc[c].nunique() <= 10]
    high_cardinality = [c for c in categorical_cols if fe_enc[c].nunique() > 10]

    print("Low-cardinality:", low_cardinality)
    print("High-cardinality:", high_cardinality)

    if low_cardinality:
        fe_enc = pd.get_dummies(
            fe_enc,
            columns=low_cardinality,
            drop_first=True,  # RU: как в дипломе / EN: as in diploma
        )
        print("One-hot encoding applied (drop_first=True).")

    for col in high_cardinality:
        le = LabelEncoder()
        fe_enc[col] = fe_enc[col].astype(str).fillna("missing")
        fe_enc[col] = le.fit_transform(fe_enc[col])
    print("Label encoding applied for high-cardinality.")

    print("FE_v3_test_encoded:", fe_enc.shape)
    return fe_enc
