# src/models/utils/schema_driven_preprocessing.py
# Stage 9 ONLY — schema-driven preprocessing for SHAP and debugging
# NOT used in Stage 10 production pipeline

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class SchemaDrivenPreprocessor:
    """
    Stage 9 ONLY:
    - loads schema.json from Stage 8
    - rebuilds engineered features for SHAP analysis
    - applies imputation + scaling for interpretation
    NOT used in production (Stage 10).
    """

    def __init__(self, schema_path: str):
        self.schema_path = Path(schema_path)

        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        # Final selected features (SK_ID_CURR removed)
        self.feature_order = [
            f for f in self.schema["final_selected_features"]
            if f != "SK_ID_CURR"
        ]

        # Imputer + scaler
        strategy = self.schema["preprocessing_rules"]["imputation_strategy"]
        self.imputer = SimpleImputer(strategy=strategy)
        self.scaler = StandardScaler()

    # ------------------------------------------------------------
    # BUILD FEATURES (Stage 9 only)
    # ------------------------------------------------------------
    def build_features(self, root: Path) -> pd.DataFrame:
        """
        Build engineered features for SHAP analysis.
        Uses Stage 8 feature engineering.
        """
        from src.features.feature_engineering import build_full_feature_matrix

        df = build_full_feature_matrix(root, selected_features=None)

        if self.schema["preprocessing_rules"]["replace_inf_with_nan"]:
            df = df.replace([np.inf, -np.inf], np.nan)

        return df

    # ------------------------------------------------------------
    # FIT (Stage 9 only)
    # ------------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        """
        Fit imputer + scaler using schema-defined feature order.
        """
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = np.nan

        df = df[self.feature_order]

        self.imputer.fit(df)
        X_num = self.imputer.transform(df)
        self.scaler.fit(X_num)

    # ------------------------------------------------------------
    # TRANSFORM (Stage 9 only)
    # ------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply schema-defined preprocessing to new data.
        """

        for col in self.feature_order:
            if col not in df.columns:
                df[col] = np.nan

        df = df[self.feature_order]

        df = df.replace([np.inf, -np.inf], np.nan)

        X_num = self.imputer.transform(df)
        X_scaled = self.scaler.transform(X_num)

        return X_scaled
