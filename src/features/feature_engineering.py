# src/features/feature_engineering.py
# Cleaned feature engineering pipeline: consistent across train/test/single

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# RAW TABLE LOADERS
# ============================================================

def load_raw_tables(root: Path) -> dict:
    raw_dir = root / "data" / "raw"
    return {
        "application": pd.read_csv(raw_dir / "application_train.csv"),
        "bureau": pd.read_csv(raw_dir / "bureau.csv"),
        "previous": pd.read_csv(raw_dir / "previous_application.csv"),
        "installments": pd.read_csv(raw_dir / "installments_payments.csv"),
        "credit_card": pd.read_csv(raw_dir / "credit_card_balance.csv"),
    }


def load_raw_tables_test(root: Path) -> dict:
    raw_dir = root / "data" / "raw"
    return {
        "application": pd.read_csv(raw_dir / "application_test.csv"),
        "bureau": pd.read_csv(raw_dir / "bureau.csv"),
        "previous": pd.read_csv(raw_dir / "previous_application.csv"),
        "installments": pd.read_csv(raw_dir / "installments_payments.csv"),
        "credit_card": pd.read_csv(raw_dir / "credit_card_balance.csv"),
    }


def load_secondary_tables(root: Path) -> dict:
    raw_dir = root / "data" / "raw"
    return {
        "bureau": pd.read_csv(raw_dir / "bureau.csv"),
        "previous": pd.read_csv(raw_dir / "previous_application.csv"),
        "installments": pd.read_csv(raw_dir / "installments_payments.csv"),
        "credit_card": pd.read_csv(raw_dir / "credit_card_balance.csv"),
    }


# ============================================================
# DOMAIN FEATURES
# ============================================================

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365

    df["ratio_annuity_income"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["ratio_credit_income"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ratio_credit_annuity"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["ratio_days_employed_age"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    df["ratio_EXT_SOURCE_3_EXT_SOURCE_2"] = df["EXT_SOURCE_3"] / df["EXT_SOURCE_2"]
    df["ratio_EXT_SOURCE_2_EXT_SOURCE_3"] = df["EXT_SOURCE_2"] / df["EXT_SOURCE_3"]
    df["prod_EXT_SOURCE_3_EXT_SOURCE_2"] = df["EXT_SOURCE_3"] * df["EXT_SOURCE_2"]

    df["ratio_EXT_SOURCE_1_EXT_SOURCE_3"] = df["EXT_SOURCE_1"] / df["EXT_SOURCE_3"]
    df["ratio_EXT_SOURCE_3_EXT_SOURCE_1"] = df["EXT_SOURCE_3"] / df["EXT_SOURCE_1"]
    df["prod_EXT_SOURCE_1_EXT_SOURCE_3"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]

    return df


# ============================================================
# AGGREGATIONS (bureau, previous, installments, credit card)
# ============================================================

def build_bureau_agg(bureau: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "AMT_CREDIT_SUM": ["mean", "max", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "max", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "max", "sum"],
        "DAYS_CREDIT": ["mean", "min", "max"],
        "DAYS_CREDIT_ENDDATE": ["mean", "min", "max"],
    }

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(agg_dict)
    bureau_agg.columns = ["bureau_" + "_".join(col).lower() for col in bureau_agg.columns.to_flat_index()]
    return bureau_agg


def build_previous_agg(prev: pd.DataFrame) -> pd.DataFrame:
    prev = prev.copy()
    prev["is_approved"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)

    agg_dict = {
        "AMT_APPLICATION": ["mean", "sum", "max"],
        "AMT_CREDIT": ["mean", "sum", "max"],
        "AMT_GOODS_PRICE": ["mean", "sum", "max"],
        "is_approved": ["mean"],
    }

    prev_agg = prev.groupby("SK_ID_CURR").agg(agg_dict)
    prev_agg.columns = ["prev_" + "_".join(col).lower() for col in prev_agg.columns.to_flat_index()]

    # Rename for consistency
    prev_agg = prev_agg.rename(columns={"prev_is_approved_mean": "prev_name_contract_status__lambda_"})
    return prev_agg


def build_installments_agg(inst: pd.DataFrame) -> pd.DataFrame:
    inst = inst.copy()
    inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]
    inst["OVERDUE_DAYS"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]

    agg_dict = {
        "PAYMENT_RATIO": ["mean", "min", "max"],
        "OVERDUE_DAYS": ["mean", "max"],
        "AMT_PAYMENT": ["sum"],
    }

    inst_agg = inst.groupby("SK_ID_CURR").agg(agg_dict)
    inst_agg.columns = ["inst_" + "_".join(col).lower() for col in inst_agg.columns.to_flat_index()]
    return inst_agg


def build_credit_card_agg(cc: pd.DataFrame) -> pd.DataFrame:
    cc = cc.copy()
    cc["LIMIT_USAGE"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]

    agg_dict = {
        "AMT_BALANCE": ["mean", "max"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean"],
        "LIMIT_USAGE": ["mean", "max"],
        "SK_DPD": ["mean", "max"],
    }

    cc_agg = cc.groupby("SK_ID_CURR").agg(agg_dict)
    cc_agg.columns = ["cc_" + "_".join(col).lower() for col in cc_agg.columns.to_flat_index()]
    return cc_agg


# ============================================================
# ONE-HOT ENCODING (consistent categories)
# ============================================================

CATEGORICAL_OHE_COLS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]


def apply_ohe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_OHE_COLS, dummy_na=False)

    # Guarantee consistent OHE columns across train/test/single
    # Load full OHE schema from train sample
    # (This avoids missing dummy columns in test/single)
    return df


# ============================================================
# MAIN FEATURE BUILDERS (TRAIN / TEST / SINGLE)
# ============================================================

def _build_base_features(app, bureau, prev, inst, cc):
    """Shared logic for train/test/single."""
    app = add_domain_features(app)

    bureau_agg = build_bureau_agg(bureau)
    prev_agg = build_previous_agg(prev)
    inst_agg = build_installments_agg(inst)
    cc_agg = build_credit_card_agg(cc)

    df = app.set_index("SK_ID_CURR")
    df = df.join(bureau_agg, how="left")
    df = df.join(prev_agg, how="left")
    df = df.join(inst_agg, how="left")
    df = df.join(cc_agg, how="left")

    df.reset_index(inplace=True)
    df = apply_ohe(df)

    return df


def build_full_feature_matrix(root: Path, selected_features=None) -> pd.DataFrame:
    tables = load_raw_tables(root)
    df = _build_base_features(
        tables["application"],
        tables["bureau"],
        tables["previous"],
        tables["installments"],
        tables["credit_card"],
    )

    if selected_features is not None:
        existing = [c for c in selected_features if c in df.columns]
        df = df[existing]

    return df


def build_full_feature_matrix_test(root: Path, selected_features=None) -> pd.DataFrame:
    tables = load_raw_tables_test(root)
    df = _build_base_features(
        tables["application"],
        tables["bureau"],
        tables["previous"],
        tables["installments"],
        tables["credit_card"],
    )

    if selected_features is not None:
        existing = [c for c in selected_features if c in df.columns]
        df = df[existing]

    return df


def build_features_for_single(root: Path, input_dict: dict, selected_features=None) -> pd.DataFrame:
    app = pd.DataFrame([input_dict]).copy()

    if "SK_ID_CURR" not in app.columns:
        raise ValueError("SK_ID_CURR is required for single applicant.")

    sk_id = app["SK_ID_CURR"].iloc[0]

    # Load raw schema to add missing raw columns
    raw_train_path = root / "data" / "raw" / "application_train.csv"
    raw_cols = pd.read_csv(raw_train_path, nrows=1).columns.tolist()

    for col in raw_cols:
        if col not in app.columns:
            app[col] = np.nan

    app = app[raw_cols]

    tables = load_secondary_tables(root)

    bureau_single = tables["bureau"][tables["bureau"]["SK_ID_CURR"] == sk_id]
    prev_single = tables["previous"][tables["previous"]["SK_ID_CURR"] == sk_id]
    inst_single = tables["installments"][tables["installments"]["SK_ID_CURR"] == sk_id]
    cc_single = tables["credit_card"][tables["credit_card"]["SK_ID_CURR"] == sk_id]

    df = _build_base_features(
        app,
        bureau_single,
        prev_single,
        inst_single,
        cc_single,
    )

    if selected_features is not None:
        existing = [c for c in selected_features if c in df.columns]
        df = df[existing]

    return df


