# src/deployment/data_loader.py
# RU: Загрузка и подготовка данных / EN: Data loading and preparation

from pathlib import Path
from typing import Iterable

import pandas as pd
import sys

from .utils import load_pkl


def _normalize_ids(ids: Iterable[int] | None):
    """RU: Преобразование списка ID / EN: Normalize ID list"""
    if ids is None:
        return None
    ids_list = list(set(int(x) for x in ids))
    return ids_list


def load_and_clean_test(project_root: Path, ids: Iterable[int] | None = None):
    """RU: Загрузка и очистка теста / EN: Load and clean test"""
    sys.path.append(str(project_root / "src"))
    from cleaning_stage1 import clean_home_credit_table  # noqa

    raw_test_path = project_root / "data" / "raw" / "application_test.csv"
    df_test_raw = pd.read_csv(raw_test_path)
    print("Raw test:", df_test_raw.shape)

    ids_list = _normalize_ids(ids)
    if ids_list is not None:
        df_test_raw = df_test_raw[df_test_raw["SK_ID_CURR"].isin(ids_list)].copy()
        if df_test_raw.empty:
            raise ValueError(f"No SK_ID_CURR from {ids_list} found in application_test.csv")
        print(f"Filtered test to {len(ids_list)} IDs → shape:", df_test_raw.shape)

    df_test_clean = clean_home_credit_table(df_test_raw, "application_test")
    print("Clean test:", df_test_clean.shape)
    return df_test_raw, df_test_clean


def load_stage1_tables(project_root: Path, config: dict, ids: Iterable[int] | None = None):
    """RU: Загрузка очищенных таблиц Stage 1 / EN: Load Stage 1 cleaned tables"""
    stage1_path = project_root / config["stages"]["stage1"]["output_path"]

    bureau = load_pkl(stage1_path / "cleaned_bureau.pkl")
    previous_application = load_pkl(stage1_path / "cleaned_previous_application.pkl")
    installments_payments = load_pkl(stage1_path / "cleaned_installments_payments.pkl")
    credit_card_balance = load_pkl(stage1_path / "cleaned_credit_card_balance.pkl")

    ids_list = _normalize_ids(ids)
    if ids_list is not None:
        bureau = bureau[bureau["SK_ID_CURR"].isin(ids_list)]
        previous_application = previous_application[previous_application["SK_ID_CURR"].isin(ids_list)]
        installments_payments = installments_payments[installments_payments["SK_ID_CURR"].isin(ids_list)]
        credit_card_balance = credit_card_balance[credit_card_balance["SK_ID_CURR"].isin(ids_list)]
        print("Filtered Stage 1 tables for given IDs.")

    print("bureau:", bureau.shape)
    print("previous_application:", previous_application.shape)
    print("installments_payments:", installments_payments.shape)
    print("credit_card_balance:", credit_card_balance.shape)

    return bureau, previous_application, installments_payments, credit_card_balance
