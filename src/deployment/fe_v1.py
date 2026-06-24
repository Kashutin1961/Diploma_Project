# src/deployment/fe_v1.py
# RU: FE_v1 — агрегаты / EN: FE_v1 — aggregations

import pandas as pd


def build_fe_v1(df_test_clean: pd.DataFrame,
                bureau: pd.DataFrame,
                previous_application: pd.DataFrame,
                installments_payments: pd.DataFrame,
                credit_card_balance: pd.DataFrame) -> pd.DataFrame:
    """RU: FE_v1 — агрегаты / EN: FE_v1 — aggregations"""
    # bureau aggregates
    bureau_agg = bureau.groupby("SK_ID_CURR").agg({
        "AMT_CREDIT_SUM": ["mean", "max", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "max", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "max", "sum"],
        "DAYS_CREDIT": ["mean", "min", "max"],
        "DAYS_CREDIT_ENDDATE": ["mean", "min", "max"],
    })
    bureau_agg.columns = ["bureau_" + "_".join(col).lower() for col in bureau_agg.columns]
    bureau_agg.reset_index(inplace=True)

    # previous_application aggregates
    previous_agg = previous_application.groupby("SK_ID_CURR").agg({
        "AMT_APPLICATION": ["mean", "sum", "max"],
        "AMT_CREDIT": ["mean", "sum", "max"],
        "AMT_GOODS_PRICE": ["mean", "sum", "max"],
        "NAME_CONTRACT_STATUS": lambda x: (x == "Approved").sum(),
    })
    previous_agg.columns = [
        "prev_" + "_".join(col).lower() if isinstance(col, tuple) else "prev_" + col
        for col in previous_agg.columns
    ]
    previous_agg.rename(columns={"prev_<lambda>": "prev_approved_count"}, inplace=True)
    previous_agg.reset_index(inplace=True)

    # installments_payments aggregates
    inst = installments_payments.copy()
    inst["payment_ratio"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]
    inst["overdue_days"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    installments_agg = inst.groupby("SK_ID_CURR").agg({
        "payment_ratio": ["mean", "min", "max"],
        "overdue_days": ["mean", "max"],
        "AMT_PAYMENT": ["sum"],
    })
    installments_agg.columns = ["inst_" + "_".join(col).lower() for col in installments_agg.columns]
    installments_agg.reset_index(inplace=True)

    # credit_card_balance aggregates
    cc = credit_card_balance.copy()
    cc["limit_usage"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]
    credit_agg = cc.groupby("SK_ID_CURR").agg({
        "AMT_BALANCE": ["mean", "max"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean"],
        "limit_usage": ["mean", "max"],
        "SK_DPD": ["mean", "max"],
    })
    credit_agg.columns = ["cc_" + "_".join(col).lower() for col in credit_agg.columns]
    credit_agg.reset_index(inplace=True)

    fe_v1 = df_test_clean.copy()
    fe_v1 = fe_v1.merge(bureau_agg, on="SK_ID_CURR", how="left")
    fe_v1 = fe_v1.merge(previous_agg, on="SK_ID_CURR", how="left")
    fe_v1 = fe_v1.merge(installments_agg, on="SK_ID_CURR", how="left")
    fe_v1 = fe_v1.merge(credit_agg, on="SK_ID_CURR", how="left")

    print("FE_v1_test:", fe_v1.shape)
    return fe_v1
