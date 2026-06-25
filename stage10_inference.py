# stage10_inference.py
# RU: Точка входа деплоймента (Stage 10) / EN: Deployment (Stage 10) entry point

import argparse
from pathlib import Path

import pandas as pd

from src.deployment.config_loader import load_config, load_stage5_summary
from src.deployment.data_loader import load_and_clean_test, load_stage1_tables
from src.deployment.fe_v1 import build_fe_v1
from src.deployment.fe_v2 import build_fe_v2
from src.deployment.fe_v3 import build_fe_v3
from src.deployment.encoding import encode_features
from src.deployment.missing import handle_missing
from src.deployment.scaling import scale_features
from src.deployment.model_loader import load_models_and_features
from src.deployment.prediction import align_features, predict_ensemble, save_submission


def parse_args():
    """RU: Парсинг аргументов CLI / EN: CLI argument parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sk-id",
        type=int,
        default=None,
        help="Single SK_ID_CURR for single-record inference",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Path to CSV with SK_ID_CURR column for subset inference",
    )
    return parser.parse_args()


def get_ids_from_args(args) -> list[int] | None:
    """RU: Получение списка ID из аргументов / EN: Get ID list from CLI args"""
    if args.sk_id is not None and args.ids_file is not None:
        raise ValueError("Use either --sk-id or --ids-file, not both.")

    if args.sk_id is not None:
        return [args.sk_id]

    if args.ids_file is not None:
        df_ids = pd.read_csv(args.ids_file)
        if "SK_ID_CURR" not in df_ids.columns:
            raise ValueError("ids-file must contain 'SK_ID_CURR' column.")
        ids = df_ids["SK_ID_CURR"].astype(int).tolist()
        print(f"Loaded {len(ids)} IDs from {args.ids_file}")
        return ids

    return None  # RU: полный тест / EN: full test


def main():
    args = parse_args()
    ids = get_ids_from_args(args)

    project_root = Path(".").resolve()
    print("Project root:", project_root)

    config = load_config(project_root)
    top_interactions, nonlinear_feats = load_stage5_summary(project_root, config)

    df_test_raw, df_test_clean = load_and_clean_test(project_root, ids)
    bureau, previous_application, installments_payments, credit_card_balance = load_stage1_tables(
        project_root, config, ids
    )

    fe_v1 = build_fe_v1(df_test_clean, bureau, previous_application, installments_payments, credit_card_balance)
    fe_v2 = build_fe_v2(fe_v1, top_interactions)
    fe_v3 = build_fe_v3(fe_v2, nonlinear_feats)
    fe_enc = encode_features(fe_v3)
    fe_mv = handle_missing(fe_enc)
    fe_final = scale_features(fe_mv, apply_scaling=False)

    models, meta_model, final_features, base_model_names = load_models_and_features(project_root)
    df_test_final = align_features(fe_final, final_features)

    # restore SK_ID_CURR / восстановление SK_ID_CURR
    df_test_final["SK_ID_CURR"] = df_test_raw["SK_ID_CURR"].values

    X_test = df_test_final[final_features].values
    meta_proba = predict_ensemble(models, meta_model, base_model_names, X_test)

    # single-record mode / режим одного клиента
    if ids is not None and len(ids) == 1:
        sk_id = ids[0]
        print("\n=== SINGLE RECORD PREDICTION ===")
        print(f"SK_ID_CURR: {sk_id}")
        print(f"PREDICTION: {float(meta_proba[0]):.6f}")
        return

    # subset or full / подмножество или полный тест
    save_submission(project_root, df_test_final["SK_ID_CURR"], meta_proba)


if __name__ == "__main__":
    main()
