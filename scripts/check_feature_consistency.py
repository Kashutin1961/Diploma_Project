"""
check_feature_consistency.py

Purpose:
    Ensure that TRAIN, TEST, and SINGLE-SAMPLE preprocessing pipelines
    produce feature matrices that match the schema exported in Stage 8.

Usage:
    python scripts/check_feature_consistency.py
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import your actual preprocessing functions
# Adjust these imports to match your project structure
from src.preprocessing.build_features import (
    build_features_train,
    build_features_test,
    build_features_single
)


# ============================================================
# 1. Load schema
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "data" / "stage_outputs" / "stage8_optimization" / "feature_schema.json"

if not SCHEMA_PATH.exists():
    print(f"[ERROR] Schema file not found: {SCHEMA_PATH}")
    sys.exit(1)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)

expected_order = schema["final_feature_order"]
expected_set = set(expected_order)
expected_dtype = schema.get("dtype_enforcement", "float32")


# ============================================================
# 2. Helper: validate a feature matrix
# ============================================================

def validate_matrix(name: str, df: pd.DataFrame):
    print(f"\n=== Checking {name} ===")

    # --- Check column set ---
    actual_set = set(df.columns)
    missing = expected_set - actual_set
    extra = actual_set - expected_set

    if missing:
        print(f"[FAIL] Missing columns in {name}: {sorted(missing)}")
        return False

    if extra:
        print(f"[FAIL] Extra columns in {name}: {sorted(extra)}")
        return False

    # --- Check column order ---
    if list(df.columns) != expected_order:
        print("[FAIL] Column order mismatch in", name)
        print("Expected:", expected_order[:10], "...")
        print("Actual:  ", list(df.columns)[:10], "...")
        return False

    # --- Check dtype ---
    wrong_dtype = [
        col for col in df.columns
        if str(df[col].dtype) != expected_dtype
    ]

    if wrong_dtype:
        print(f"[FAIL] Wrong dtypes in {name}: {wrong_dtype}")
        print("Expected dtype:", expected_dtype)
        return False

    print(f"[OK] {name} is consistent.")
    return True


# ============================================================
# 3. Build features via all three paths
# ============================================================

print("\n=== Building TRAIN features ===")
df_train = build_features_train(sample_size=500)   # small sample is enough

print("\n=== Building TEST features ===")
df_test = build_features_test(sample_size=500)

print("\n=== Building SINGLE-SAMPLE features ===")
sample_row = df_train.iloc[0].to_dict()
df_single = build_features_single(sample_row)


# ============================================================
# 4. Validate all three
# ============================================================

ok_train = validate_matrix("TRAIN", df_train)
ok_test = validate_matrix("TEST", df_test)
ok_single = validate_matrix("SINGLE", df_single)

# ============================================================
# 5. Final result
# ============================================================

if ok_train and ok_test and ok_single:
    print("\n=== ALL CHECKS PASSED ===")
    sys.exit(0)
else:
    print("\n=== FEATURE CONSISTENCY CHECK FAILED ===")
    sys.exit(1)
