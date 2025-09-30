from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)

SENTINEL_DAY = 365243

# ----------------------------
# helpers (shared across files)
# ----------------------------
def daylike_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("DAYS_") or c in ("DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT")]

def replace_day_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    for c in daylike_cols(df):
        if c in df.columns:
            df[c] = df[c].replace(SENTINEL_DAY, pd.NA)
    return df

def impute_basic(df: pd.DataFrame) -> pd.DataFrame:
    # categorical → "Unknown"; numeric → median
    # Special handling for day-like columns to avoid type conflicts
    day_cols = daylike_cols(df)
    
    # Handle day-like columns first (they might be object type after sentinel replacement)
    for c in day_cols:
        if c in df.columns and df[c].isna().any():
            median_val = df[c].median()
            if pd.isna(median_val):
                # If all values are NA, use 0 as a reasonable default
                df.loc[:, c] = df[c].fillna(0)
            else:
                # For object dtype columns, use a different approach to avoid warnings
                if df[c].dtype == 'object':
                    # Create a mask for missing values and fill them
                    mask = df[c].isna()
                    df.loc[mask, c] = median_val
                else:
                    df.loc[:, c] = df[c].fillna(median_val)
    
    # Now handle regular categorical and numeric columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Remove day-like columns from categorical handling since we already processed them
    cat_cols = [c for c in cat_cols if c not in day_cols]
    num_cols = df.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    
    # Handle remaining categorical columns
    if cat_cols:
        df.loc[:, cat_cols] = df[cat_cols].fillna("Unknown")
    
    # Handle remaining numeric columns
    for c in num_cols:
        if df[c].isna().any():
            df.loc[:, c] = df[c].fillna(df[c].median())
    
    # Use pandas object inference to properly handle type conversion
    df = df.infer_objects(copy=False)
    return df

def winsorize(df: pd.DataFrame, cols: list[str], lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            lo = df[c].quantile(lower_q)
            hi = df[c].quantile(upper_q)
            df.loc[:, c] = df[c].clip(lo, hi)
    return df

def summarize(tag: str, df: pd.DataFrame):
    n, m = df.shape
    miss = df.isna().mean().sort_values(ascending=False).head(5)
    print(f"[{tag}] shape={n}x{m} | top5 NA%:\n{miss.round(3)}\n")

def clean_flags_to_int(df: pd.DataFrame, prefix="FLAG_") -> pd.DataFrame:
    # ensure FLAG_* columns are 0/1 ints (some are already)
    for c in [c for c in df.columns if c.startswith(prefix)]:
        if df[c].dtype == "O":  # strings 'Y'/'N' → keep for OHE later; don't coerce here
            continue
        df.loc[:, c] = df[c].fillna(0).astype(int)
    return df

# ----------------------------
# file-specific cleaners
# ----------------------------
def clean_application_train() -> pd.DataFrame:
    df = pd.read_csv(RAW / "application_train.csv")
    df = df.drop_duplicates()
    df = replace_day_sentinel(df)
    # Keep 'Y'/'N' in FLAG_OWN_CAR/REALTY as categories; enforce other FLAG_* to ints
    df = clean_flags_to_int(df, prefix="FLAG_")
    # Light outlier caps on key monetary columns
    df = winsorize(df, ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"])
    # Do not impute TARGET; only features
    feat_cols = [c for c in df.columns if c != "TARGET"]
    df[feat_cols] = impute_basic(df[feat_cols])
    summarize("application_train(clean)", df)
    df.to_parquet(INTERIM / "application_train_clean.parquet", index=False)
    return df

def clean_application_test() -> pd.DataFrame:
    df = pd.read_csv(RAW / "application_test.csv")
    df = df.drop_duplicates()
    df = replace_day_sentinel(df)
    df = clean_flags_to_int(df, prefix="FLAG_")
    df = winsorize(df, ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"])
    df = impute_basic(df)
    summarize("application_test(clean)", df)
    df.to_parquet(INTERIM / "application_test_clean.parquet", index=False)
    return df

def clean_bureau() -> pd.DataFrame:
    df = pd.read_csv(RAW / "bureau.csv")
    df = df.drop_duplicates()
    df = replace_day_sentinel(df)
    df = impute_basic(df)
    summarize("bureau(clean)", df)
    df.to_parquet(INTERIM / "bureau_clean.parquet", index=False)
    return df

def clean_bureau_balance() -> pd.DataFrame:
    df = pd.read_csv(RAW / "bureau_balance.csv")
    df = df.drop_duplicates()
    # no special day sentinel here; months are ints (negatives = history)
    df = impute_basic(df)
    summarize("bureau_balance(clean)", df)
    df.to_parquet(INTERIM / "bureau_balance_clean.parquet", index=False)
    return df

def clean_previous_application() -> pd.DataFrame:
    df = pd.read_csv(RAW / "previous_application.csv")
    df = df.drop_duplicates()
    df = replace_day_sentinel(df)
    # Normalize some crazy interest columns if present (clip extreme positives)
    df = winsorize(df, ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_GOODS_PRICE"])
    df = impute_basic(df)
    summarize("previous_application(clean)", df)
    df.to_parquet(INTERIM / "previous_application_clean.parquet", index=False)
    return df

def clean_pos_cash() -> pd.DataFrame:
    df = pd.read_csv(RAW / "POS_CASH_balance.csv")
    df = df.drop_duplicates()
    df = impute_basic(df)
    summarize("POS_CASH_balance(clean)", df)
    df.to_parquet(INTERIM / "pos_cash_balance_clean.parquet", index=False)
    return df

def clean_credit_card() -> pd.DataFrame:
    df = pd.read_csv(RAW / "credit_card_balance.csv")
    df = df.drop_duplicates()
    df = impute_basic(df)
    summarize("credit_card_balance(clean)", df)
    df.to_parquet(INTERIM / "credit_card_balance_clean.parquet", index=False)
    return df

def clean_installments() -> pd.DataFrame:
    df = pd.read_csv(RAW / "installments_payments.csv")
    df = df.drop_duplicates()
    df = replace_day_sentinel(df)  # safety, though days here are usually valid
    df = winsorize(df, ["AMT_INSTALMENT", "AMT_PAYMENT"])
    df = impute_basic(df)
    summarize("installments_payments(clean)", df)
    df.to_parquet(INTERIM / "installments_payments_clean.parquet", index=False)
    return df

def clean_columns_description() -> pd.DataFrame:
    # just load/save for completeness; useful for data dictionary
    # Handle encoding issues with the CSV file
    try:
        df = pd.read_csv(RAW / "HomeCredit_columns_description.csv")
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        df = pd.read_csv(RAW / "HomeCredit_columns_description.csv", encoding='latin-1')
    summarize("columns_description(pass-through)", df)
    df.to_parquet(INTERIM / "columns_description.parquet", index=False)
    return df

def main():
    print("=== SafeLend: cleaning raw datasets → data/interim/*.parquet ===")
    clean_application_train()
    clean_application_test()
    clean_bureau()
    clean_bureau_balance()
    clean_previous_application()
    clean_pos_cash()
    clean_credit_card()
    clean_installments()
    clean_columns_description()
    print("=== Done. Next step: SQL aggregations per SK_ID_CURR in sql/*.sql ===")

if __name__ == "__main__":
    main()