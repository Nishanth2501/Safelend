from __future__ import annotations
import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

AGGS = [
    ("bureau_agg.parquet",          "BUR"),
    ("bureau_balance_agg.parquet",  "BB"),
    ("previous_agg.parquet",        "PREV"),
    ("pos_agg.parquet",             "POS"),
    ("credit_card_agg.parquet",     "CC"),
    ("installments_agg.parquet",    "INST"),
]

def load_app(kind: str) -> pd.DataFrame:
    fn = INTERIM / f"application_{kind}_clean.parquet"
    df = pd.read_parquet(fn)
    return df

def load_agg(fn: str) -> pd.DataFrame:
    return pd.read_parquet(INTERIM / fn)

def left_join_all(app: pd.DataFrame) -> pd.DataFrame:
    out = app.copy()
    for fn, _prefix in AGGS:
        agg = load_agg(fn)
        # ensure unique key
        if agg.duplicated("SK_ID_CURR").any():
            agg = agg.drop_duplicates("SK_ID_CURR")
        out = out.merge(agg, on="SK_ID_CURR", how="left")
    return out

def add_core_ratios(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    if {"AMT_CREDIT","AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["APP_CREDIT_TO_INCOME"]  = df["AMT_CREDIT"]  / (df["AMT_INCOME_TOTAL"]+eps)
    if {"AMT_ANNUITY","AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["APP_ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"]+eps)
    if {"AMT_ANNUITY","AMT_CREDIT"}.issubset(df.columns):
        df["APP_ANNUITY_TO_CREDIT"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"]+eps)
    # age / employment (years)
    if "DAYS_BIRTH" in df.columns:
        df["APP_AGE_YEARS"] = (-df["DAYS_BIRTH"]).clip(lower=0) / 365.0
    if "DAYS_EMPLOYED" in df.columns:
        df["APP_EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"]).clip(lower=0) / 365.0
    # documents total
    flag_doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if flag_doc_cols:
        df["APP_DOC_COUNT"] = df[flag_doc_cols].sum(axis=1)
    return df

def final_impute(df: pd.DataFrame) -> pd.DataFrame:
    # After joins, new NaNs appear in agg columns → fill sensibly:
    num = df.select_dtypes(include=["number"]).columns
    cat = df.select_dtypes(exclude=["number"]).columns
    df[cat] = df[cat].fillna("Unknown")
    for c in num:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df

def align_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Ensure test has all train columns except TARGET; add missing with neutral values
    y = train["TARGET"]
    X_train = train.drop(columns=["TARGET"])
    X_test  = test.copy()

    # One-hot encoding will later handle categoricals; for now keep raw (tree models tolerate)
    train_cols = X_train.columns
    test_cols  = X_test.columns
    missing_in_test = [c for c in train_cols if c not in test_cols]
    for c in missing_in_test:
        # neutral imputation (0); safe for numeric; categorical will be handled later
        X_test[c] = 0
    # Optional: drop extras in test not in train (rare)
    extra_in_test = [c for c in test_cols if c not in train_cols]
    if extra_in_test:
        X_test = X_test.drop(columns=extra_in_test)
    # Reorder
    X_test = X_test[X_train.columns]
    return pd.concat([X_train, y], axis=1), X_test

def main():
    # Load base apps
    app_tr = load_app("train")  # has TARGET
    app_te = load_app("test")   # no TARGET

    # Merge aggregates
    tr = left_join_all(app_tr)
    te = left_join_all(app_te)

    # Core ratios / features
    tr = add_core_ratios(tr)
    te = add_core_ratios(te)

    # Final impute post-join
    # (keep TARGET aside so we don't impute over it)
    y = tr["TARGET"]
    tr_feats = tr.drop(columns=["TARGET"])
    tr_feats = final_impute(tr_feats)
    tr = pd.concat([tr_feats, y], axis=1)
    te = final_impute(te)

    # Align columns between train and test (minus TARGET)
    tr, te = align_columns(tr, te)

    # Save
    tr.to_parquet(PROCESSED / "train_ready.parquet", index=False)
    te.to_parquet(PROCESSED / "test_ready.parquet", index=False)

    # Small samples as CSV for quick peeks
    tr.head(2000).to_csv(PROCESSED / "train_ready_sample.csv", index=False)
    te.head(2000).to_csv(PROCESSED / "test_ready_sample.csv", index=False)

    print("Saved:")
    print(" - data/processed/train_ready.parquet (+ sample CSV)")
    print(" - data/processed/test_ready.parquet  (+ sample CSV)")
    print(f"Train rows: {len(tr)} | Test rows: {len(te)} | Train cols: {tr.shape[1]} | Test cols: {te.shape[1]}")

if __name__ == "__main__":
    main()