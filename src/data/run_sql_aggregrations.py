from __future__ import annotations
import duckdb
from pathlib import Path

SQL_DIR = Path("sql")
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS = [
    ("bureau_agg.sql",            "data/interim/bureau_agg.parquet"),
    ("bureau_balance_agg.sql",    "data/interim/bureau_balance_agg.parquet"),
    ("previous_agg.sql",          "data/interim/previous_agg.parquet"),
    ("pos_agg.sql",               "data/interim/pos_agg.parquet"),
    ("credit_card_agg.sql",       "data/interim/credit_card_agg.parquet"),
    ("installments_agg.sql",      "data/interim/installments_agg.parquet"),
]

def main():
    con = duckdb.connect()
    for sql_file, out_path in JOBS:
        sql_path = SQL_DIR / sql_file
        print(f"→ Running {sql_file}")
        query = open(sql_path, "r").read().strip()
        # Remove trailing semicolon if present
        if query.endswith(';'):
            query = query[:-1]
        # Run the query as a subselect and write directly to Parquet
        con.execute(f"COPY ({query}) TO '{out_path}' (FORMAT PARQUET);")
        # quick sanity check
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
        print(f"   wrote {n} rows to {out_path}")
    print("All aggregations done.")

if __name__ == "__main__":
    main()