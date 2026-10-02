"""Inspect the AI231 ME2 dataset spreadsheet (data/ai231_me2_dataset.xlsx).

Prints per-sheet: shape, columns, dtypes, nulls, distinct values for
categorical columns, and head rows. Usage: python tools/inspect_dataset.py
"""

import pandas as pd

PATH = r"C:\Users\Jan\Desktop\AI 231 ME2 V3\data\ai231_me2_dataset.xlsx"


def main() -> None:
    xl = pd.ExcelFile(PATH)
    print("SHEETS:", xl.sheet_names)
    for s in xl.sheet_names:
        df = xl.parse(s)
        print("=" * 70)
        print(f"SHEET: {s!r}  shape={df.shape}")
        print("COLUMNS:", list(df.columns))
        print("DTYPES:")
        print(df.dtypes)
        print("NULLS:")
        print(df.isna().sum())
        for c in df.columns:
            nu = df[c].nunique()
            if nu <= 60:
                vals = sorted(df[c].dropna().astype(str).unique().tolist())
                print(f"-- distinct [{c}] ({nu}):", vals)
            else:
                print(f"-- numeric-ish [{c}]: unique={nu}")
        print("HEAD (10):")
        print(df.head(10).to_string())


if __name__ == "__main__":
    main()
