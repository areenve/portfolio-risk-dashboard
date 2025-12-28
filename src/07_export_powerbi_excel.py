from pathlib import Path
import pandas as pd

IN_DIR = Path("data/processed")
OUT_XLSX = IN_DIR / "powerbi_upload.xlsx"

TABLES = {
    "dim_date": IN_DIR / "dim_date.csv",
    "dim_asset": IN_DIR / "dim_asset.csv",
    "portfolio_daily": IN_DIR / "portfolio_daily.csv",
    "risk_daily": IN_DIR / "risk_daily.csv",
    "risk_contrib": IN_DIR / "risk_contrib.csv",
    "stress_daily": IN_DIR / "stress_daily.csv",
}

# Which columns must be dates in each sheet
DATE_COLS = {
    "dim_date": ["date"],
    "portfolio_daily": ["date"],
    "risk_daily": ["date"],
    "risk_contrib": ["date"],
    "stress_daily": ["date"],
}

def load_table(path: Path) -> pd.DataFrame:
    # read normally; we'll convert types after
    return pd.read_csv(path)

def enforce_types(name: str, df: pd.DataFrame) -> pd.DataFrame:
    # Convert required date columns to datetime (Excel date)
    for c in DATE_COLS.get(name, []):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()

    # Optional: make sure numeric columns are numeric
    for c in df.columns:
        if c not in DATE_COLS.get(name, []) and c != "ticker" and c != "asset_class" and c != "sector" and c != "scenario":
            df[c] = pd.to_numeric(df[c], errors="ignore")

    return df

def main():
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(
        OUT_XLSX,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd",
        date_format="yyyy-mm-dd",
    ) as writer:
        for sheet, path in TABLES.items():
            df = load_table(path)
            df = enforce_types(sheet, df)

            # Write to Excel
            df.to_excel(writer, sheet_name=sheet, index=False)

            # Format the date columns explicitly (belt & suspenders)
            workbook = writer.book
            worksheet = writer.sheets[sheet]
            date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd"})

            for c in DATE_COLS.get(sheet, []):
                if c in df.columns:
                    col_idx = df.columns.get_loc(c)
                    worksheet.set_column(col_idx, col_idx, 12, date_fmt)

    print(f"Saved: {OUT_XLSX}")

if __name__ == "__main__":
    main()

