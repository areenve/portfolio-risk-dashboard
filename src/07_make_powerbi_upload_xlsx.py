import pandas as pd
from pathlib import Path

FILES = {
    "portfolio_daily": "data/processed/portfolio_daily.csv",
    "risk_daily": "data/processed/risk_daily.csv",
    "risk_contrib": "data/processed/risk_contrib.csv",
    "stress_daily": "data/processed/stress_daily.csv",
    # dims (if you created them):
    "dim_date": "data/processed/dim_date.csv",
    "dim_asset": "data/processed/dim_asset.csv",
}

OUT = "data/processed/powerbi_upload.xlsx"

Path("data/processed").mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    for sheet, path in FILES.items():
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            df.to_excel(w, sheet_name=sheet, index=False)

print(f"Saved: {OUT}")
