from __future__ import annotations

import pandas as pd
from pathlib import Path


def main() -> None:
    # Date dimension from portfolio dates
    port = pd.read_csv("data/processed/portfolio_daily.csv")
    d = pd.to_datetime(port["date"]).drop_duplicates().sort_values()
    dim_date = pd.DataFrame({"date": d.dt.date})
    dim_date["year"] = d.dt.year
    dim_date["month"] = d.dt.month
    dim_date["month_name"] = d.dt.strftime("%b")
    dim_date["quarter"] = d.dt.quarter
    dim_date["week"] = d.dt.isocalendar().week.astype(int)

    # Asset dimension from universe (but tickers in your data have .US)
    uni = pd.read_csv("data/sample/universe.csv")
    uni["ticker"] = uni["ticker"].astype(str).str.strip().str.upper()
    # attach .US if missing (matches your pipeline tickers)
    uni["ticker"] = uni["ticker"].apply(lambda x: x if "." in x or x.startswith("^") else f"{x}.US")
    dim_asset = uni[["ticker", "asset_class", "sector"]].drop_duplicates()

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    dim_date.to_csv("data/processed/dim_date.csv", index=False)
    dim_asset.to_csv("data/processed/dim_asset.csv", index=False)

    Path("data/sample").mkdir(parents=True, exist_ok=True)
    dim_date.tail(200).to_csv("data/sample/dim_date_sample.csv", index=False)
    dim_asset.to_csv("data/sample/dim_asset_sample.csv", index=False)

    print("Saved: data/processed/dim_date.csv")
    print("Saved: data/processed/dim_asset.csv")


if __name__ == "__main__":
    main()
