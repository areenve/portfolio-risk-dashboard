from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def month_ends(dates: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dates).sort_values()
    # month-end markers based on available trading dates
    month = dt.dt.to_period("M")
    is_last = month != month.shift(-1)
    return dt[is_last].dt.date


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="data/sample/universe.csv")
    ap.add_argument("--prices", default="data/processed/prices.parquet")
    ap.add_argument("--out", default="data/processed/positions.csv")
    ap.add_argument("--scheme", choices=["equal"], default="equal")
    args = ap.parse_args()

    prices = pd.read_parquet(args.prices)
    if prices.empty:
        raise ValueError("prices.parquet is empty")

    # IMPORTANT: use tickers exactly as they appear in prices.parquet (e.g., GLD.US)
    tickers = sorted(prices["ticker"].dropna().unique().tolist())
    if not tickers:
        raise ValueError("No tickers found in prices.parquet")


    # Ensure we have an initial rebalance on the earliest available trading date
    min_date = pd.to_datetime(prices["date"]).min().date()

    rebalance_dates = list(month_ends(prices["date"]))

    # Add the initial date if it's not already included
    if min_date not in rebalance_dates:
        rebalance_dates = [min_date] + rebalance_dates


    n = len(tickers)
    w = 1.0 / n

    rows = []
    for d in rebalance_dates:
        for t in tickers:
            rows.append({"rebalance_date": d, "ticker": t, "weight": w})

    pos = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos.to_csv(out_path, index=False)

    # GitHub-safe sample (first 200 rows)
    Path("data/sample").mkdir(parents=True, exist_ok=True)
    pos.head(200).to_csv("data/sample/positions_sample.csv", index=False)

    print(f"Saved: {out_path} ({len(pos):,} rows)")
    print("Saved: data/sample/positions_sample.csv")


if __name__ == "__main__":
    main()
