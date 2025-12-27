from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/processed/prices.parquet")
    ap.add_argument("--positions", default="data/processed/positions.csv")
    ap.add_argument("--out_portfolio", default="data/processed/portfolio_daily.csv")
    ap.add_argument("--out_returns", default="data/processed/returns_asset.csv")
    ap.add_argument("--start_value", type=float, default=100.0)
    args = ap.parse_args()

    # ---- load prices ----
    prices = pd.read_parquet(args.prices).copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # compute asset returns
    prices["ret"] = prices.groupby("ticker")["close"].pct_change()
    rets = prices.dropna(subset=["ret"])[["date", "ticker", "ret"]].copy()

    # ---- load positions (monthly rebalances) ----
    pos = pd.read_csv(args.positions).copy()
    pos["rebalance_date"] = pd.to_datetime(pos["rebalance_date"])
    pos = pos.sort_values(["ticker", "rebalance_date"]).reset_index(drop=True)

    # ---- expand monthly weights to daily weights per ticker ----
    # Create a full daily calendar for each ticker using the price dates we actually have
    all_dates = (
        prices[["date", "ticker"]]
        .drop_duplicates()
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    # Merge monthly weights onto the daily calendar, then forward-fill within each ticker
    daily_w = all_dates.merge(
        pos.rename(columns={"rebalance_date": "date"})[["date", "ticker", "weight"]],
        on=["date", "ticker"],
        how="left",
    )

    daily_w["weight"] = daily_w.groupby("ticker")["weight"].ffill()

    # If there are still missing weights (shouldn't happen if positions start early enough)
    if daily_w["weight"].isna().any():
        miss = daily_w[daily_w["weight"].isna()][["ticker", "date"]].head(20)
        raise ValueError(
            "Some daily weights are missing after forward-fill. "
            "This means your first rebalance_date is after the first price date.\n"
            f"Example missing rows:\n{miss}"
        )

    # ---- join returns with daily weights ----
    merged = rets.merge(daily_w, on=["date", "ticker"], how="inner")

    merged["wret"] = merged["weight"] * merged["ret"]

    port = (
        merged.groupby("date", as_index=False)["wret"]
        .sum()
        .rename(columns={"wret": "port_ret"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Portfolio value index
    port["port_value"] = args.start_value * (1.0 + port["port_ret"]).cumprod()

    # ---- save outputs ----
    outp = Path(args.out_portfolio)
    outp.parent.mkdir(parents=True, exist_ok=True)
    port.to_csv(outp, index=False)

    outr = Path(args.out_returns)
    outr.parent.mkdir(parents=True, exist_ok=True)
    merged[["date", "ticker", "ret", "weight"]].to_csv(outr, index=False)

    # GitHub-safe sample
    Path("data/sample").mkdir(parents=True, exist_ok=True)
    port.tail(200).to_csv("data/sample/portfolio_daily_sample.csv", index=False)

    print(f"Saved: {outp} ({len(port):,} days)")
    print(f"Saved: {outr} ({len(merged):,} rows)")
    print("Saved: data/sample/portfolio_daily_sample.csv")


if __name__ == "__main__":
    main()
