from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import pandas as pd
import requests


def to_stooq_symbol(ticker: str) -> str:
    """
    Stooq commonly uses:
      - US stocks/ETFs: TICKER.US (e.g., AAPL.US)
      - Indices: ^SPX, ^DJI
    We'll auto-append .US if you pass "SPY" (no suffix).
    """
    t = ticker.strip()
    if not t:
        raise ValueError("Empty ticker")

    # If user already provided a suffix or an index symbol, keep as-is
    if "." in t or t.startswith("^"):
        return t.lower()

    # Default for your ETF universe
    return f"{t}.US".lower()


def download_stooq_csv(symbol: str, start: str, end: str) -> pd.DataFrame:
    # Stooq uses YYYYMMDD in d1/d2
    d1 = pd.to_datetime(start).strftime("%Y%m%d")
    d2 = pd.to_datetime(end).strftime("%Y%m%d")

    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    # If symbol is invalid, sometimes you get a tiny/empty response
    text = r.text.strip()
    if len(text) < 20:
        raise ValueError(f"Empty response for {symbol} from Stooq")

    df = pd.read_csv(StringIO(text))
    if df.empty or "Date" not in df.columns:
        raise ValueError(f"Unexpected CSV for {symbol}. Columns: {list(df.columns)}")

    df = df.rename(columns={"Date": "date", "Close": "close"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["date", "close"]].copy()
    df["ticker"] = symbol.upper()
    return df.sort_values("date")


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    symbol = to_stooq_symbol(ticker)

    # Try with .US (or as provided)
    try:
        return download_stooq_csv(symbol, start, end)
    except Exception:
        # Fallback: try raw ticker without .US (sometimes stooq uses different notation)
        if symbol.endswith(".us"):
            alt = symbol.replace(".us", "")
            return download_stooq_csv(alt, start, end)
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="data/sample/universe.csv")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-26")
    ap.add_argument("--out", default="data/processed/prices.parquet")
    args = ap.parse_args()

    uni = pd.read_csv(args.universe)
    tickers = uni["ticker"].dropna().unique().tolist()
    if not tickers:
        raise ValueError("No tickers found in universe.csv")

    frames = []
    for t in tickers:
        df = fetch_one(t, args.start, args.end)
        frames.append(df)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out_path, index=False)

    # GitHub-safe sample: last 120 rows per ticker
    sample = prices.groupby("ticker").tail(120)
    sample.to_csv("data/sample/prices_sample.csv", index=False)

    print(f"Saved: {out_path} ({len(prices):,} rows)")
    print("Saved: data/sample/prices_sample.csv")


if __name__ == "__main__":
    main()
