from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


TRADING_DAYS = 252


def rolling_beta(port_ret: pd.Series, bench_ret: pd.Series, window: int) -> pd.Series:
    """beta = cov(port, bench) / var(bench) over rolling window"""
    cov = port_ret.rolling(window).cov(bench_ret)
    var = bench_ret.rolling(window).var()
    return cov / var


def hist_var_cvar(returns: pd.Series, window: int, alpha: float) -> tuple[pd.Series, pd.Series]:
    """
    Historical VaR and CVaR (Expected Shortfall).
    returns: daily returns
    alpha: 0.95 or 0.99
    Output is positive "loss" numbers.
    """
    q = 1.0 - alpha  # tail probability

    # rolling quantile gives the (q)-quantile of returns (a negative number in bad times)
    qret = returns.rolling(window).quantile(q)

    # CVaR: mean of returns <= quantile within each rolling window
    # We'll compute with a rolling apply (fast enough for daily series).
    def cvar_func(x: np.ndarray) -> float:
        thresh = np.quantile(x, q)
        tail = x[x <= thresh]
        if len(tail) == 0:
            return np.nan
        return float(np.mean(tail))

    cvar_ret = returns.rolling(window).apply(cvar_func, raw=True)

    # Convert to positive loss convention
    var = -qret
    cvar = -cvar_ret
    return var, cvar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", default="data/processed/portfolio_daily.csv")
    ap.add_argument("--returns_asset", default="data/processed/returns_asset.csv")
    ap.add_argument("--benchmark", default="SPY.US")  # must exist in returns_asset tickers
    ap.add_argument("--out", default="data/processed/risk_daily.csv")
    ap.add_argument("--out_rc", default="data/processed/risk_contrib.csv")
    ap.add_argument("--window_var", type=int, default=252)
    ap.add_argument("--window_beta", type=int, default=252)
    args = ap.parse_args()

    # ---- Load portfolio series ----
    port = pd.read_csv(args.portfolio)
    port["date"] = pd.to_datetime(port["date"])
    port = port.sort_values("date").reset_index(drop=True)

    # ---- Drawdown ----
    port["peak"] = port["port_value"].cummax()
    port["drawdown"] = port["port_value"] / port["peak"] - 1.0
    port["max_drawdown_to_date"] = port["drawdown"].cummin()

    # ---- Rolling vol ----
    for w in (30, 60, 252):
        port[f"vol_{w}"] = port["port_ret"].rolling(w).std() * np.sqrt(TRADING_DAYS)

    # ---- Beta vs benchmark ----
    rets = pd.read_csv(args.returns_asset)
    rets["date"] = pd.to_datetime(rets["date"])
    rets = rets.sort_values(["ticker", "date"]).reset_index(drop=True)

    bench = (
        rets[rets["ticker"] == args.benchmark][["date", "ret"]]
        .rename(columns={"ret": "bench_ret"})
        .sort_values("date")
    )

    merged_beta = port[["date", "port_ret"]].merge(bench, on="date", how="inner").sort_values("date")
    merged_beta["beta"] = rolling_beta(merged_beta["port_ret"], merged_beta["bench_ret"], args.window_beta)

    port = port.merge(merged_beta[["date", "beta"]], on="date", how="left")

    # ---- Historical VaR / CVaR ----
    for alpha in (0.95, 0.99):
        var, cvar = hist_var_cvar(port["port_ret"], args.window_var, alpha)
        a = int(alpha * 100)
        port[f"var{a}_{args.window_var}"] = var
        port[f"cvar{a}_{args.window_var}"] = cvar

    # ---- Risk contribution (optional but impressive) ----
    # Compute monthly (or last available date each month) risk contribution shares using covariance matrix
    # We'll do month-end dates based on portfolio date series
    port["month"] = port["date"].dt.to_period("M")
    month_end_dates = port.groupby("month")["date"].max().sort_values().tolist()

    rc_rows = []
    for d in month_end_dates:
        end = pd.to_datetime(d)
        start = end - pd.Timedelta(days=365 * 2)  # lookback range (approx) before filtering exact window
        # take last window_var trading days up to date
        window_df = rets[(rets["date"] <= end) & (rets["date"] >= start)]

        # pivot to wide return matrix: rows=dates, cols=tickers
        mat = window_df.pivot_table(index="date", columns="ticker", values="ret").dropna(axis=0, how="any")
        if len(mat) < args.window_var:
            continue
        mat = mat.tail(args.window_var)

        # weights at date d (use last available weights on/before d)
        w_df = (
            rets[rets["date"] <= end][["date", "ticker", "weight"]]
            .sort_values(["ticker", "date"])
            .groupby("ticker")
            .tail(1)
        )
        w_df = w_df.set_index("ticker").reindex(mat.columns)
        if w_df["weight"].isna().any():
            continue

        w = w_df["weight"].to_numpy()
        cov = np.cov(mat.to_numpy(), rowvar=False)

        # portfolio stdev
        port_var = float(w.T @ cov @ w)
        if port_var <= 0:
            continue
        port_std = np.sqrt(port_var)

        # marginal risk contribution: (Σ w) / σ_p
        mcr = (cov @ w) / port_std

        # component contribution shares: w_i * mcr_i / σ_p
        rc_share = (w * mcr) / port_std

        for ticker, share in zip(mat.columns, rc_share):
            rc_rows.append({"date": end.date(), "ticker": ticker, "rc_share": float(share)})

    rc = pd.DataFrame(rc_rows)

    # ---- Save ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep only columns you’ll use in BI (clean table)
    keep_cols = [
        "date",
        "port_ret",
        "port_value",
        "drawdown",
        "max_drawdown_to_date",
        "vol_30",
        "vol_60",
        "vol_252",
        "beta",
        f"var95_{args.window_var}",
        f"cvar95_{args.window_var}",
        f"var99_{args.window_var}",
        f"cvar99_{args.window_var}",
    ]
    port[keep_cols].to_csv(out_path, index=False)

    out_rc = Path(args.out_rc)
    rc.to_csv(out_rc, index=False)

    # GitHub-safe samples
    Path("data/sample").mkdir(parents=True, exist_ok=True)
    port[keep_cols].tail(200).to_csv("data/sample/risk_daily_sample.csv", index=False)
    if not rc.empty:
        rc.tail(200).to_csv("data/sample/risk_contrib_sample.csv", index=False)

    print(f"Saved: {out_path} ({len(port):,} rows)")
    print(f"Saved: {out_rc} ({len(rc):,} rows)")
    print("Saved: data/sample/risk_daily_sample.csv")
    if not rc.empty:
        print("Saved: data/sample/risk_contrib_sample.csv")


if __name__ == "__main__":
    main()
