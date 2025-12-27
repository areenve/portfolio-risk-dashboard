from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def build_value_path(returns: pd.Series, start_value: float = 100.0) -> pd.Series:
    return start_value * (1.0 + returns).cumprod()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", default="data/processed/portfolio_daily.csv")
    ap.add_argument("--out", default="data/processed/stress_daily.csv")
    args = ap.parse_args()

    port = pd.read_csv(args.portfolio)
    port["date"] = pd.to_datetime(port["date"])
    port = port.sort_values("date").reset_index(drop=True)

    # Baseline path for reference
    baseline = port[["date", "port_ret"]].copy()
    baseline["scenario"] = "Baseline"
    baseline["scenario_value"] = build_value_path(baseline["port_ret"], start_value=100.0)

    # --- Scenario 1: COVID crash window (approx) ---
    # You can tweak these dates later
    covid = port[(port["date"] >= "2020-02-19") & (port["date"] <= "2020-03-23")][["date", "port_ret"]].copy()
    covid["scenario"] = "COVID Crash (2020-02-19 to 2020-03-23)"
    covid["scenario_value"] = build_value_path(covid["port_ret"], start_value=100.0)

    # --- Scenario 2: 2022 drawdown window (rates + equities repricing) ---
    y2022 = port[(port["date"] >= "2022-01-03") & (port["date"] <= "2022-10-12")][["date", "port_ret"]].copy()
    y2022["scenario"] = "2022 Drawdown (2022-01-03 to 2022-10-12)"
    y2022["scenario_value"] = build_value_path(y2022["port_ret"], start_value=100.0)

    # --- Scenario 3: Worst 30 trading days in your sample (path replay) ---
    port["roll30"] = (1.0 + port["port_ret"]).rolling(30).apply(lambda x: x.prod() - 1.0, raw=True)
    worst_end = port["roll30"].idxmin()
    worst_end_date = port.loc[worst_end, "date"]
    worst_start_date = port.loc[worst_end - 29, "date"] if worst_end >= 29 else port["date"].min()

    worst30 = port[(port["date"] >= worst_start_date) & (port["date"] <= worst_end_date)][["date", "port_ret"]].copy()
    worst30["scenario"] = f"Worst 30 Trading Days (ending {worst_end_date.date()})"
    worst30["scenario_value"] = build_value_path(worst30["port_ret"], start_value=100.0)

    out = pd.concat([baseline, covid, y2022, worst30], ignore_index=True)
    out = out[["scenario", "date", "scenario_value"]].sort_values(["scenario", "date"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # GitHub-safe sample
    Path("data/sample").mkdir(parents=True, exist_ok=True)
    out.groupby("scenario").tail(120).to_csv("data/sample/stress_daily_sample.csv", index=False)

    print(f"Saved: {out_path} ({len(out):,} rows)")
    print("Saved: data/sample/stress_daily_sample.csv")


if __name__ == "__main__":
    main()
