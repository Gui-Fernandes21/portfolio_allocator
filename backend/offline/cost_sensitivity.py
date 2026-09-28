"""
This module performs a check to see how the study's conclusions depend on the assumed
transaction-cost rate.

This module needs no retraining. Net-of-cost series at any rate is only a recomputation:

'net_t = gross_t - cost_rate * turnover_t'

Usage example:
    python cost_sensitivity.py
    python cost_sensitivity.py --rates 0 5 10 25 50 100

Outputs to results/:
- cost_sensitivity.csv          net Sharpe per strategy per rate
- cost_sensitivity_margins.csv  each variant's margin over 1/N per rate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_results import (
    RAW_PRICES_PATH,
    RESULTS_DIR,
    RUNS_DIR,
    annualized_return,
    daily_turnover,
    load_run_log,
    max_drawdown,
    sharpe_ratio,
)

DEFAULT_RATES_BPS = [0, 5, 10, 25, 50, 100]

RL_VARIANTS = ["raw", "dsr", "sortino", "cost"]
BASELINE_1N = "ew_1overN"


#  loading series -------------------
def load_run_series(entry):
    """
    Return (gross daily returns, daily turnover) for one run.
    Returns None if the required CSV files are missing.
    """
    
    returns_path = RUNS_DIR / entry["returns_csv"]
    actions_name = entry.get("actions_csv", f"actions_{entry['run_tag']}.csv")
    actions_path = RUNS_DIR / actions_name

    if not returns_path.exists() or not actions_path.exists():
        return None
    
    gross = pd.read_csv(returns_path)["daily_return"].to_numpy()
    weights = pd.read_csv(actions_path, index_col=0)
    turnover = daily_turnover(weights)
    
    # Align the two series in case of any off-by-one or missing data issues.
    common_length = min(len(gross), len(turnover))
    
    return gross[:common_length], turnover[:common_length]


def load_baseline_series(test_start, test_end):
    """Return gross and turnover for 1/N and for SPY buy-and-hold."""
    
    prices = pd.read_csv(RAW_PRICES_PATH)
    pivot = prices.pivot_table(index="date", columns="tic", values="close").sort_index()
    window = pivot[(pivot.index >= test_start) & (pivot.index <= test_end)]
    
    asset_returns = window.pct_change().dropna()
    number_of_assets = asset_returns.shape[1]
    equal_weight = 1.0 / number_of_assets

    portfolio_returns = asset_returns.mean(axis=1).to_numpy()
    
    
    # Turnover of restoring equal weight after one day of return drift.
    drifted = (equal_weight * (1.0 + asset_returns.to_numpy()) / (1.0 + portfolio_returns)[:, None]    )
    rebalance_turnover = np.abs(equal_weight - drifted).sum(axis=1)

    spy_returns = asset_returns["SPY"].to_numpy()
    return (
        (portfolio_returns, rebalance_turnover),
        (spy_returns, np.zeros_like(spy_returns)),  # buy-and-hold strategy has zero turnover
    )


# sweeping ------------------------
def sweep(rates_bps):
    """Produce Net Sharpe for every strategy at every cost rate."""

    entries = load_run_log()
    rows = []

    # --- one row per [rate, variant, seed, fold] ---
    for entry in entries:
        series = load_run_series(entry)
        if series is None:
            continue

        gross, turnover = series

        for rate_bps in rates_bps:
            cost_rate = rate_bps / 10_000.0
            net = gross - cost_rate * turnover

            rows.append({
                "rate_bps": rate_bps,
                "strategy": entry["variant"],
                "seed": entry["seed"],
                "fold": entry["fold"],
                "sharpe_net": sharpe_ratio(net),
                "annual_return_net": annualized_return(net),
                "max_drawdown_net": max_drawdown(net),
                "mean_daily_turnover": turnover.mean(),
            })

    # --- baselines ----------------------
    fold_windows = {
        entry["fold"]: (entry["test_start"], entry["test_end"])
        for entry in entries if (RUNS_DIR / entry["returns_csv"]).exists()
    }

    for fold, (test_start, test_end) in sorted(fold_windows.items()):
        ew_baseline, spy_baseline = load_baseline_series(test_start, test_end)

        baseline_tuple = (BASELINE_1N, ew_baseline)
        spy_tuple = ("spy_bh", spy_baseline) 

        for strategy, (gross, turnover) in [baseline_tuple, spy_tuple]:
            for rate_bps in rates_bps:
                cost_rate = rate_bps / 10_000.0
                net = gross - cost_rate * turnover
                rows.append({
                    "rate_bps": rate_bps,
                    "strategy": strategy,
                    "seed": -1, # baseline has no seed
                    "fold": fold,
                    "sharpe_net": sharpe_ratio(net),
                    "annual_return_net": annualized_return(net),
                    "max_drawdown_net": max_drawdown(net),
                    "mean_daily_turnover": turnover.mean(),
                })

    return pd.DataFrame(rows)


def summarise(per_run):
    """Seed-and-fold-averaged net Sharpe per strategy per rate"""
    
    # Average over seeds first, then folds, so each fold weighs equally regardless of how many seeds a strategy has.
    by_fold = per_run.groupby(["rate_bps", "strategy", "fold"])["sharpe_net"].mean().reset_index()
    
    return by_fold.groupby(["rate_bps", "strategy"])["sharpe_net"].mean().unstack("strategy")


def margins_over_baseline(per_run):
    by_fold = per_run.groupby(["rate_bps", "strategy", "fold"])["sharpe_net"].mean().unstack("strategy")
    records = []
    
    for rate_bps in sorted(by_fold.index.get_level_values("rate_bps").unique()):
        block = by_fold.xs(rate_bps, level="rate_bps")
        
        for variant in RL_VARIANTS:
            difference = block[variant] - block[BASELINE_1N]
            records.append({
                "rate_bps": rate_bps,
                "variant": variant,
                "mean_margin_over_1N": difference.mean(),
                "folds_ahead_of_1N": int((difference > 0).sum()),
                "folds_total": len(difference),
            })
    
    return pd.DataFrame(records)


def paired_wins_over_raw(per_run):
    """
    For each rate, how many of the 20 matched (seed, fold) pairs
    does each risk-shaped variant win against the raw-return variant?
    """
    VARIANTS = ["dsr", "sortino", "cost"]

    rl_only = per_run[per_run.seed >= 0]
    records = []

    for rate_bps in sorted(rl_only.rate_bps.unique()):
        block = rl_only[rl_only.rate_bps == rate_bps].pivot_table(index=["seed", "fold"], columns="strategy", values="sharpe_net")
        for variant in VARIANTS:
            difference = block[variant] - block["raw"]
            records.append({
                "rate_bps": rate_bps,
                "variant": variant,
                "mean_margin_over_raw": difference.mean(),
                "pairs_won_over_raw": int((difference > 0).sum()),
                "pairs_total": len(difference),
            })
            
    return pd.DataFrame(records)


def main():
    # Parse command-line arguments and run the sweep, summarise, and margin calculations.
    parser = argparse.ArgumentParser(description="Sweep the evaluation transaction-cost rate.")
    parser.add_argument("--rates", nargs="+", type=int, default=DEFAULT_RATES_BPS,help="cost rates in basis points")
    args = parser.parse_args()


    per_run = sweep(args.rates)
    if per_run.empty:
        raise SystemExit("no runs found; run the ablation first")

    summary = summarise(per_run)
    ordered_columns = [column for column in ["raw", "dsr", "sortino", "cost", BASELINE_1N, "spy_bh"] if column in summary.columns]
    
    summary = summary[ordered_columns]
    summary.round(4).to_csv(RESULTS_DIR / "cost_sensitivity.csv", index=False)

    margins = margins_over_baseline(per_run)
    margins.round(4).to_csv(RESULTS_DIR / "cost_sensitivity_margins.csv",index=False)

    print("\n=== mean net Sharpe by cost rate (fold-averaged) ===")
    print(summary.round(3).to_string())
    print("\n=== margin over 1/N, and folds ahead (of 4) ===")
    
    pivot_margin = margins.pivot(index="rate_bps", columns="variant",values="mean_margin_over_1N")[RL_VARIANTS]
    pivot_folds = margins.pivot(index="rate_bps", columns="variant",values="folds_ahead_of_1N")[RL_VARIANTS]
    combined = pivot_margin.round(4).astype(str) + "  (" + pivot_folds.astype(str) + "/4)"
    
    print(combined.to_string())

    paired = paired_wins_over_raw(per_run)
    paired.round(4).to_csv(RESULTS_DIR / "cost_sensitivity_paired.csv",index=False)
    
    print("\n=== H1 statistic: matched pairs won against the raw variant (of 20) ===")
    paired_pivot = paired.pivot(index="rate_bps", columns="variant",values="pairs_won_over_raw")
    margin_pivot = paired.pivot(index="rate_bps", columns="variant",values="mean_margin_over_raw")
    
    order = ["dsr", "sortino", "cost"]
    
    print((paired_pivot[order].astype(str) + "/20  (" + margin_pivot[order].round(4).astype(str) + ")").to_string())
    print(f"\ntables written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
