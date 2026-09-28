"""
Turns the ablation's per-run outputs into the study's findings tables.

Reads (from results/):
    run_log.json                        one entry per completed run
    runs/returns_<tag>.csv              holdout daily returns (date, daily_return)
    runs/actions_<tag>.csv              daily portfolio weights (index=date, cols=tickers)

Writes (to results/):
    metrics_per_run.csv                 every metric for every run + baselines
    summary_by_variant.csv              mean ± std per variant, across seeds x folds
    summary_by_variant_fold.csv         per-fold means (seed-averaged) per variant
    ranking_by_fold.csv                 ranking per fold incl. baselines (1/N, SPY)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# fixed settings
COST_RATE = 0.001         # same as TransactionCostReward
TRADING_DAYS_PER_YEAR = 252
RANKING_METRIC = "sharpe_net"   # the headline metric used for per-fold rankings

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"
RUN_LOG_PATH = RESULTS_DIR / "run_log.json"
RAW_PRICES_PATH = Path(__file__).resolve().parent / "data_cache" / "raw_prices.csv"

# Model-free baselines evaluated on the SAME test windows and metric code:
# ew_1overN — daily-rebalanced equal weight (DeMiguel et al.'s 1/N)
# spy_bh    — SPY buy-and-hold
BASELINE_VARIANTS = ["ew_1overN", "spy_bh"]


# metrics
def annualized_return(daily_returns):
    cumulative = float(np.prod(1.0 + daily_returns))
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    
    return cumulative ** (1.0 / years) - 1.0 if years > 0 else np.nan # avoid divide-by-zero


def sharpe_ratio(daily_returns):
    if daily_returns.std(ddof=1) == 0:
        return np.nan
    
    return daily_returns.mean() / daily_returns.std(ddof=1)* np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(daily_returns):
    downside = np.minimum(daily_returns, 0.0)
    downside_deviation = np.sqrt(np.mean(downside ** 2))
    
    if downside_deviation == 0:
        return np.nan
    
    return daily_returns.mean() / downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(daily_returns):
    wealth = np.cumprod(1.0 + daily_returns)
    running_peak = np.maximum.accumulate(wealth)
    return ((wealth - running_peak) / running_peak).min()


def daily_turnover(weights):
    number_of_assets = weights.shape[1]
    equal_weights = np.full(number_of_assets, 1.0 / number_of_assets) 
    previous = np.vstack([equal_weights, weights.values[:-1]]) 
    
    return np.abs(weights.values - previous).sum(axis=1)


# per run metrics
def compute_run_metrics(entry: dict) -> dict | None:
    """All metrics for one run. Returns None if returns file is missing."""
    
    returns_path = RUNS_DIR / entry["returns_csv"]
    if not returns_path.exists():
        print(f"WARNING: missing {returns_path.name}; skipping {entry['run_tag']}")
        return None
    
    gross_returns = pd.read_csv(returns_path)["daily_return"].to_numpy()

    actions_name = entry.get("actions_csv", f"actions_{entry['run_tag']}.csv")
    actions_path = RUNS_DIR / actions_name
    
    if actions_path.exists():
        weights = pd.read_csv(actions_path, index_col=0)
        turnover = daily_turnover(weights)
        
        # Align the two series in case of any off-by-one or missing days, then compute net returns.
        common_length = min(len(turnover), len(gross_returns))
        turnover = turnover[:common_length]
        gross_aligned = gross_returns[:common_length]
        net_returns = gross_aligned - COST_RATE * turnover
        mean_turnover = float(turnover.mean())
    else:
        print(f"WARNING: no actions file for {entry['run_tag']}. net-of-cost metrics is not available.")
        net_returns, mean_turnover = None, np.nan

    metrics = {
        "run_tag": entry["run_tag"],
        "variant": entry["variant"],
        "seed": entry["seed"],
        "fold": entry["fold"],
        "test_days": len(gross_returns),
        "mean_daily_turnover": mean_turnover,
        
        # GROSS 
        "annual_return_gross": annualized_return(gross_returns),
        "sharpe_gross": sharpe_ratio(gross_returns),
        "sortino_gross": sortino_ratio(gross_returns),
        "max_drawdown_gross": max_drawdown(gross_returns),
    }
    
    if net_returns is not None:
        metrics.update({
            "annual_return_net": annualized_return(net_returns),
            "sharpe_net": sharpe_ratio(net_returns),
            "sortino_net": sortino_ratio(net_returns),
            "max_drawdown_net": max_drawdown(net_returns),
        })
    else:
        metrics.update({
            "annual_return_net": np.nan, "sharpe_net": np.nan,
            "sortino_net": np.nan, "max_drawdown_net": np.nan,
        })
        
    return metrics


# baselines
def compute_baseline_metrics(fold, test_start: str, test_end: str):
    """
    Uses the SAME cached prices the training data came from and the SAME
    metric functions as the RL variants. 
    """
    prices = pd.read_csv(RAW_PRICES_PATH)
    price_pivot = prices.pivot_table(index="date", columns="tic", values="close").sort_index()
    window = price_pivot[(price_pivot.index >= test_start) & (price_pivot.index <= test_end)]
    asset_returns = window.pct_change().dropna()
    number_of_assets = asset_returns.shape[1]
    equal_weight = 1.0 / number_of_assets

    rows = []

    portfolio_returns = asset_returns.mean(axis=1).to_numpy()
    
    # turnover of daily rebalancing
    drifted = equal_weight * (1.0 + asset_returns.to_numpy()) / (1.0 + portfolio_returns)[:, None]
    rebalance_turnover = np.abs(equal_weight - drifted).sum(axis=1)
    net_returns = portfolio_returns - COST_RATE * rebalance_turnover
    
    rows.append({
        "run_tag": f"baseline_ew_fold{fold}", 
        "variant": "ew_1overN",
        "seed": -1, 
        "fold": fold, 
        "test_days": len(portfolio_returns),
        "mean_daily_turnover": float(rebalance_turnover.mean()),
        "annual_return_gross": annualized_return(portfolio_returns),
        "sharpe_gross": sharpe_ratio(portfolio_returns),
        "sortino_gross": sortino_ratio(portfolio_returns),
        "max_drawdown_gross": max_drawdown(portfolio_returns),
        "annual_return_net": annualized_return(net_returns),
        "sharpe_net": sharpe_ratio(net_returns),
        "sortino_net": sortino_ratio(net_returns),
        "max_drawdown_net": max_drawdown(net_returns),
    })

    # SPY buy-and-hold
    spy_returns = asset_returns["SPY"].to_numpy()
    rows.append({
        "run_tag": f"baseline_spy_fold{fold}", 
        "variant": "spy_bh",
        "seed": -1, 
        "fold": fold, 
        "test_days": len(spy_returns),
        "mean_daily_turnover": 0.0,
        "annual_return_gross": annualized_return(spy_returns),
        "sharpe_gross": sharpe_ratio(spy_returns),
        "sortino_gross": sortino_ratio(spy_returns),
        "max_drawdown_gross": max_drawdown(spy_returns),
        "annual_return_net": annualized_return(spy_returns),
        "sharpe_net": sharpe_ratio(spy_returns),
        "sortino_net": sortino_ratio(spy_returns),
        "max_drawdown_net": max_drawdown(spy_returns),
    })
    
    return rows


# full-period series
def full_period_drawdowns(per_run: pd.DataFrame, fold_windows: dict):
    rows = []

    rl_runs = per_run[per_run.seed >= 0]
    
    for variant in sorted(rl_runs.variant.unique()):
        fold_parts = []
        for fold in sorted(rl_runs.fold.unique()):
            seed_series = []
            for seed in sorted(rl_runs[rl_runs.variant == variant].seed.unique()):

                tag = f"{variant}_seed{seed}_fold{fold}"
                returns_path = RUNS_DIR / f"returns_{tag}.csv"
                actions_path = RUNS_DIR / f"actions_{tag}.csv"
                
                if not (returns_path.exists() and actions_path.exists()):
                    continue
                
                returns_frame = pd.read_csv(returns_path)
                weights = pd.read_csv(actions_path, index_col=0)
                turnover = daily_turnover(weights)
                
                # Align the two series in case of any off-by-one or missing days, then compute net returns.
                common_length = min(len(turnover), len(returns_frame))
                seed_series.append(pd.Series(
                    returns_frame["daily_return"].to_numpy()[:common_length]
                    - COST_RATE * turnover[:common_length],
                    index=returns_frame["date"][:common_length],
                ))
            if seed_series:
                fold_parts.append(pd.concat(seed_series, axis=1).mean(axis=1)) # average across seeds for this fold
        if fold_parts:
            stitched = pd.concat(fold_parts).sort_index().to_numpy()
            rows.append({"strategy": variant,
                         "full_period_max_drawdown_net": max_drawdown(stitched)})

    # Baselines over the union of the test windows, from the cached prices.
    full_start = min(start for start, _ in fold_windows.values())
    full_end = max(end for _, end in fold_windows.values())
    
    prices = pd.read_csv(RAW_PRICES_PATH)
    price_pivot = prices.pivot_table(index="date", columns="tic", values="close").sort_index()
    
    # Select the window of data to use for the baselines.
    window = price_pivot[(price_pivot.index >= full_start) & (price_pivot.index <= full_end)]
    
    # Compute the baselines on the stitched window.
    asset_returns = window.pct_change().dropna()
    number_of_assets = asset_returns.shape[1]
    equal_weight = 1.0 / number_of_assets
    
    # Compute the equal-weight portfolio returns.
    ew_gross = asset_returns.mean(axis=1).to_numpy()
    
    # Compute the drifted weights.
    drifted = equal_weight * (1.0 + asset_returns.to_numpy()) / (1.0 + ew_gross)[:, None]
    ew_net = ew_gross - COST_RATE * np.abs(equal_weight - drifted).sum(axis=1)
    
    rows.append({"strategy": "ew_1overN","full_period_max_drawdown_net": max_drawdown(ew_net)})
    rows.append({"strategy": "spy_bh","full_period_max_drawdown_net": max_drawdown(asset_returns["SPY"].to_numpy())})
    
    return pd.DataFrame(rows)


# aggregation logic
def load_run_log():
    if not RUN_LOG_PATH.exists():
        raise FileNotFoundError(f"no run log at {RUN_LOG_PATH}; run the ablation first")
    
    log = json.loads(RUN_LOG_PATH.read_text())
    
    deduplicated = {entry["run_tag"]: entry for entry in log}
    
    return list(deduplicated.values())


def main():
    entries = load_run_log()
    rows = [metrics for metrics in (compute_run_metrics(e) for e in entries) if metrics is not None]
    
    if not rows:
        raise SystemExit("no runs with outputs found")

    successful_tags = {row["run_tag"] for row in rows}
    fold_windows = {entry["fold"]: (entry["test_start"], entry["test_end"]) for entry in entries if entry["run_tag"] in successful_tags}
    
    for fold, (test_start, test_end) in sorted(fold_windows.items()):
        rows.extend(compute_baseline_metrics(fold, test_start, test_end))

    per_run = pd.DataFrame(rows).sort_values(["variant", "fold", "seed"])
    per_run.to_csv(RESULTS_DIR / "metrics_per_run.csv", index=False)

    metric_columns = [column for column in per_run.columns if column not in ("run_tag", "variant", "seed", "fold")]

    # Per-variant summary across all seeds x folds: the headline table.
    summary = per_run.groupby("variant")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.round(4)
    summary.to_csv(RESULTS_DIR / "summary_by_variant.csv")

    # Per-fold means averaged over seeds.
    # Generate the consistency table
    by_fold = (per_run.groupby(["variant", "fold"])[metric_columns].mean().round(4))
    by_fold.to_csv(RESULTS_DIR / "summary_by_variant_fold.csv")

    # Ranking per fold on the headline metric: does the ordering hold up?
    ranking = (by_fold[RANKING_METRIC].unstack("fold").rank(ascending=False).astype("Int64"))
    ranking.columns = [f"fold_{fold}" for fold in ranking.columns]
    ranking.to_csv(RESULTS_DIR / "ranking_by_fold.csv")

    # Concatenate full-period drawdowns.
    full_period = full_period_drawdowns(per_run, fold_windows).round(4)
    full_period.to_csv(RESULTS_DIR / "full_period_drawdowns.csv", index=False)
    
    print("\n=== concatenated full-period max drawdown ===")
    print(full_period.to_string())

    print(f"\nruns aggregated: {len(per_run)}")
    print(f"\n=== summary by variant (mean over seeds x folds) ===")
    display_columns = ["sharpe_gross_mean", 
                       "sharpe_net_mean", 
                       "sharpe_net_std",
                       "annual_return_net_mean", 
                       "max_drawdown_net_mean",
                       "mean_daily_turnover_mean"]
    
    available = [column for column in display_columns if column in summary.columns]
    
    print(summary[available].to_string())
    print(f"\n=== variant rank per fold ({RANKING_METRIC}, 1 = best) ===")
    print(ranking.to_string())
    print(f"\ntables written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
