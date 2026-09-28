from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import pandas as pd

# Blocking a live trading modules that are not used for the offline pipeline
for shortcut in ("finrl.test", "finrl.trade", "finrl.train"):
    blocked_module = types.ModuleType(shortcut)
    setattr(blocked_module, shortcut.rsplit(".", 1)[-1], None)
    sys.modules[shortcut] = blocked_module

from finrl.meta.preprocessor.preprocessors import data_split
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS

from data_cache import load_cached_dataset
from walk_forward import make_walk_forward_folds
from finrl_portfolio_env import FinRLPortfolioEnv
from rewards import make_reward

# FIXED SETTINGS
REWARD_VARIANTS = ["raw", "dsr", "sortino", "cost"]
SEED_LIST = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 50_000
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0001,
    "batch_size": 128,
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"
WEIGHTS_DIR = RESULTS_DIR / "weights"
RUN_LOG_PATH = RESULTS_DIR / "run_log.json"

# Where the live API expects its frozen policies.
FINAL_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
FINAL_MANIFEST_PATH = FINAL_MODELS_DIR / "manifest.json"
FINAL_SEED = 0  # the shipped policy is a single fixed-seed retrain

def run_single(dataset, 
               fold, 
               variant: str, 
               seed: int, 
               total_timesteps: int,
               save_weights: bool) -> dict:
    """Train one policy and evaluate it on its fold's test window."""
    
    train_dataframe = data_split(dataset, fold.train_start, fold.train_end)
    test_dataframe = data_split(dataset, fold.test_start, fold.test_end)

    stock_dimension = len(train_dataframe.tic.unique()) 
    
    finrl_env_kwargs = dict(
        hmax=100,
        initial_amount=1_000_000,
        transaction_cost_pct=0.001,
        state_space=stock_dimension,
        stock_dim=stock_dimension,
        tech_indicator_list=INDICATORS,
        action_space=stock_dimension,
        reward_scaling=1e-4,
    )

    run_tag = f"{variant}_seed{seed}_fold{fold.fold_number}"
    print(f"\n===== {run_tag} | {fold.describe()} =====")
    start_time = time.time()

    # train policy
    train_environment = FinRLPortfolioEnv(df=train_dataframe, reward_function=make_reward(variant), **finrl_env_kwargs)
    
    vectorised_train_environment, _ = train_environment.get_sb_env()
    
    agent = DRLAgent(env=vectorised_train_environment)
    
    ppo_model = agent.get_model("ppo", model_kwargs=PPO_PARAMS, seed=seed)
    
    trained_ppo_model = agent.train_model(
        model=ppo_model,
        tb_log_name=f"ppo_{run_tag}",
        total_timesteps=total_timesteps,
    )
    
    train_seconds = time.time() - start_time

    # freeze the output weights
    weights_path = None
    if save_weights:
        
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        weights_path = WEIGHTS_DIR / f"ppo_{run_tag}.zip"
        trained_ppo_model.save(weights_path)
        
        from stable_baselines3 import PPO
        
        PPO.load(weights_path, device="cpu")  # fails if the save fails
        print(f"weights saved and reloaded ok -> {weights_path.name}")

    # evaluate
    test_environment = FinRLPortfolioEnv(df=test_dataframe, reward_function=make_reward(variant), **finrl_env_kwargs)
    daily_return_dataframe, action_dataframe = DRLAgent.DRL_prediction(model=trained_ppo_model, environment=test_environment)
    
    # Save the test results
    RUNS_DIR.mkdir(parents=True, exist_ok=True) # make sure the directory exists
    returns_path = RUNS_DIR / f"returns_{run_tag}.csv"
    daily_return_dataframe.to_csv(returns_path, index=False)
    
    actions_path = RUNS_DIR / f"actions_{run_tag}.csv"
    action_dataframe.to_csv(actions_path) 
    
    print(f"holdout returns -> {returns_path.name} | weights -> {actions_path.name}")

    return {
        "run_tag": run_tag,
        "variant": variant,
        "seed": seed,
        "fold": fold.fold_number,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "total_timesteps": total_timesteps,
        "ppo_params": PPO_PARAMS,
        "train_seconds": round(train_seconds, 1),
        "returns_csv": str(returns_path.name),
        "actions_csv": str(actions_path.name),
        "weights_zip": str(weights_path.name) if weights_path else None,
    }

# Final retrain
def run_final(dataset, variant: str, total_timesteps: int) -> dict:
    """
    - Train the final policy for one variant and export it.
    - NOT a evaluation run.
    - Retrain on the full available dataset.
    - Uses the most recent rolling window and with the same length as in the study.
    """

    from walk_forward import ROLLING_TRAIN_YEARS

    trading_dates = sorted(dataset.date.unique())
    last_date = trading_dates[-1]
    rolling_cutoff = (pd.Timestamp(last_date) - pd.DateOffset(years=ROLLING_TRAIN_YEARS)).strftime("%Y-%m-%d")

    end_exclusive = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    train_dataframe = data_split(dataset, rolling_cutoff, end_exclusive)
    train_start = train_dataframe.date.min()
    train_end = train_dataframe.date.max()

    stock_dimension = len(train_dataframe.tic.unique())
    finrl_env_kwargs = dict(
        hmax=100,
        initial_amount=1_000_000,
        transaction_cost_pct=0.001,
        state_space=stock_dimension,
        stock_dim=stock_dimension,
        tech_indicator_list=INDICATORS,
        action_space=stock_dimension,
        reward_scaling=1e-4,
    )

    print(f"\n===== FINAL {variant} | train {train_start}..{train_end} "
          f"| seed {FINAL_SEED} | {total_timesteps} steps =====")
    start_time = time.time()

    train_environment = FinRLPortfolioEnv(df=train_dataframe, reward_function=make_reward(variant), **finrl_env_kwargs)
    vectorised_train_environment, _ = train_environment.get_sb_env()
    agent = DRLAgent(env=vectorised_train_environment)
    ppo_model = agent.get_model("ppo", model_kwargs=PPO_PARAMS, seed=FINAL_SEED)

    trained_ppo_model = agent.train_model(
        model=ppo_model,
        tb_log_name=f"ppo_final_{variant}",
        total_timesteps=total_timesteps,
    )

    # Save the weights.
    FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True) # make sure the directory exists
    weights_path = FINAL_MODELS_DIR / f"ppo_{variant}.zip" 
    trained_ppo_model.save(weights_path)

    from stable_baselines3 import PPO
    PPO.load(weights_path, device="cpu")  # fails if the save is bad
    print(f"exported + reload-verified -> {weights_path}")

    return {
        "variant": variant,
        "weights_file": weights_path.name,
        "train_start": str(train_start),
        "train_end": str(train_end),
        "seed": FINAL_SEED,
        "total_timesteps": total_timesteps,
        "ppo_params": PPO_PARAMS,
        "train_seconds": round(time.time() - start_time, 1),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

# record what the policies were trained on
def write_final_manifest(entries: list[dict]) -> None:
    manifest = {}
    if FINAL_MANIFEST_PATH.exists():
        manifest = json.loads(FINAL_MANIFEST_PATH.read_text())
    for entry in entries:
        manifest[entry["variant"]] = entry
    FINAL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


#  logging
def append_run_log(entry: dict) -> None:
    """Append one run's record to results/run_log.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True) # make sure the directory exists
    log = []
    
    if RUN_LOG_PATH.exists():
        log = json.loads(RUN_LOG_PATH.read_text())
    
    log.append(entry)
    RUN_LOG_PATH.write_text(json.dumps(log, indent=2))


#  main
def main():
    parser = argparse.ArgumentParser(description="PPO reward study runner.")
    parser.add_argument("--variants", nargs="+", default=REWARD_VARIANTS, choices=REWARD_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEED_LIST)
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="fold numbers to run")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="small run with weight save/load checks")
    parser.add_argument("--final", action="store_true", help="retrains one policy per variant on the most recent rolling window and export it.")
    args = parser.parse_args()

    if args.final:
        dataset = load_cached_dataset()
        entries = [run_final(dataset, variant, args.timesteps) for variant in args.variants]
        write_final_manifest(entries)
        return

    if args.smoke:
        args.variants = ['raw']
        args.seeds = [0]
        args.folds = [1]
        args.timesteps = 2048
        args.save_weights = True
        print("SMOKE MODE: raw / seed 0 / fold 1 /", args.timesteps, "timesteps")

    dataset = load_cached_dataset()
    
    all_folds = make_walk_forward_folds(sorted(dataset.date.unique()))
    
    selected_folds = [
        fold for fold in all_folds
        if args.folds is None or fold.fold_number in args.folds
    ]

    total_runs = len(args.variants) * len(args.seeds) * len(selected_folds)
    
    print(f"running {total_runs} runs: variants={args.variants} "
          f"seeds={args.seeds} folds={[f.fold_number for f in selected_folds]} "
          f"timesteps={args.timesteps}")

    for fold in selected_folds:
        for variant in args.variants:
            for seed in args.seeds:
                entry = run_single(
                    dataset, 
                    fold, 
                    variant, 
                    seed,
                    total_timesteps=args.timesteps,
                    save_weights=args.save_weights,
                )
                append_run_log(entry)

    print(f"\ndone. run log -> {RUN_LOG_PATH}")


if __name__ == "__main__":
    main()
