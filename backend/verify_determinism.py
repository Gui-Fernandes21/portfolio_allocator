"""
Tests if identical inputs always produce identical weights.

For every frozen policy:
  1. builds ONE observation from the cached offline dataset (no network)
  2. predicts twice with the loaded policy -> asserts identical actions
  3. reloads the policy fresh from disk and predicts again -> asserts identical actions
  4. applies the softmax and prints the resulting weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "offline"))
from data_cache import load_cached_dataset 

from app.core.config import REWARD_VARIANTS, model_weights_path 
from app.services.observation_builder import observation_from_processed 
from app.services.allocation_service import _softmax 


def main() -> None:
    from stable_baselines3 import PPO

    observation = observation_from_processed(load_cached_dataset())
    print(f"observation as_of {observation.as_of}, shape {observation.state.shape}\n")

    failures = 0
    for variant in REWARD_VARIANTS:
        weights_path = model_weights_path(variant)
        
        if not weights_path.exists():
            print(f"{variant:8s}  SKIP (no weights at {weights_path.name})")
            continue

        policy = PPO.load(weights_path, device="cpu")
        action_first, _ = policy.predict(observation.state, deterministic=True)
        action_second, _ = policy.predict(observation.state, deterministic=True)

        policy_reloaded = PPO.load(weights_path, device="cpu")
        action_reloaded, _ = policy_reloaded.predict(observation.state, deterministic=True)

        same_call = np.array_equal(action_first, action_second)
        same_reload = np.array_equal(action_first, action_reloaded)
        status = "OK  " if (same_call and same_reload) else "FAIL"
        if status == "FAIL":
            failures += 1

        portfolio_weights = _softmax(np.asarray(action_first, dtype=float).flatten())
        weight_string = ", ".join(
            f"{ticker}={weight:.4f}" for ticker, weight in zip(observation.tickers, portfolio_weights)
        )
        print(f"{variant:8s}  {status}  repeat={same_call} reload={same_reload}")
        print(f"          weights: {weight_string}")

    if failures:
        raise SystemExit(f"\n{failures} variant(s) FAILED the determinism check")
    print("\nall variants deterministic: repeated and reloaded predictions are identical")


if __name__ == "__main__":
    main()
