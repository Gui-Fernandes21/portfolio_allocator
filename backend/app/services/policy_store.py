"""
Loads and caches the frozen PPO policies produced by the offline pipeline.
"""

from __future__ import annotations

from app.core.config import REWARD_VARIANTS, model_weights_path

# Cache so each policy is deserialized once per process.
_loaded_policies: dict[str, object] = {}

def available_variants():
    """Reward variants whose frozen weights exist on disk."""
    return [variant for variant in REWARD_VARIANTS if model_weights_path(variant).exists()]


def load_policy(variant):
    """
    Return the frozen PPO policy for 'variant', or 'None' if not available.
    """
    if variant in _loaded_policies:
        return _loaded_policies[variant]

    weights_path = model_weights_path(variant)
    if not weights_path.exists():
        return None

    # Imported lazily so the API can boot without RL stack installed.
    from stable_baselines3 import PPO

    policy = PPO.load(weights_path, device="cpu")
    _loaded_policies[variant] = policy
    return policy
