"""
The implementations of the four reward variants in comparison. 
Swapping the reward variant is a one-line change and NOTHING ELSE in the pipeline differs between runs.

Data, network, and evaluation are fixed.

References of the reward implementations: Moody & Saffell (2001); Sortino & Price (1994).
"""

from __future__ import annotations
import numpy as np


class RewardFunction:
    """Interface. 
    reset() - called at the start of every episode. 
    step() - returns a scalar for one timestep given the realised one-step
    portfolio return and the turnover (L1 change in weights)."""
    
    name = "base"

    def reset(self) -> None:
        raise NotImplementedError

    def step(self, port_return: float, turnover: float = 0.0) -> float:
        raise NotImplementedError


class RawReturnReward(RewardFunction):
    """Baseline: reward = one-step portfolio return."""
    name = "raw"

    def reset(self):
        pass

    def step(self, port_return, turnover=0.0):
        return float(port_return)


class DifferentialSharpeReward(RewardFunction):
    """Differential Sharpe Ratio (Moody & Saffell, 2001).

    Holds exponential moving averages of the first (A) and second (B) moments
    of returns and uses the analytic derivative of the Sharpe ratio w.r.t. the
    adaptation rate eta as the per-step reward:

        D_t = (B_{t-1} dA - 0.5 A_{t-1} dB) / (B_{t-1} - A_{t-1}^2)^{3/2}

    giving a risk-adjusted signal available online at every step.

    Training aids:
        1. 'warmup' - lets the moment estimates settle before any reward is calculated.
        2. 'clip' - guards against spikes. Inpired by PPO clipping.
    """

    name = "dsr"

    def __init__(self, 
                 eta: float = 0.01, 
                 eps: float = 1e-8,
                 warmup: int = 10, 
                 clip: float = 10.0):
        self.eta = eta
        self.eps = eps
        self.warmup = warmup
        self.clip = clip

    def reset(self):
        self.A = 0.0 # first moment return
        self.B = 0.0 # second moment return 
        self.n = 0  # time step counter
        self._init = False

    def step(self, port_return, turnover=0.0):
        gain = float(port_return)
        self.n += 1
        if not self._init:
            self.A, self.B, self._init = gain, gain * gain, True
            return 0.0
        dA = gain - self.A
        dB = gain * gain - self.B
        variance = max(self.B - self.A * self.A, 0.0)
        denominator = (variance + self.eps) ** 1.5
        dsr_value = (self.B * dA - 0.5 * self.A * dB) / denominator 
        self.A += self.eta * dA
        self.B += self.eta * dB
        if self.n <= self.warmup:
            return 0.0
        return float(np.clip(dsr_value, -self.clip, self.clip))


class DownsideRiskReward(RewardFunction):
    """Sortino-style downside-risk reward.

    Equal-sized losses are penalised more heavily than equal-sized gains are rewarded.

    DESIGN CHOICE: this simple scaled-return form was chosen over a Moody & Saffell
    style "differential Sortino" derivative for simplicity and because it
    mirrors the DSR variant's structure (EMA state, warmup, clipping), so
    the two risk-adjusted variants differ only in WHICH risk they penalise
    (total variance vs downside deviation).
    """

    name = "sortino"

    def __init__(self, 
                 eta: float = 0.01, 
                 target: float = 0.0, 
                 eps: float = 1e-6,
                 warmup: int = 10, 
                 clip: float = 10.0):
        self.eta = eta
        self.target = target
        self.eps = eps
        self.warmup = warmup
        self.clip = clip

    def reset(self):
        self.dd = 0.0
        self.n = 0
        self._init = False

    def step(self, port_return, turnover=0.0):
        gain = float(port_return)
        self.n += 1
        downside = min(gain - self.target, 0.0) ** 2

        if not self._init:
            self.dd, self._init = downside, True
            return 0.0

        scaled = gain / np.sqrt(self.dd + self.eps)
        self.dd += self.eta * (downside - self.dd)

        if self.n <= self.warmup:
            return 0.0

        return float(np.clip(scaled, -self.clip, self.clip))


class TransactionCostReward(RewardFunction):
    """Cost-aware reward = gross one-step return - cost_rate * turnover.

    Turnover is the L1 change in portfolio weights between steps. 
    
    'cost_rate' - the round-trip cost per unit of weight traded (0.001 = 10 bps)."""
    name = "cost"

    def __init__(self, cost_rate: float = 0.001):
        self.cost_rate = cost_rate

    def reset(self):
        pass

    def step(self, port_return, turnover=0.0):
        return float(port_return - self.cost_rate * turnover)


# Registry so the app can select a variant by name.
REWARD_REGISTRY = {
    "raw": RawReturnReward,
    "dsr": DifferentialSharpeReward,
    "sortino": DownsideRiskReward,
    "cost": TransactionCostReward,
}

# Constructs a reward object from the registry by name.
def make_reward(name: str, **kwargs) -> RewardFunction:
    if name not in REWARD_REGISTRY:
        raise KeyError(f"unknown reward '{name}'. options: {list(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name](**kwargs)
