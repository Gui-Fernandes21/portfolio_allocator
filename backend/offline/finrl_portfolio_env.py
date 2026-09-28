"""
- StockPortfolioEnv subclass
- Swaps FinRL's built-in reward for one ablation reward strategies.

- Offline pipeline environment
- Only isolated to the reward computation

- Every line that differs from FinRL is marked '# <-- CHANGED'
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv

from rewards import RewardFunction  # reward-strategy classes

class FinRLPortfolioEnv(StockPortfolioEnv):
    """StockPortfolioEnv with a pluggable, risk-modulated reward."""

    def __init__(self, df, reward_function: RewardFunction, **kwargs):
        self.reward_function = reward_function  # <-- CHANGED
        super().__init__(df=df, **kwargs)
        self.reward_function.reset()  # <-- CHANGED

    def reset(self, *, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        self.reward_function.reset()  # <-- CHANGED
        return state, info

    # CHANGED: The reward is computed by the reward variant.
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            os.makedirs("results", exist_ok=True)  # <-- CHANGED 
            df_cumulative_return = pd.DataFrame(self.portfolio_return_memory)
            df_cumulative_return.columns = ["daily_return"]
            plt.plot(df_cumulative_return.daily_return.cumsum(), "r")
            plt.savefig("results/cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("results/rewards.png")
            plt.close()

            print("=================================")
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")

            # compute Sharpe ratio for the whole period
            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                mean = df_daily_return["daily_return"].mean()
                std = df_daily_return["daily_return"].std()
                sharpe_ratio = (252**0.5) * mean / std
                print("Sharpe: ", sharpe_ratio)
            print("=================================")

            return self.state, self.reward, self.terminal, False, {}

        # reward swapped
        portfolio_weights = self.softmax_normalization(actions) 
        self.actions_memory.append(portfolio_weights)
        last_day_memory = self.data

        # advance one trading day and rebuild the covariance-matrix observation
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        list_of_tech_indicators = [self.data[tech].values.tolist() for tech in self.tech_indicator_list]
        self.state = np.append(np.array(self.covs), list_of_tech_indicators, axis=0)

        # one-step portfolio returns = per-asset return weighted by the chosen weights
        portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * portfolio_weights)
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
        self.portfolio_value = new_portfolio_value

        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(self.data.date.unique()[0])
        self.asset_memory.append(new_portfolio_value)

        # ----------------------------------------------------------------------
        # CHANGED: FinRL's logic is replaced with the selected reward strategy.
        previous_weights = np.array(self.actions_memory[-2])
        portfolio_turnover = float(np.sum(np.abs(np.array(portfolio_weights) - previous_weights)))

        self.reward = self.reward_function.step(float(portfolio_return), portfolio_turnover)  # <-- CHANGED
        # ----------------------------------------------------------------------

        return self.state, self.reward, self.terminal, False, {}
