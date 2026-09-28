from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))
from rewards import (  # noqa: E402
    DifferentialSharpeReward,
    DownsideRiskReward,
    RawReturnReward,
    TransactionCostReward,
    make_reward,
)

class TestRawReturnReward(unittest.TestCase):
    def test_reward_is_exactly_the_return(self):
        reward = RawReturnReward()
        reward.reset()
        self.assertEqual(reward.step(0.0123), 0.0123)
        self.assertEqual(reward.step(-0.02, turnover=0.5), -0.02)  # ignores turnover


class TestTransactionCostReward(unittest.TestCase):
    def test_reward_is_return_minus_cost(self):
        reward = TransactionCostReward(cost_rate=0.001)
        reward.reset()
        self.assertAlmostEqual(reward.step(0.01, turnover=0.2),0.01 - 0.001 * 0.2, places=12)

    def test_zero_turnover_means_no_penalty(self):
        reward = TransactionCostReward(cost_rate=0.001)
        reward.reset()
        self.assertEqual(reward.step(0.01, turnover=0.0), 0.01)


class TestDifferentialSharpeReward(unittest.TestCase):

    def test_warmup_steps_emit_zero(self):
        reward = DifferentialSharpeReward(warmup=10)
        reward.reset()
        emitted = [reward.step(r) for r in [0.01, -0.01] * 5]  # 10 steps
        self.assertEqual(emitted, [0.0] * 10)

    def test_reward_is_clipped_after_extreme_return(self):
        reward = DifferentialSharpeReward(warmup=5, clip=10.0)
        reward.reset()
        for step_return in [0.001, -0.001, 0.001, -0.001, 0.001, -0.001]:
            reward.step(step_return)
        spike = reward.step(10.0)  # absurd 1000% day
        self.assertEqual(abs(spike), 10.0)

    def test_reset_clears_moment_state(self):
        reward = DifferentialSharpeReward(warmup=2)
        reward.reset()
        for step_return in [0.01, -0.02, 0.03, 0.01]:
            reward.step(step_return)
        reward.reset()
        self.assertEqual(reward.step(0.05), 0.0)  # first step after reset
        self.assertEqual(reward.n, 1)


class TestDownsideRiskReward(unittest.TestCase):
    def test_warmup_steps_emit_zero(self):
        reward = DownsideRiskReward(warmup=10)
        reward.reset()
        emitted = [reward.step(r) for r in [0.01, -0.01] * 5]
        self.assertEqual(emitted, [0.0] * 10)

    def test_sign_follows_the_return_after_warmup(self):
        reward = DownsideRiskReward(warmup=3)
        reward.reset()
        for step_return in [0.01, -0.01, 0.01, -0.01]:
            reward.step(step_return)
        self.assertGreater(reward.step(0.02), 0.0)
        self.assertLess(reward.step(-0.02), 0.0)

    def test_reward_is_clipped(self):
        reward = DownsideRiskReward(warmup=2, clip=10.0)
        reward.reset()
        for step_return in [0.001, 0.001, 0.001]:  # no downside
            reward.step(step_return)
        spike = reward.step(5.0)
        self.assertLessEqual(abs(spike), 10.0)


class TestRegistry(unittest.TestCase):
    def test_all_four_variants_resolve(self):
        expected_classes = {
            "raw": RawReturnReward,
            "dsr": DifferentialSharpeReward,
            "sortino": DownsideRiskReward,
            "cost": TransactionCostReward,
        }
        for name, expected_class in expected_classes.items():
            self.assertIsInstance(make_reward(name), expected_class)

    def test_unknown_variant_raises(self):
        with self.assertRaises(KeyError):
            make_reward("does_not_exist")


if __name__ == "__main__":
    unittest.main()
