from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from agents.risk_manager import ( 
    MAX_SINGLE_ASSET_WEIGHT,
    MIN_ASSETS_HELD,
    RiskManagerAgent,
    apply_concentration_cap,
    check_minimum_diversification,
)
from orchestration.messages import AgentContext, AgentOutput 

TICKERS = ["EEM", "EFA", "GLD", "IWM", "QQQ", "SPY", "TLT"]


def make_weights(values: list[float]) -> list[dict]:
    return [{"ticker": t, "weight": w} for t, w in zip(TICKERS, values)]


def total(weights: list[dict]) -> float:
    return sum(w["weight"] for w in weights)


class TestConcentrationCap(unittest.TestCase):
    def test_compliant_allocation_is_untouched(self):
        weights = make_weights([1 / 7] * 7)
        adjusted, adjustments = apply_concentration_cap(weights)
        self.assertEqual(adjustments, [])
        for before, after in zip(weights, adjusted):
            self.assertAlmostEqual(before["weight"], after["weight"], places=6)

    def test_single_violator_is_capped(self):
        # SPY at 70%, the rest sharing 30%.
        weights = make_weights([0.05] * 6 + [0.70])
        weights = [{"ticker": t, "weight": w} for t, w in zip(TICKERS, [0.05, 0.05, 0.05, 0.05, 0.05, 0.70, 0.05])]
        adjusted, adjustments = apply_concentration_cap(weights)
        by_ticker = {e["ticker"]: e["weight"] for e in adjusted}

        self.assertLessEqual(by_ticker["SPY"], MAX_SINGLE_ASSET_WEIGHT + 1e-6)
        self.assertAlmostEqual(total(adjusted), 1.0, places=6)
        self.assertEqual(len(adjustments), 1)
        self.assertEqual(adjustments[0]["rule"], "concentration_cap")
        self.assertIn("SPY", adjustments[0]["detail"])

    # The case that SPY at 80% and QQQ at 15%
    # Redistributing SPY's excess would push QQQ far above the cap, so the algorithm must re-check.
    def test_redistribution_does_not_create_a_new_violator(self):
        weights = [{"ticker": t, "weight": w} for t, w in zip(TICKERS, [0.01, 0.01, 0.01, 0.01, 0.15, 0.80, 0.01])]
        adjusted, adjustments = apply_concentration_cap(weights)
        for entry in adjusted:
            self.assertLessEqual(
                entry["weight"], MAX_SINGLE_ASSET_WEIGHT + 1e-6,
                f"{entry['ticker']} exceeds the cap after redistribution",
            )
        self.assertAlmostEqual(total(adjusted), 1.0, places=5)

    def test_weights_still_sum_to_one_after_capping(self):
        for values in (
            [0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01],
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.45, 0.45, 0.10, 0.0, 0.0, 0.0, 0.0],
        ):
            adjusted, _ = apply_concentration_cap(make_weights(values))
            self.assertAlmostEqual(total(adjusted), 1.0, places=5)
            
    # One asset holds everything; the rest are exactly zero
    def test_zero_weight_receivers_absorb_excess_evenly(self):
        weights = make_weights([1.0] + [0.0] * 6)
        adjusted, _ = apply_concentration_cap(weights)
        by_ticker = {e["ticker"]: e["weight"] for e in adjusted}
        self.assertLessEqual(by_ticker["EEM"], MAX_SINGLE_ASSET_WEIGHT + 1e-6)
        self.assertAlmostEqual(total(adjusted), 1.0, places=5)
        receivers = [v for t, v in by_ticker.items() if t != "EEM"]
        self.assertTrue(all(v > 0 for v in receivers))

    def test_infeasible_cap_is_reported_not_applied(self):
        weights = make_weights([1 / 7] * 7)
        adjusted, adjustments = apply_concentration_cap(weights, cap=0.05)
        self.assertEqual(len(adjustments), 1)
        self.assertIn("infeasible", adjustments[0]["detail"])
        self.assertEqual(adjusted, weights)  # unchanged

    def test_empty_allocation_is_handled(self):
        adjusted, adjustments = apply_concentration_cap([])
        self.assertEqual(adjusted, [])
        self.assertEqual(adjustments, [])

    def test_capping_is_deterministic(self):
        weights = make_weights([0.62, 0.20, 0.08, 0.05, 0.03, 0.01, 0.01])
        first, _ = apply_concentration_cap(weights)
        second, _ = apply_concentration_cap(weights)
        self.assertEqual(first, second)


class TestMinimumDiversification(unittest.TestCase):
    def test_diversified_allocation_raises_no_flag(self):
        self.assertEqual(check_minimum_diversification(make_weights([1 / 7] * 7)), [])

    def test_concentrated_allocation_is_flagged_but_not_altered(self):
        weights = make_weights([0.60, 0.395, 0.005, 0.0, 0.0, 0.0, 0.0])
        flags = check_minimum_diversification(weights)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["rule"], "minimum_diversification")
        self.assertIn("not altered", flags[0]["detail"])

    def test_boundary_at_exactly_the_minimum_is_not_flagged(self):
        values = [0.34, 0.33, 0.33] + [0.0] * 4
        self.assertEqual(len([v for v in values if v >= 0.01]), MIN_ASSETS_HELD)
        self.assertEqual(check_minimum_diversification(make_weights(values)), [])


class TestRiskManagerAgent(unittest.TestCase):
    def context_with(self, weights, **extra) -> AgentContext:
        context = AgentContext(request={"variant": "dsr"})
        payload = {"weights": weights, "variant": "dsr", "as_of": "2026-08-19", "policy_loaded": True, "note": None}
        payload.update(extra)
        context.add(AgentOutput(agent_name="execution_agent", payload=payload))
        return context

    def test_passes_through_compliant_weights(self):
        weights = make_weights([1 / 7] * 7)
        output = RiskManagerAgent().run(self.context_with(weights))
        self.assertTrue(output.ok)
        self.assertEqual(output.payload["adjustments"], [])
        self.assertIn("No limit was breached", output.summary)

    def test_reports_the_cap_it_applied(self):
        weights = [{"ticker": t, "weight": w} for t, w in zip(TICKERS, [0.05, 0.05, 0.05, 0.05, 0.05, 0.70, 0.05])]
        output = RiskManagerAgent().run(self.context_with(weights))
        self.assertEqual(len(output.payload["adjustments"]), 1)
        self.assertIn("concentration limit", output.summary)
        self.assertLessEqual(
            max(e["weight"] for e in output.payload["weights"]),
            MAX_SINGLE_ASSET_WEIGHT + 1e-6,
        )

    def test_carries_provenance_forward(self):
        output = RiskManagerAgent().run(self.context_with(make_weights([1 / 7] * 7)))
        self.assertEqual(output.payload["variant"], "dsr")
        self.assertEqual(output.payload["as_of"], "2026-08-19")
        self.assertTrue(output.payload["policy_loaded"])

    def test_degraded_allocation_note_is_preserved(self):
        output = RiskManagerAgent().run(self.context_with(
            make_weights([1 / 7] * 7),
            policy_loaded=False, 
            note="live observation unavailable",
        ))
        self.assertFalse(output.payload["policy_loaded"])
        self.assertEqual(output.payload["note"], "live observation unavailable")

    def test_missing_upstream_allocation_does_not_crash(self):
        empty = AgentContext(request={"variant": "dsr"})
        output = RiskManagerAgent().run(empty)
        self.assertTrue(output.ok)
        self.assertEqual(output.payload["weights"], [])
        self.assertIn("No allocation", output.summary)

    def test_never_uses_an_llm(self):
        self.assertFalse(RiskManagerAgent().uses_llm)


if __name__ == "__main__":
    unittest.main()
