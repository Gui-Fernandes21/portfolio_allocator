"""
The Coordinator is the only agent that writes to the user, so the
explanation must state the real weights and must return even when no model is available.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import agents.coordinator as coordinator 
from orchestration.llm import LLMUnavailable  
from orchestration.messages import AgentContext, AgentOutput  

TICKERS = ["EEM", "EFA", "GLD", "IWM", "QQQ", "SPY", "TLT"]

WEIGHTS = [
    {"ticker": ticker, "weight": weight}
    for ticker, weight in zip(TICKERS, [0.0563, 0.075, 0.0938, 0.0938, 0.2062, 0.40, 0.075])
]

CAPPED_ALLOCATION = {
    "weights": WEIGHTS,
    "variant": "dsr",
    "as_of": "2026-08-19",
    "policy_loaded": True,
    "constraints": {"max_single_asset_weight": 0.40},
    "adjustments": [{
        "rule": "concentration_cap",
        "detail": "SPY reduced from 68.0% to 40.0% (cap 40%). The difference was redistributed across the other holdings",
    }],
}


def context_with(allocation=None, market=None, sentiment=None):
    context = AgentContext(request={"variant": "dsr"})
    if market is not None:
        context.add(AgentOutput(agent_name="market_analyst", payload=market))
    if sentiment is not None:
        context.add(AgentOutput(agent_name="sentiment_analyser", payload=sentiment))
    if allocation is not None:
        context.add(AgentOutput(agent_name="risk_manager", payload=allocation))
    return context


class TestNumericFaithfulness(unittest.TestCase):
    def test_restated_figures_pass(self):
        source = "SPY 40.0%, QQQ 20.6%"
        self.assertEqual(coordinator.unsupported_percentages("SPY is 40.0%.", source), [])

    def test_rounding_is_tolerated(self):
        source = "SPY 40.0%, QQQ 20.6%"
        self.assertEqual(coordinator.unsupported_percentages("SPY 40%, QQQ 21%", source), [])

    def test_invented_figures_are_caught(self):
        source = "SPY 40.0%, QQQ 20.6%"
        self.assertEqual(coordinator.unsupported_percentages("SPY is 55% after a 12% gain.", source),[55.0, 12.0])

    def test_negative_percentages_handled(self):
        self.assertEqual(coordinator.unsupported_percentages("down -8.0%", "fell -8.0%"), [])

    def test_text_without_percentages_is_fine(self):
        self.assertEqual(coordinator.unsupported_percentages("No numbers here.", "x"), [])


class TestTemplateExplanation(unittest.TestCase):

    def test_states_every_weight_correctly(self):
        text = coordinator.build_template_explanation(CAPPED_ALLOCATION, {}, {})

        for entry in WEIGHTS:
            self.assertIn(
                f"{entry['ticker']} {entry['weight'] * 100:.1f}%", text,
                f"{entry['ticker']} missing or misstated",
            )

    # The fallback must also pass the check.
    def test_template_invents_no_figures(self):
        prompt = coordinator.build_prompt(CAPPED_ALLOCATION, {}, {})
        template = coordinator.build_template_explanation(CAPPED_ALLOCATION, {}, {})

        self.assertEqual(coordinator.unsupported_percentages(template, prompt), [])

    def test_reports_the_constraint_that_fired(self):
        text = coordinator.build_template_explanation(CAPPED_ALLOCATION, {}, {})
        self.assertIn("risk limits changed the proposal", text)
        self.assertIn("SPY reduced", text)

    def test_says_so_when_no_constraint_fired(self):
        allocation = dict(CAPPED_ALLOCATION, adjustments=[])
        text = coordinator.build_template_explanation(allocation, {}, {})
        self.assertIn("No risk limit was breached", text)

    def test_degraded_allocation_is_not_presented_as_a_recommendation(self):
        allocation = dict(CAPPED_ALLOCATION, policy_loaded=False, note="live observation unavailable", adjustments=[])
        text = coordinator.build_template_explanation(allocation, {}, {})
        self.assertIn("not a model recommendation", text)
        self.assertIn("live observation unavailable", text)

    def test_context_is_marked_as_not_influencing_the_allocation(self):
        text = coordinator.build_template_explanation(
            CAPPED_ALLOCATION,
            {"narrative": "Equities rose."},
            {"classified": True, "summary_line": "SPY positive."},
        )
        self.assertIn("did not influence the allocation", text)


class TestPromptAssembly(unittest.TestCase):
    def test_prompt_contains_weights_and_constraints(self):
        prompt = coordinator.build_prompt(CAPPED_ALLOCATION, {}, {})
        self.assertIn("SPY (US large-cap equities): 40.0%", prompt)
        self.assertIn("Maximum for any single ETF: 40%", prompt)
        self.assertIn("Adjustments applied", prompt)

    def test_prompt_labels_context_as_non_causal(self):
        prompt = coordinator.build_prompt(CAPPED_ALLOCATION, {"narrative": "Equities rose."}, {})
        self.assertIn("not an input to the allocation", prompt)

    def test_prompt_flags_a_degraded_allocation(self):
        prompt = coordinator.build_prompt(dict(CAPPED_ALLOCATION, policy_loaded=False), {}, {})
        self.assertIn("NO", prompt)
        self.assertIn("placeholder", prompt)


class TestCoordinatorAgent(unittest.TestCase):
    def test_uses_the_generated_text_when_it_is_faithful(self):
        faithful = ("The policy recommends SPY at 40.0% and QQQ at 20.6%, with the rest spread across the remaining ETFs.")
        with mock.patch.object(coordinator, "complete") as fake:
            fake.return_value = mock.Mock(text=faithful)
            output = coordinator.CoordinatorAgent().run(context_with(CAPPED_ALLOCATION))

        self.assertTrue(output.payload["generated"])
        self.assertEqual(output.payload["explanation"], faithful)
        self.assertIsNone(output.payload["fallback_reason"])

    def test_discards_generated_text_that_invents_a_figure(self):
        unfaithful = "The policy recommends SPY at 55.0%, its highest ever."
        with mock.patch.object(coordinator, "complete") as fake:
            fake.return_value = mock.Mock(text=unfaithful)
            output = coordinator.CoordinatorAgent().run(context_with(CAPPED_ALLOCATION))

        self.assertFalse(output.payload["generated"])
        self.assertNotIn("55.0%", output.payload["explanation"])
        self.assertIn("figures not present", output.payload["fallback_reason"])
        # The user still gets a correct explanation.
        self.assertIn("SPY 40.0%", output.payload["explanation"])

    def test_falls_back_to_template_without_a_model(self):
        with mock.patch.object(coordinator, "complete", side_effect=LLMUnavailable("no key")):
            output = coordinator.CoordinatorAgent().run(context_with(CAPPED_ALLOCATION))

        self.assertTrue(output.ok)
        self.assertFalse(output.payload["generated"])
        self.assertIn("SPY 40.0%", output.payload["explanation"])
        self.assertIn("no key", output.payload["fallback_reason"])

    def test_prefers_risk_manager_weights_over_raw_allocation(self):
        """The explanation should still describe the post-constraint portfolio."""
        context = AgentContext(request={"variant": "dsr"})
        context.add(AgentOutput(
            agent_name="execution_agent",
            payload={"weights": [{"ticker": "SPY", "weight": 0.68}], "variant": "dsr", "policy_loaded": True}))
        context.add(AgentOutput(agent_name="risk_manager", payload=CAPPED_ALLOCATION))

        with mock.patch.object(coordinator, "complete", side_effect=LLMUnavailable("no key")):
            output = coordinator.CoordinatorAgent().run(context)

        self.assertIn("SPY 40.0%", output.payload["explanation"])
        self.assertNotIn("68.0%", output.payload["explanation"].split("risk limits")[0])

    def test_no_allocation_yields_no_explanation(self):
        with mock.patch.object(coordinator, "complete") as fake:
            output = coordinator.CoordinatorAgent().run(context_with(None))
        fake.assert_not_called()
        self.assertTrue(output.ok)
        self.assertIsNone(output.payload["explanation"])

    def test_never_alters_the_weights(self):
        with mock.patch.object(coordinator, "complete", side_effect=LLMUnavailable("no key")):
            output = coordinator.CoordinatorAgent().run(context_with(CAPPED_ALLOCATION))
        self.assertEqual(output.payload["weights"], WEIGHTS)


if __name__ == "__main__":
    unittest.main()
