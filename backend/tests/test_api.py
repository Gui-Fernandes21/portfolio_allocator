from __future__ import annotations

import sys
import types
import unittest
from datetime import date
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

FAKE_INDICATORS = ["macd", "rsi_30", "cci_30"]


def _stub_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module


# Registering Stubs
_stub_module("finrl")
_stub_module("finrl.config", INDICATORS=FAKE_INDICATORS)
_stub_module("yfinance", download=None)

try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi not installed")
class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.main import app
        from app.services import policy_store
        import app.services.observation_builder as observation_builder
        from app.services.observation_builder import Observation

        cls.client = TestClient(app)
        cls.policy_store = policy_store
        cls.observation_builder = observation_builder
        cls.tickers = ["EEM", "EFA", "GLD", "IWM", "QQQ", "SPY", "TLT"]
        cls.fixed_observation = Observation(
            state=np.zeros((7 + len(FAKE_INDICATORS), 7)),
            as_of=date(2024, 12, 30),
            tickers=cls.tickers,
        )
        cls._saved_weights_path = policy_store.model_weights_path
        cls._saved_get_observation = observation_builder.get_todays_observation

    def setUp(self):
        self.policy_store._loaded_policies.clear()
        self.policy_store.model_weights_path = type(self)._saved_weights_path
        self.observation_builder.get_todays_observation = type(
            self
        )._saved_get_observation

    def test_allocation_returns_softmaxed_policy_action(self):
        class FakePolicy:
            def predict(self, state, deterministic=True):
                assert deterministic
                return np.array([1.0, 0, 0, 0, 0, 0, 2.0]), None

        self.policy_store._loaded_policies["dsr"] = FakePolicy()
        self.observation_builder._cache.clear()
        self.observation_builder.get_todays_observation = lambda: self.fixed_observation

        body = self.client.post("/api/v1/allocation", json={"variant": "dsr"}).json()
        self.assertTrue(body["policy_loaded"])
        self.assertIsNone(body["note"])
        self.assertEqual(body["as_of"], "2024-12-30")
        self.assertEqual([w["ticker"] for w in body["weights"]], self.tickers)

        weights = [w["weight"] for w in body["weights"]]
        expected = np.exp([1, 0, 0, 0, 0, 0, 2.0])
        expected = expected / expected.sum()
        self.assertTrue(np.allclose(weights, expected, atol=1e-5))
        self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_invalid_variant_rejected_with_422(self):
        response = self.client.post("/api/v1/allocation", json={"variant": "bogus"})
        self.assertEqual(response.status_code, 422)

    def test_default_variant_is_dsr(self):
        self.policy_store.model_weights_path = lambda variant: Path(
            "/nonexistent/ppo.zip"
        )
        body = self.client.post("/api/v1/allocation", json={}).json()
        self.assertEqual(body["variant"], "dsr")

    def test_health_reports_available_variants(self):
        self.policy_store.model_weights_path = lambda variant: Path(
            "/nonexistent/ppo.zip"
        )
        body = self.client.get("/api/v1/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["available_variants"], [])

    # fallbacks
    def _assert_equal_weight_stub(self, body):
        self.assertFalse(body["policy_loaded"])
        self.assertIsNotNone(body["note"])
        weights = [w["weight"] for w in body["weights"]]
        self.assertTrue(np.allclose(weights, 1 / 7))
        self.assertAlmostEqual(sum(weights), 1.0, places=9)

    def test_missing_weights_falls_back_to_flagged_stub(self):
        self.policy_store.model_weights_path = lambda variant: Path(
            "/nonexistent/ppo.zip"
        )
        body = self.client.post("/api/v1/allocation", json={"variant": "raw"}).json()
        self._assert_equal_weight_stub(body)
        self.assertIn("no frozen weights", body["note"])

    def test_unloadable_weights_fall_back_to_flagged_stub(self):
        self.policy_store.model_weights_path = (
            lambda variant: BACKEND_DIR / "app" / "main.py"
        )
        _stub_module(
            "stable_baselines3",
            PPO=types.SimpleNamespace(
                load=lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError("bad zip")
                )
            ),
        )
        body = self.client.post(
            "/api/v1/allocation", json={"variant": "sortino"}
        ).json()
        self._assert_equal_weight_stub(body)
        self.assertIn("failed to load weights", body["note"])

    def test_observation_failure_falls_back_to_flagged_stub(self):
        class FakePolicy:
            def predict(self, state, deterministic=True):
                return np.zeros(7), None

        self.policy_store._loaded_policies["dsr"] = FakePolicy()
        self.observation_builder._cache.clear()

        def raise_provider_down():
            raise RuntimeError("provider down")

        self.observation_builder.get_todays_observation = raise_provider_down
        body = self.client.post("/api/v1/allocation", json={"variant": "dsr"}).json()
        self._assert_equal_weight_stub(body)
        self.assertIn("live observation unavailable", body["note"])


if __name__ == "__main__":
    unittest.main()
