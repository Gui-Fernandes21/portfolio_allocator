from __future__ import annotations

from orchestration.base import Agent
from orchestration.messages import AgentContext, AgentOutput

# fixed constraints
MAX_SINGLE_ASSET_WEIGHT = 0.40   # concentration cap per ETF
MIN_ASSETS_HELD = 3              # advisory diversification floor
MEANINGFUL_WEIGHT = 0.01         # below this an ETF is not meaningful
WEIGHT_EPSILON = 1e-9            # numerical tolerance
MAX_CAPPING_PASSES = 20          # cap redistribution iterations


def apply_concentration_cap(
    weights: list[dict],
    cap: float = MAX_SINGLE_ASSET_WEIGHT,
):
    """
    Enforce a per-asset weight cap, redistributing the excess.

    Capping one asset and renormalising can push a second asset above the
    cap, so this repeats until no asset violates it (or the cap becomes
    infeasible). Returns the adjusted weights and a list of adjustments.
    """
    adjustments: list[dict] = []
    if not weights:
        return weights, adjustments

    values = {entry["ticker"]: float(entry["weight"]) for entry in weights}

    # With n assets, a cap below 1/n cannot be satisfied by any portfolio.
    if cap * len(values) < 1.0 - WEIGHT_EPSILON:
        adjustments.append({
            "rule": "concentration_cap",
            "detail": f"cap of {cap:.0%} is infeasible across {len(values)} assets. Left unchanged",
        })
        return weights, adjustments

    original = dict(values)
    capped_tickers: set[str] = set()

    for _ in range(MAX_CAPPING_PASSES):
        violators = [
            ticker for ticker, weight in values.items()
            if weight > cap + WEIGHT_EPSILON
        ]
        if not violators:
            break

        excess = 0.0
        for ticker in violators:
            excess += values[ticker] - cap
            values[ticker] = cap
            capped_tickers.add(ticker)

        receivers = {
            ticker: weight for ticker, weight in values.items()
            if ticker not in capped_tickers
        }
        if not receivers:
            break  # everything is at the capped 

        receiving_total = sum(receivers.values())
        if receiving_total > WEIGHT_EPSILON:
            for ticker, weight in receivers.items():
                values[ticker] += excess * (weight / receiving_total)
        else:
            # spread the excess evenly.
            share = excess / len(receivers)
            for ticker in receivers:
                values[ticker] += share

    for ticker in sorted(capped_tickers):
        adjustments.append({
            "rule": "concentration_cap",
            "detail": (
                f"{ticker} reduced from {original[ticker]:.1%} to "
                f"{values[ticker]:.1%} (cap {cap:.0%}). The difference was redistributed"
            ),
        })

    adjusted = [
        {"ticker": entry["ticker"], "weight": round(values[entry["ticker"]], 6)}
        for entry in weights
    ]
    return adjusted, adjustments


def check_minimum_diversification(
    weights: list[dict],
    minimum: int = MIN_ASSETS_HELD):
    
    held = [e for e in weights if float(e["weight"]) >= MEANINGFUL_WEIGHT]
    if weights and len(held) < minimum:
        return [{
            "rule": "minimum_diversification",
            "detail": (
                f"only {len(held)} of {len(weights)} ETFs carry a meaningful "
                f"weight (at least {MEANINGFUL_WEIGHT:.0%}); below the "
                f"guideline of {minimum}. Flagged for the user, not altered."
            ),
        }]
    return []


class RiskManagerAgent(Agent):
    name = "risk_manager"
    uses_llm = False  # deterministic

    def run(self, context: AgentContext) -> AgentOutput:
        allocation = context.payload_of("execution_agent")
        weights = allocation.get("weights", [])

        constraints = {
            "max_single_asset_weight": MAX_SINGLE_ASSET_WEIGHT,
            "min_assets_held": MIN_ASSETS_HELD,
        }

        if not weights:
            return self.output(
                payload={"weights": [], "adjustments": [], "constraints": constraints},
                summary=("No allocation was produced. No risk checks could be applied.")
            )

        weights, adjustments = apply_concentration_cap(weights)
        adjustments += check_minimum_diversification(weights)

        enforced = [a for a in adjustments if a["rule"] == "concentration_cap"]
        if enforced:
            summary = (
                f"Applied {len(enforced)} concentration limit(s): "
                f"no ETF may exceed {MAX_SINGLE_ASSET_WEIGHT:.0%} of the portfolio."
            )
        else:
            summary = (
                "Checked the allocation against the fixed risk limits "
                f"(max {MAX_SINGLE_ASSET_WEIGHT:.0%} per ETF). No limit was breached"
            )
        if any(a["rule"] == "minimum_diversification" for a in adjustments):
            summary += " The portfolio is flagged as under-diversified."

        return self.output(
            payload={
                "weights": weights,
                "adjustments": adjustments,
                "constraints": constraints,
                "variant": allocation.get("variant"),
                "as_of": allocation.get("as_of"),
                "policy_loaded": allocation.get("policy_loaded", False),
                "note": allocation.get("note"),
            },
            summary=summary,
        )
