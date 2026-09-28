"""
Goals of this implementation:

- Assembles the other agents' outputs into a single explanation for the user.
- The explanation must describe the allocation and the constraints that were actually applied.
"""
from __future__ import annotations

import re

from orchestration.base import Agent
from orchestration.llm import LLMUnavailable, complete
from orchestration.messages import AgentContext, AgentOutput

SYSTEM_PROMPT = (
    "You explain a portfolio recommendation to a non-expert retail "
    "investor. You are given the recommended weights, the risk constraints "
    "that were applied, and optional market and sentiment context. "
    "Describe only what you are given. Never invent a reason, never restate "
    "a weight incorrectly, and never present the market or sentiment "
    "commentary as the cause of the allocation. It was not an input to it. "
    "State clearly that the allocation came from a model trained on "
    "historical price data. Do not give financial advice. "
    "Write three to five sentences in simple terms."
)

ASSET_NAMES = {
    "SPY": "US large-cap equities",
    "QQQ": "US technology equities",
    "IWM": "US small-cap equities",
    "EFA": "developed international equities",
    "EEM": "emerging-market equities",
    "GLD": "gold",
    "TLT": "long-dated US Treasuries",
}

VARIANT_NAMES = {
    "raw": "raw return",
    "dsr": "differential Sharpe ratio",
    "sortino": "downside (Sortino) risk",
    "cost": "transaction-cost penalty",
}

PERCENTAGE_TOLERANCE = 0.55

_PERCENTAGE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")

# Uses Regex to find all percentages in a string
def extract_percentages(text):
    return [float(match) for match in _PERCENTAGE_PATTERN.findall(text or "")]

# Validation
def unsupported_percentages(reply, source):
    """
    Percentages in 'reply' that do not correspond to anything in 'source'.

    Anything unmatched was not derived from the allocation provided (hallucination).
    """
    allowed = extract_percentages(source)
    unsupported = []
    for value in extract_percentages(reply):
        if not any(abs(value - candidate) <= PERCENTAGE_TOLERANCE
                   for candidate in allowed):
            unsupported.append(value)
    return unsupported


def format_weight_lines(weights: list[dict]):
    ordered = sorted(weights, key=lambda w: -float(w["weight"]))
    return [
        f"  {entry['ticker']} ({ASSET_NAMES.get(entry['ticker'], 'ETF')}): "
        f"{float(entry['weight']) * 100:.1f}%"
        for entry in ordered
    ]

# Deterministic template.
def build_template_explanation(
    allocation: dict,
    market: dict,
    sentiment: dict,
):
    weights = allocation.get("weights", [])
    variant = allocation.get("variant")
    reward_name = VARIANT_NAMES.get(variant, variant or "selected")
    parts: list[str] = []

    if allocation.get("policy_loaded"):
        parts.append(
            f"This allocation comes from a reinforcement-learning policy "
            f"trained on historical price data using the {reward_name} "
            f"reward, applied to market data as of {allocation.get('as_of', 'the latest available date')}."
        )
    else:
        note = allocation.get("note") or "the trained policy was unavailable"
        parts.append(
            f"The trained policy could not be run, so an equal-weight "
            f"placeholder is shown instead ({note}). This is not a model recommendation."
        )

    if weights:
        largest = max(weights, key=lambda w: float(w["weight"]))

        sorted_weights = sorted(weights, key=lambda w: float(w["weight"]), reverse=True)
        weight_summary = "; ".join(f"{w['ticker']} {float(w['weight']):.1%}" for w in sorted_weights)
        ticker = largest["ticker"]
        asset_name = ASSET_NAMES.get(ticker, "ETF")

        parts.append(
            f"The recommended weights are: {weight_summary}. "
            f"The largest holding is {ticker} ({asset_name})."
        )

    adjustments = allocation.get("adjustments", [])
    if adjustments:
        details = " ".join(a["detail"].rstrip(".") + "." for a in adjustments)
        parts.append("The risk limits changed the proposal: " + details)
    elif weights:
        cap = allocation.get("constraints", {}).get("max_single_asset_weight")
        if cap:
            parts.append(
                f"No risk limit was breached: every holding was already "
                f"within the {cap * 100:.0f}% maximum for a single ETF."
            )

    context_notes = []
    if market.get("narrative"):
        context_notes.append(market["narrative"])

    if sentiment.get("classified"):
        context_notes.append(sentiment.get("summary_line", ""))

    if context_notes:
        parts.append(
            "The following context is provided for your own judgement and "
            "did not influence the allocation: " + " ".join(context_notes)
        )

    return " ".join(part for part in parts if part)

# Assemble what the model is allowed to describe.
def build_prompt(allocation: dict, market: dict, sentiment: dict) -> str:
    lines = ["RECOMMENDED ALLOCATION"]
    variant = allocation.get("variant")
    lines.append(f"Reward formulation used in training: {VARIANT_NAMES.get(variant, variant)}")
    lines.append(f"Data as of: {allocation.get('as_of', 'unknown')}")
    
    if allocation.get("policy_loaded"):
        produced_by = "yes"
    else:
        produced_by = ("NO — this is an equal-weight placeholder because the policy could not be run")
    lines.append(f"Produced by the trained policy: {produced_by}")
    
    if allocation.get("note"):
        lines.append(f"Note: {allocation['note']}")

    lines += ["", "WEIGHTS"] + format_weight_lines(allocation.get("weights", []))

    lines += ["", "RISK CONSTRAINTS"]
    constraints = allocation.get("constraints", {})
    
    if constraints.get("max_single_asset_weight"):
        lines.append(f"  Maximum for any single ETF: {constraints['max_single_asset_weight'] * 100:.0f}%")
    
    adjustments = allocation.get("adjustments", [])
    if adjustments:
        lines.append("  Adjustments applied:")
        lines += [f"    - {a['detail']}" for a in adjustments]
    else:
        lines.append("  No constraint was breached; weights are unchanged.")

    lines += ["", "CONTEXT (not an input to the allocation — do not present "
                  "it as a reason for the weights)"]
    if market.get("narrative"):
        lines.append(f"  Market: {market['narrative']}")
    if sentiment.get("classified"):
        lines.append(f"  News sentiment: {sentiment.get('summary_line', '')}")
    if len(lines) and lines[-1].startswith("CONTEXT"):
        lines.append("  None available.")

    lines += ["", "Explain this recommendation to the investor."]
    return "\n".join(lines)


class CoordinatorAgent(Agent):
    name = "coordinator"
    uses_llm = True

    def run(self, context: AgentContext) -> AgentOutput:
        # Prefer the risk-adjusted weights
        # Fall back to the raw allocation
        allocation = (context.payload_of("risk_manager") or context.payload_of("execution_agent"))
        market = context.payload_of("market_analyst")
        sentiment = context.payload_of("sentiment_analyser")

        if not allocation.get("weights"):
            return self.output(
                payload={"explanation": None, "weights": [], "generated": False},
                summary=("No allocation was produced, so there is nothing to explain."),
            )

        template = build_template_explanation(allocation, market, sentiment)
        prompt = build_prompt(allocation, market, sentiment)

        explanation = template
        generated = False
        fallback_reason = None

        try:
            candidate = complete(
                                 prompt, 
                                 role="coordinator", 
                                 system=SYSTEM_PROMPT,
                                 max_tokens=500)
            candidate = candidate.text.strip()
        except LLMUnavailable as error:
            candidate = None
            fallback_reason = str(error)

        if candidate:
            invented = unsupported_percentages(candidate, prompt)
            if invented:
                # If model hallucinated percentages, use the verified template.
                fallback_reason = (
                    "generated explanation has figures not present in the allocation "
                    f"({', '.join(f'{val}%' for val in invented)}). fallback template used instead"
                )
            else:
                explanation = candidate
                generated = True

        return self.output(
            payload={
                "explanation": explanation,
                "generated": generated,
                "fallback_reason": fallback_reason,
                "weights": allocation.get("weights", []),
                "adjustments": allocation.get("adjustments", []),
                "variant": allocation.get("variant"),
                "as_of": allocation.get("as_of"),
                "policy_loaded": allocation.get("policy_loaded", False),
            },
            summary=(
                "Assembled the explained recommendation."
                if generated else
                "Assembled the explained recommendation from the verified template."
            ),
        )
