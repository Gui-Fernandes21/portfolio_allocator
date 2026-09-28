"""
Goals of this implementation:

- Produces the recommended allocation.
"""

from __future__ import annotations

from orchestration.base import Agent
from orchestration.messages import AgentContext, AgentOutput

DEFAULT_VARIANT = "dsr"

class ExecutionAgent(Agent):
    name = "execution_agent"
    uses_llm = False # deterministic

    def run(self, context: AgentContext) -> AgentOutput:
        variant = context.request.get("variant") or DEFAULT_VARIANT

        from app.services.allocation_service import get_allocation

        try:
            allocation = get_allocation(variant)
        except Exception as error:
            return self.output(
                error=f"{type(error).__name__}: {error}",
                summary="Could not produce an allocation.",
            )

        weights = [{"ticker": w.ticker, "weight": float(w.weight)} for w in allocation.weights]

        if allocation.policy_loaded:
            summary = (
                f"Ran the frozen {variant} policy on market data to "
                f"{allocation.as_of} and produced weights for {len(weights)} ETFs."
            )
        else:
            summary = (f"Could not run the policy, equal-weight placeholder shown instead: {allocation.note}")

        return self.output(
            payload={
                "variant": allocation.variant,
                "weights": weights,
                "as_of": allocation.as_of.isoformat(),
                "policy_loaded": allocation.policy_loaded,
                "note": allocation.note,
            },
            summary=summary,
        )
