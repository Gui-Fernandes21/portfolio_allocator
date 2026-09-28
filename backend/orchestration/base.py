"""
The interface every agent implements.

Agents should have:
    1. Name - a unique identifier for the agent.
    2. uses_llm - a declaration of whether it uses a language model.
    3. A 'run' method that turns a context into an output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from orchestration.messages import AgentContext, AgentOutput


class Agent(ABC):
    """Base class for all agents in the pipeline."""

    # Agent identifier to trace reasoning in the pipeline.
    name = "agent"

    # Whether this agent calls a language model.
    uses_llm = False

    @abstractmethod
    def run(self, context: AgentContext) -> AgentOutput:
        """
        Produce this agent's contribution.

        Implementations should return an 'AgentOutput' always, even when errors happpen.
        Agents are expected to handle their own failures.
        """
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    def output(
        self,
        payload: dict | None = None,
        summary: str = "",
        error: str | None = None,
    ) -> AgentOutput:
        """Build an AgentOutput already tagged with this agent's identity."""
        
        return AgentOutput(
            agent_name=self.name,
            payload=payload or {},
            summary=summary,
            used_llm=self.uses_llm,
            error=error,
        )
