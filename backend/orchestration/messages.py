"""
The data structures agents exchange.

'AgentContext' -> The request parameters plus everything earlier agents in the pipeline produced.
'AgentOutput'  -> A payload for downstream agents plus a summary for the reasoning trail.

Every output is retained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AgentOutput:
    """One agent's contribution to the pipeline."""
    
    # The name of the agent that produced this output.
    agent_name: str
    # Structured data for downstream agents to consume.
    payload: dict[str, Any] = field(default_factory=dict)
    # One or two sentences the user could read, for the reasoning trail.
    summary: str = ""
    # True when this output came from a language model, so the UI can
    # visually distinguish generated commentary from computed results.
    used_llm: bool = False
    # Populated when the agent failed; the pipeline records rather than raises.
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class AgentContext:
    """Request context for the pipeline. This is what each agent sees as input."""

    # Request parameters (e.g. which reward variant the user selected).
    request: dict[str, Any] = field(default_factory=dict)
    # Completed outputs, in execution order.
    outputs: list[AgentOutput] = field(default_factory=list)

    # add a new outpuot to the context
    def add(self, output: AgentOutput) -> None:
        self.outputs.append(output)

    # get the most recent output from a named agent.
    def get(self, agent_name: str) -> AgentOutput | None:
        for output in reversed(self.outputs): # search backwards so the last output is returned
            if output.agent_name == agent_name:
                return output
        return None

    # get the payload of a named agent
    def payload_of(self, agent_name: str) -> dict[str, Any]:
        output = self.get(agent_name)
        return output.payload if output and output.ok else {}

    # Get the reasoning trail
    def trace(self) -> list[dict[str, Any]]:
        return [
            {
                "agent": output.agent_name,
                "summary": output.summary,
                "used_llm": output.used_llm,
                "ok": output.ok,
                "error": output.error,
            }
            for output in self.outputs
        ]
