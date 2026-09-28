"""
Runs a sequence of agents and collects their outputs.

Execution is simple. Agents run in a fixed order,
each seeing everything produced before it. There is no dynamic planning,
no agent-to-agent negotiation, and no loop. 

An agent that raises and error is recorded as a failed output and the
pipeline continues. If the sentiment agent fails, the user should still receive an 
allocation, properly marked.
"""

from __future__ import annotations

from orchestration.base import Agent
from orchestration.messages import AgentContext

class Pipeline:

    # constructors
    def __init__(self, agents: list[Agent]):
        if not agents:
            raise ValueError("pipeline requires at least one agent")
        
        duplicate_names = {a.name for a in agents}
        
        if len(duplicate_names) != len(agents):
            raise ValueError("agent names must be unique within a pipeline")
        
        self.agents = agents

    def run(self, request: dict | None = None) -> AgentContext:
        """Execute every agent in order and return the populated context."""
        context = AgentContext(request=request or {})
        for agent in self.agents:
            try:
                output = agent.run(context)
            except Exception as error:  # backstop; agents should self-handle
                output = agent.output(
                    error=f"{type(error).__name__}: {error}",
                    summary=f"{agent.name} failed and was skipped.",
                )
            context.add(output)
        return context

    def describe(self):
        return [
            {"order": i, "agent": a.name, "uses_llm": a.uses_llm}
            for i, a in enumerate(self.agents, start=1)
        ]
