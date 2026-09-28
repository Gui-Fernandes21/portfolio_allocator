"""
orchestration
=============
A small in-process orchestration layer written for this project.

It provides the generic machinery for running a sequence of specialised
agents and collecting an auditable trace of what each one contributed:

    messages.py   the data passed between agents
    base.py       the Agent interface every agent implements
    pipeline.py   the runner that executes agents in order
    llm.py        the ONLY module permitted to call a language model

Nothing in this package knows anything about portfolios, ETFs, or reward
functions; the project-specific agents live in ``backend/agents``. The
split is deliberate — it keeps this layer testable without market data and
makes the agent roles readable on their own.

Note on naming: this is a purpose-built module for this system, not a
general-purpose agent framework, and it is deliberately not named after
any existing project.
"""
