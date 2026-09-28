from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from agents import build_pipeline 
from orchestration.base import Agent 
from orchestration.messages import AgentContext, AgentOutput 
from orchestration.pipeline import Pipeline 

DETERMINISTIC_AGENTS = ["execution_agent", "risk_manager"]
LLM_MODULE = "orchestration.llm"

LOCAL_PACKAGES = ("agents", "orchestration", "app")


def imported_modules(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def module_source_path(module_name: str) -> Path | None:
    """Resolve a local module name to its file, or None if it is not mine."""
    if not module_name.startswith(LOCAL_PACKAGES):
        return None
    relative = Path(*module_name.split("."))
    for candidate in (
        BACKEND_DIR / relative.with_suffix(".py"),
        BACKEND_DIR / relative / "__init__.py",
    ):
        if candidate.exists():
            return candidate
    return None


def transitively_imported_modules(module_path: Path) -> set[str]:
    seen_files: set[Path] = set()
    reachable: set[str] = set()
    queue = [module_path]

    while queue:
        current = queue.pop()
        if current in seen_files:
            continue
        seen_files.add(current)
        for name in imported_modules(current):
            reachable.add(name)
            resolved = module_source_path(name)
            if resolved is not None and resolved not in seen_files:
                queue.append(resolved)
    return reachable


class TestAllocationPathIsLLMFree(unittest.TestCase):
    """The property the transparency argument depends on."""

    def test_deterministic_agents_do_not_import_the_llm_client(self):
        """Direct imports: the first line of defence."""
        for agent_module in DETERMINISTIC_AGENTS:
            path = BACKEND_DIR / "agents" / f"{agent_module}.py"
            self.assertTrue(path.exists(), f"missing {path}")
            imports = imported_modules(path)
            offending = {n for n in imports if n.startswith(LLM_MODULE)}
            self.assertEqual(
                offending,
                set(),
                f"{agent_module}.py imports {offending}. Agents on the allocation path must not reach a language model",
            )

    def test_allocation_path_does_not_reach_the_llm_client_transitively(self):
        for agent_module in DETERMINISTIC_AGENTS:
            path = BACKEND_DIR / "agents" / f"{agent_module}.py"
            reachable = transitively_imported_modules(path)
            offending = {n for n in reachable if n.startswith(LLM_MODULE)}
            self.assertEqual(
                offending,
                set(),
                f"{agent_module}.py transitively reaches {offending}; the "
                "allocation path must remain free of language-model calls",
            )
            self.assertNotIn(
                "openai",
                {n.split(".")[0] for n in reachable},
                f"{agent_module}.py transitively reaches the openai package",
            )

    def test_deterministic_agents_declare_uses_llm_false(self):
        pipeline = build_pipeline()
        by_name = {a.name: a for a in pipeline.agents}
        for agent_module in DETERMINISTIC_AGENTS:
            self.assertIn(agent_module, by_name)
            self.assertFalse(
                by_name[agent_module].uses_llm,
                f"{agent_module} must declare uses_llm = False",
            )

    def test_llm_client_is_the_only_openai_importer(self):
        """No agent may bypass the single LLM entry point."""
        for path in (BACKEND_DIR / "agents").glob("*.py"):
            imports = imported_modules(path)
            self.assertNotIn(
                "openai",
                {n.split(".")[0] for n in imports},
                f"{path.name} imports openai directly. all model calls must go through {LLM_MODULE}",
            )


class TestPipelineShape(unittest.TestCase):
    def setUp(self):
        self.pipeline = build_pipeline()

    def test_five_agents_in_documented_order(self):
        self.assertEqual(
            [a.name for a in self.pipeline.agents],
            [
                "market_analyst",
                "sentiment_analyser",
                "execution_agent",
                "risk_manager",
                "coordinator",
            ],
        )

    def test_risk_manager_runs_after_execution(self):
        names = [a.name for a in self.pipeline.agents]
        self.assertLess(
            names.index("execution_agent"),
            names.index("risk_manager"),
            "constraints must be applied to an existing allocation",
        )

    def test_coordinator_runs_last(self):
        self.assertEqual(self.pipeline.agents[-1].name, "coordinator")

    def test_describe_exposes_the_llm_free_segment(self):
        described = self.pipeline.describe()
        self.assertEqual(len(described), 5)
        llm_free = [d["agent"] for d in described if not d["uses_llm"]]
        self.assertEqual(sorted(llm_free), sorted(DETERMINISTIC_AGENTS))


class TestPipelineExecution(unittest.TestCase):
    def test_agents_run_in_order_and_see_earlier_outputs(self):
        class First(Agent):
            name = "first"

            def run(self, context: AgentContext) -> AgentOutput:
                return self.output(payload={"value": 1}, summary="one")

        class Second(Agent):
            name = "second"

            def run(self, context: AgentContext) -> AgentOutput:
                seen = context.payload_of("first").get("value")
                return self.output(payload={"seen": seen}, summary="two")

        context = Pipeline([First(), Second()]).run()
        self.assertEqual([o.agent_name for o in context.outputs], ["first", "second"])
        self.assertEqual(context.payload_of("second")["seen"], 1)

    def test_a_failing_agent_does_not_stop_the_pipeline(self):
        class Exploding(Agent):
            name = "exploding"

            def run(self, context: AgentContext) -> AgentOutput:
                raise RuntimeError("provider down")

        class Later(Agent):
            name = "later"

            def run(self, context: AgentContext) -> AgentOutput:
                return self.output(summary="still ran")

        context = Pipeline([Exploding(), Later()]).run()
        self.assertFalse(context.get("exploding").ok)
        self.assertIn("provider down", context.get("exploding").error)
        self.assertTrue(context.get("later").ok)

    def test_trace_records_every_agent(self):
        context = Pipeline(build_pipeline().agents).run({"variant": "dsr"})
        trace = context.trace()
        self.assertEqual(len(trace), 5)
        self.assertTrue(all("summary" in entry for entry in trace))

    def test_empty_pipeline_rejected(self):
        with self.assertRaises(ValueError):
            Pipeline([])


if __name__ == "__main__":
    unittest.main()
