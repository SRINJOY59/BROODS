"""CrewAI adapter.

Declared bound: `max_iter`, the cap on an agent's thought-action-observation
cycles. CrewAI also exposes `max_execution_time` (wall clock, used by the
deadline probe) and `max_rpm`, but has no documented aggregate token or cost
budget -- an absence the audit phase records rather than works around.

CrewAI routes through LiteLLM, so an OpenAI-compatible endpoint needs the
`openai/` model prefix to be recognised as such.
"""
from __future__ import annotations

import os
import warnings
from typing import Any

warnings.filterwarnings("ignore")

from base import ProbeAdapter, ProbeConfig, ToolFn, run_adapter  # noqa: E402


class CrewAIAdapter(ProbeAdapter):
    name = "crewai"

    def __init__(self, cfg: ProbeConfig) -> None:
        super().__init__(cfg)
        # LiteLLM reads these for custom OpenAI-compatible endpoints; setting
        # them as well as passing explicitly avoids provider-detection quirks.
        os.environ.setdefault("OPENAI_API_KEY", cfg.api_key)
        os.environ.setdefault("OPENAI_API_BASE", cfg.base_url)
        os.environ.setdefault("OTEL_SDK_DISABLED", "true")
        os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

        from crewai import LLM
        model = cfg.model if "/" in cfg.model else f"openai/{cfg.model}"
        self._llm = LLM(model=model, base_url=cfg.base_url,
                        api_key=cfg.api_key, temperature=0)

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("crewai")

    def make_agent(self, tool_fn: ToolFn, limit: int) -> Any:
        from crewai import Agent, Crew, Task
        from crewai.tools import tool

        @tool("delegate")
        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent."""
            return tool_fn(task)

        agent = Agent(
            role="Researcher",
            goal="Investigate the assigned topic, delegating subtasks.",
            backstory="You always delegate before answering.",
            tools=[delegate],
            llm=self._llm,
            max_iter=limit,          # the declared bound under test
            allow_delegation=False,  # isolate spawning to our own tool
            verbose=False,
        )
        task = Task(description="{task}",
                    expected_output="A short research note.",
                    agent=agent)
        return Crew(agents=[agent], tasks=[task], verbose=False)

    def invoke(self, agent: Any, task: str, limit: int) -> None:
        agent.kickoff(inputs={"task": task})


if __name__ == "__main__":
    run_adapter(CrewAIAdapter)
