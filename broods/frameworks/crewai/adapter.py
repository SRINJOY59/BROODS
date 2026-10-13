"""CrewAI adapter.

Iteration bound: `max_iter` on Agent, CrewAI's documented cap on
thought-action-observation cycles.

Wall-clock bound: `max_execution_time` on Agent -- a per-run deadline, unlike
LangGraph's per-step `step_timeout`.

External cancel: `kickoff()` is fully synchronous and blocking with no
interrupt hook in this build. Reported unsupported rather than faked with a
thread-kill, which Python cannot do safely or honestly for a blocking call.
"""
from __future__ import annotations

import os
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import sys as _sys, pathlib as _pl
# APPEND, never insert(0): this directory is named after the framework
# it wraps, so putting frameworks/ at the front of sys.path makes
# `import langgraph` resolve to frameworks/langgraph/ instead of the
# real library. Appending lets site-packages win and still finds base.py.
_sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

from base import (Capability, FrameworkAdapter, AgentHandle,  # noqa: E402
                  Unsupported, ToolFn, SYSTEM_PROMPT)
from _entrypoint import main  # noqa: E402


class CrewAIAdapter(FrameworkAdapter):
    name = "crewai"

    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        super().__init__(model, base_url, api_key)
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        os.environ.setdefault("OPENAI_API_BASE", base_url)
        os.environ.setdefault("OTEL_SDK_DISABLED", "true")
        os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
        from crewai import LLM
        # Always prefix openai/ -- we always route through our own
        # OpenAI-compatible proxy, regardless of which real model or provider
        # sits behind it, and the real model's id often itself contains a
        # "/" (e.g. "nvidia/nemotron-..."), which previously suppressed the
        # prefix entirely and made LiteLLM try (and fail) to resolve nvidia
        # as a native provider instead of speaking OpenAI's protocol to us.
        self._llm = self._make_llm()

    def _make_llm(self):
        from crewai import LLM
        return LLM(model=f"openai/{self.model}", base_url=self.base_url,
                   api_key=self.api_key, temperature=0)

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("crewai")

    def supports(self, cap: Capability) -> bool:
        # Introspected, not asserted. The previous hand-written list reported
        # TOKEN_BOUND as absent; Agent.max_tokens exists.
        from crewai import Agent
        f = set(getattr(Agent, "model_fields", {}))
        if cap in (Capability.ITERATION_BOUND,):
            return "max_iter" in f
        if cap in (Capability.WALL_CLOCK_BOUND,):
            return "max_execution_time" in f
        if cap in (Capability.RATE_BOUND,):
            return "max_rpm" in f
        if cap in (Capability.TOKEN_BOUND,):
            return "max_tokens" in f
        return False

    def make_agent(self, tool_fn: ToolFn, *, limit=None, deadline_s=None,
                   token_limit=None, context_limit=None,
                   concurrency=None, rpm=None) -> AgentHandle:
        if context_limit is not None:
            raise Unsupported("crewai: no per-request context bound")
        if concurrency is not None:
            raise Unsupported("crewai: no concurrency bound")
        from crewai import Agent, Crew, Task
        from crewai.tools import tool

        @tool("delegate")
        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent."""
            return tool_fn(task)

        own_llm = self._make_llm()   # per-agent, for the ablation
        kwargs: dict[str, Any] = dict(
            role="Researcher",
            goal="Investigate the assigned topic, delegating subtasks.",
            backstory=SYSTEM_PROMPT,
            tools=[delegate], llm=own_llm, allow_delegation=False, verbose=False,
        )
        if limit is not None:
            kwargs["max_iter"] = limit
        # Found by introspection: Agent.max_tokens, documented as "Maximum
        # number of tokens for the agent's execution". Whether that is an
        # aggregate budget or a per-call output cap is what the probe decides.
        if token_limit is not None:
            kwargs["max_tokens"] = token_limit
        if deadline_s is not None:
            kwargs["max_execution_time"] = deadline_s
        if rpm is not None:
            kwargs["max_rpm"] = rpm

        self.agents_built += 1
        agent = Agent(**kwargs)
        crew_task = Task(description="{task}", expected_output="A short research note.",
                         agent=agent)
        return AgentHandle(obj=Crew(agents=[agent], tasks=[crew_task], verbose=False))

    def invoke(self, handle: AgentHandle, task: str) -> None:
        handle.obj.kickoff(inputs={"task": task})

    def report_usage(self, handle: AgentHandle) -> dict | None:
        m = getattr(handle.obj, "usage_metrics", None)
        if m is None:
            return None
        total = getattr(m, "total_tokens", None)
        if total is None and isinstance(m, dict):
            total = m.get("total_tokens")
        return None if total is None else {"total_tokens": int(total)}


if __name__ == "__main__":
    main(CrewAIAdapter)
