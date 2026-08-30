"""LangGraph adapter.

Declared bound: `recursion_limit`, LangGraph's only documented limit on agent
execution. Note it is declared per-invocation rather than per-agent, and it
counts GRAPH STEPS, not LLM calls -- the gap between those two units is itself
one of the things the suite measures.
"""
from __future__ import annotations

import warnings
from typing import Any

warnings.filterwarnings("ignore")

from base import ProbeAdapter, ProbeConfig, ToolFn, run_adapter  # noqa: E402


class LangGraphAdapter(ProbeAdapter):
    name = "langgraph"

    def __init__(self, cfg: ProbeConfig) -> None:
        super().__init__(cfg)
        from langchain_openai import ChatOpenAI
        self._llm = ChatOpenAI(model=cfg.model, base_url=cfg.base_url,
                               api_key=cfg.api_key, temperature=0)

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("langgraph")

    def make_agent(self, tool_fn: ToolFn, limit: int) -> Any:
        from langchain_core.tools import tool
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError:  # LangGraph >= 2.0
            from langchain.agents import create_agent as create_react_agent

        @tool
        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent."""
            return tool_fn(task)

        # recursion_limit is supplied at invoke time, not construction time.
        return create_react_agent(self._llm, [delegate])

    def invoke(self, agent: Any, task: str, limit: int) -> None:
        agent.invoke({"messages": [{"role": "user", "content": task}]},
                     config={"recursion_limit": limit})


if __name__ == "__main__":
    run_adapter(LangGraphAdapter)
