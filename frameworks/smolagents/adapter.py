"""smolagents (Hugging Face) adapter.

  iteration bound : ToolCallingAgent(max_steps=...)
  wall clock      : none native
  external cancel : run() is synchronous with no interrupt hook

ToolCallingAgent rather than CodeAgent: the latter answers by writing and
executing Python, which would let it bypass the delegate tool entirely and
measure something other than the declared bound.
"""
from __future__ import annotations

import warnings
from typing import Any

warnings.filterwarnings("ignore")

import sys as _sys, pathlib as _pl
# APPEND, never insert(0) -- see the note in any sibling adapter.
_sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

from base import (Capability, FrameworkAdapter, AgentHandle,  # noqa: E402
                  Unsupported, ToolFn, SYSTEM_PROMPT)
from _entrypoint import main  # noqa: E402


class SmolagentsAdapter(FrameworkAdapter):
    name = "smolagents"

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("smolagents")

    def supports(self, cap: Capability) -> bool:
        return cap in (Capability.ITERATION_BOUND,)   # == not is; see base.Capability

    def make_agent(self, tool_fn: ToolFn, *, limit=None, deadline_s=None,
                   token_limit=None, context_limit=None,
                   concurrency=None, rpm=None) -> AgentHandle:
        if context_limit is not None:
            raise Unsupported("smolagents: no per-request context bound")
        if concurrency is not None:
            raise Unsupported("smolagents: no concurrency bound")
        if rpm is not None:
            raise Unsupported("smolagents: no request-rate bound")
        if token_limit is not None:
            raise Unsupported("smolagents: no aggregate token budget")
        if deadline_s is not None:
            raise Unsupported("smolagents: no native per-run wall-clock bound")
        from smolagents import OpenAIServerModel, ToolCallingAgent, tool

        @tool
        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent.

            Args:
                task: The subtask to hand to the helper.
            """
            # smolagents parses this docstring for the tool schema, so the
            # Args: section is load-bearing, not decoration.
            return tool_fn(task)

        self.agents_built += 1
        model = OpenAIServerModel(model_id=self.model, api_base=self.base_url,
                                  api_key=self.api_key)
        kwargs: dict[str, Any] = {"tools": [delegate], "model": model}
        if limit is not None:
            kwargs["max_steps"] = limit
        agent = ToolCallingAgent(**kwargs)
        return AgentHandle(obj=agent)

    def invoke(self, handle: AgentHandle, task: str) -> None:
        handle.obj.run(f"{SYSTEM_PROMPT}\n\n{task}")


if __name__ == "__main__":
    main(SmolagentsAdapter)
