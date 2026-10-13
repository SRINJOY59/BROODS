"""Strands Agents (AWS) adapter.

  iteration bound : agent(task, limits={"turns": L})           per invocation
  token budget    : agent(task, limits={"total_tokens": T})    per invocation
  external cancel : agent(task, cancel_signal=threading.Event())
  wall clock      : none native (the OpenAI client timeout is per request)

Strands documents its `Limits` as bounding "a single invoke_async /
stream_async call only; counters are not cumulative" -- per agent by design,
which is exactly what the depth series tests.
"""
from __future__ import annotations

import threading
import time
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import sys as _sys, pathlib as _pl
# APPEND, never insert(0): this directory is named `strands`, and putting
# frameworks/ first would make `import strands` resolve to it.
_sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

from base import (Capability, FrameworkAdapter, AgentHandle, RunHandle,  # noqa: E402
                  Unsupported, ToolFn, SYSTEM_PROMPT)
from _entrypoint import main  # noqa: E402


class StrandsAdapter(FrameworkAdapter):
    name = "strands"

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("strands-agents")

    def supports(self, cap: Capability) -> bool:
        return cap in (Capability.ITERATION_BOUND, Capability.TOKEN_BOUND,
                       Capability.EXTERNAL_CANCEL)

    def make_agent(self, tool_fn: ToolFn, *, limit=None, deadline_s=None,
                   token_limit=None, context_limit=None,
                   concurrency=None, rpm=None) -> AgentHandle:
        if deadline_s is not None:
            raise Unsupported("strands: no per-invocation wall-clock bound")
        if context_limit is not None:
            raise Unsupported("strands: no per-request context bound")
        if concurrency is not None:
            raise Unsupported("strands: no declarable concurrency bound")
        if rpm is not None:
            raise Unsupported("strands: no request-rate bound")
        from strands import Agent, tool
        from strands.models.openai import OpenAIModel

        @tool
        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent.

            Args:
                task: the subtask to delegate
            """
            return tool_fn(task)

        # One model object per agent: the OpenAI client inside binds to the
        # event loop that first uses it, and nested agents run on others.
        model = OpenAIModel(client_args={"base_url": self.base_url,
                                         "api_key": self.api_key},
                            model_id=self.model)
        self.agents_built += 1
        agent = Agent(model=model, tools=[delegate], system_prompt=SYSTEM_PROMPT,
                      callback_handler=None, name=f"agent_{self.agents_built}")
        limits: dict[str, int] = {}
        if limit is not None:
            limits["turns"] = int(limit)
        if token_limit is not None:
            limits["total_tokens"] = int(token_limit)
        return AgentHandle(obj={"agent": agent, "limits": limits or None,
                                "result": None})

    def _call(self, h: dict, task: str, cancel: threading.Event | None = None):
        kw: dict[str, Any] = {"limits": h["limits"]}
        if cancel is not None:
            kw["cancel_signal"] = cancel
        h["result"] = h["agent"](task, **kw)
        return h["result"]

    def invoke(self, handle: AgentHandle, task: str) -> None:
        self._call(handle.obj, task)

    def report_usage(self, handle: AgentHandle) -> dict | None:
        res = handle.obj.get("result")
        u = getattr(getattr(res, "metrics", None), "accumulated_usage", None)
        if not u:
            return None
        t = u.get("totalTokens") if isinstance(u, dict) else getattr(u, "totalTokens", None)
        return None if t is None else {"total_tokens": int(t)}

    def start(self, handle: AgentHandle, task: str) -> RunHandle:
        ev = threading.Event()

        def _spin():
            try:
                self._call(handle.obj, task, ev)
            except Exception:  # noqa: BLE001
                pass
        th = threading.Thread(target=_spin, daemon=True)
        th.start()
        return RunHandle(obj=(ev, th), started_at=time.time())

    def cancel(self, run: RunHandle) -> float:
        ev, _th = run.obj
        ev.set()
        return time.time()

    def join(self, run: RunHandle, timeout: float = 30.0) -> None:
        run.obj[1].join(timeout=timeout)


if __name__ == "__main__":
    main(StrandsAdapter)
