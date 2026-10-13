"""Agno adapter.

  iteration bound : Agent(tool_call_limit=L)   -- counts TOOL CALLS, not model calls
  external cancel : Agent.cancel_run(run_id)
  token budget    : none (OpenAILike max_tokens bounds one completion's output)
  wall clock      : none native (OpenAILike timeout is per request)

Telemetry is disabled: Agno otherwise reports runs to its own API, which is
both outbound traffic the operator did not ask for and noise at the socket.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("AGNO_TELEMETRY", "false")

import sys as _sys, pathlib as _pl
# APPEND, never insert(0) -- this directory is named `agno`.
_sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

from base import (Capability, FrameworkAdapter, AgentHandle, RunHandle,  # noqa: E402
                  Unsupported, ToolFn, SYSTEM_PROMPT)
from _entrypoint import main  # noqa: E402


class AgnoAdapter(FrameworkAdapter):
    name = "agno"

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("agno")

    def supports(self, cap: Capability) -> bool:
        return cap in (Capability.ITERATION_BOUND, Capability.EXTERNAL_CANCEL)

    def make_agent(self, tool_fn: ToolFn, *, limit=None, deadline_s=None,
                   token_limit=None, context_limit=None,
                   concurrency=None, rpm=None) -> AgentHandle:
        if deadline_s is not None:
            raise Unsupported("agno: no per-run wall-clock bound")
        if token_limit is not None:
            raise Unsupported("agno: no aggregate token budget")
        if context_limit is not None:
            raise Unsupported("agno: no per-request context bound")
        if concurrency is not None:
            raise Unsupported("agno: no concurrency bound")
        if rpm is not None:
            raise Unsupported("agno: no request-rate bound")
        from agno.agent import Agent
        from agno.models.openai.like import OpenAILike

        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent.

            Args:
                task: the subtask to delegate
            """
            return tool_fn(task)

        self.agents_built += 1
        kw = {"tool_call_limit": int(limit)} if limit is not None else {}
        agent = Agent(model=OpenAILike(id=self.model, base_url=self.base_url,
                                       api_key=self.api_key),
                      tools=[delegate], instructions=SYSTEM_PROMPT,
                      telemetry=False, name=f"agent_{self.agents_built}", **kw)
        return AgentHandle(obj={"agent": agent, "result": None})

    def invoke(self, handle: AgentHandle, task: str) -> None:
        h = handle.obj
        h["result"] = h["agent"].run(task)

    def report_usage(self, handle: AgentHandle) -> dict | None:
        res = handle.obj.get("result")
        m = getattr(res, "metrics", None)
        t = getattr(m, "total_tokens", None)
        return None if t is None else {"total_tokens": int(t)}

    def start(self, handle: AgentHandle, task: str) -> RunHandle:
        h, rid = handle.obj, str(uuid.uuid4())

        def _spin():
            try:
                h["result"] = h["agent"].run(task, run_id=rid)
            except Exception:  # noqa: BLE001
                pass
        th = threading.Thread(target=_spin, daemon=True)
        th.start()
        return RunHandle(obj=(h["agent"], rid, th), started_at=time.time())

    def cancel(self, run: RunHandle) -> float:
        agent, rid, _th = run.obj
        agent.cancel_run(rid)
        return time.time()

    def join(self, run: RunHandle, timeout: float = 30.0) -> None:
        run.obj[2].join(timeout=timeout)


if __name__ == "__main__":
    main(AgnoAdapter)
