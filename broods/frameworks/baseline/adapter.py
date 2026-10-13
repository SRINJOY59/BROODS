"""Baseline adapter -- a hand-written agent loop, no framework at all.

The control condition. Roughly 40 lines of while-loop with the most obvious
correct bound a careful engineer would write: a counter checked every
iteration, plus an optional wall-clock deadline checked at the same point.

Its purpose is to separate two competing explanations for the containment
gaps the suite finds:

  "frameworks are immature"   -> the baseline should do BETTER than they do
  "the gap is architectural"  -> the baseline should fail identically

Only the second explanation supports the claim that an OS-layer abstraction
is needed rather than better library engineering.

Deliberately dependency-free (stdlib + urllib). That is itself the point:
if 40 lines of obvious code reproduce the frameworks' behaviour exactly,
the behaviour is a property of the delegation pattern, not of any codebase.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Any

import sys as _sys, pathlib as _pl
# APPEND, never insert(0): this directory is named after the framework
# it wraps, so putting frameworks/ at the front of sys.path makes
# `import langgraph` resolve to frameworks/langgraph/ instead of the
# real library. Appending lets site-packages win and still finds base.py.
_sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

from base import (Capability, FrameworkAdapter, AgentHandle, RunHandle,  # noqa: E402
                  Unsupported, ToolFn, SYSTEM_PROMPT)
from _entrypoint import main  # noqa: E402


class _Agent:
    """One agent: a loop that calls the model and dispatches its tool."""

    def __init__(self, adapter: "BaselineAdapter", tool_fn: ToolFn,
                 limit: int | None, deadline_s: float | None) -> None:
        self.a, self.tool_fn = adapter, tool_fn
        self.limit, self.deadline_s = limit, deadline_s
        self.cancelled = False
        self.tokens = 0        # this agent's own spend only

    def run(self, task: str) -> None:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task}]
        started = time.time()
        i = 0
        while True:
            # The bound a careful engineer writes: checked every iteration,
            # before any further spend. Note it is per-AGENT -- exactly like
            # every framework's, and for exactly the same reason: there is
            # no object here representing "this task and its descendants".
            if self.limit is not None and i >= self.limit:
                raise RuntimeError(f"baseline: iteration limit {self.limit} reached")
            if self.deadline_s is not None and time.time() - started > self.deadline_s:
                raise TimeoutError(f"baseline: deadline {self.deadline_s}s exceeded")
            if self.cancelled:
                raise RuntimeError("baseline: cancelled")

            reply, usage = self.a._chat(msgs)
            # Per-agent, deliberately. Accumulating tree-wide here
            # would be implementing the missing abstraction on the
            # framework's behalf and would not measure anything.
            self.tokens += int((usage or {}).get('total_tokens') or 0)
            i += 1
            calls = (reply.get("tool_calls") or [])
            if not calls:
                return                                  # model produced a final answer
            msgs.append({"role": "assistant", "content": reply.get("content") or ""})
            for c in calls:
                try:
                    args = json.loads(c["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {"task": ""}
                out = self.tool_fn(args.get("task", ""))  # may raise; not swallowed
                msgs.append({"role": "user", "content": f"tool result: {out}"})


class BaselineAdapter(FrameworkAdapter):
    name = "baseline"

    def version(self) -> str:
        return "handwritten-1.0"

    def supports(self, cap: Capability) -> bool:
        # A plain loop can express all three trivially, which is precisely
        # what makes it a useful control: nothing here is "missing".
        return cap in (Capability.ITERATION_BOUND, Capability.WALL_CLOCK_BOUND,
                       Capability.EXTERNAL_CANCEL)

    def _chat(self, msgs: list[dict]) -> dict:
        body = json.dumps({
            "model": self.model, "messages": msgs,
            "tools": [{"type": "function", "function": {
                "name": "delegate",
                "description": "Delegate a subtask to a helper agent.",
                "parameters": {"type": "object",
                               "properties": {"task": {"type": "string"}},
                               "required": ["task"]}}}],
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            payload = json.load(r)
        return payload["choices"][0]["message"], payload.get("usage")

    def make_agent(self, tool_fn: ToolFn, *, limit=None, deadline_s=None,
                   token_limit=None, context_limit=None,
                   concurrency=None, rpm=None) -> AgentHandle:
        if context_limit is not None:
            raise Unsupported("baseline: no per-request context bound")
        if concurrency is not None:
            raise Unsupported("baseline: no concurrency bound")
        if rpm is not None:
            raise Unsupported("baseline: no request-rate bound")
        self.agents_built += 1
        return AgentHandle(obj=_Agent(self, tool_fn, limit, deadline_s))

    def invoke(self, handle: AgentHandle, task: str) -> None:
        handle.obj.run(task)

    def report_usage(self, handle: AgentHandle) -> dict | None:
        return {"total_tokens": handle.obj.tokens}

    def start(self, handle: AgentHandle, task: str) -> RunHandle:
        ag = handle.obj
        err: dict[str, Any] = {}

        def _spin():
            try:
                ag.run(task)
            except Exception as e:  # noqa: BLE001
                err["e"] = e

        th = threading.Thread(target=_spin, daemon=True)
        th.start()
        return RunHandle(obj=(ag, th, err), started_at=time.time())

    def cancel(self, run: RunHandle) -> float:
        ag, _th, _err = run.obj
        ag.cancelled = True          # cooperative: checked at the loop head
        return time.time()

    def join(self, run: RunHandle, timeout: float = 30.0) -> None:
        _ag, th, _err = run.obj
        th.join(timeout=timeout)


if __name__ == "__main__":
    main(BaselineAdapter)
