"""PydanticAI adapter.

The most interesting system in the suite, because it is the only one found
that exposes an AGGREGATE budget: `UsageLimits(total_tokens_limit=...)`
bounds tokens rather than steps, which is the quantity that actually costs
money. Every other framework offers only a step counter.

  iteration bound : UsageLimits(request_limit=...)
  token bound     : UsageLimits(total_tokens_limit=...)   <- unique here
  wall clock      : none native
  external cancel : agent.run() is a coroutine, so a real asyncio Task

Whether that aggregate budget COMPOSES across delegation is the question
worth asking: it is the one mechanism in the whole suite with a plausible
claim to L4, and either answer is informative.
"""
from __future__ import annotations

import asyncio
import threading
import time
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import sys as _sys, pathlib as _pl
# APPEND, never insert(0): this directory is named after the framework it
# wraps, so putting frameworks/ at the front of sys.path would shadow the
# real library. Appending lets site-packages win and still finds base.py.
_sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

from base import (Capability, FrameworkAdapter, AgentHandle, RunHandle,  # noqa: E402
                  Unsupported, ToolFn, SYSTEM_PROMPT)
from _entrypoint import main  # noqa: E402


def _usage_of(obj) -> Any:
    """Extract a usage object across API shapes.

    `usage` is a plain property on current releases and was a method on
    older ones; calling the property raises TypeError ('RunUsage' object is
    not callable), which a broad except turns into "this framework reports
    no usage" -- a false negative that reads exactly like a real finding.
    """
    u = getattr(obj, "usage", None)
    if callable(u):
        try:
            return u()
        except Exception:  # noqa: BLE001
            return None
    return u


def _drive(coro) -> Any:
    """Run a coroutine whether or not a loop is already spinning here.

    PydanticAI raises UserError on `run_sync()` called inside a synchronous
    tool -- which is exactly what the recursive probe does when a parent
    delegates to a child. Without this, every child failed instantly, the
    probe's tool swallowed it as "child stopped", and the framework scored a
    perfect L4 for producing no traffic at all.
    """
    import concurrent.futures
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


class PydanticAIAdapter(FrameworkAdapter):
    name = "pydantic_ai"

    def version(self) -> str:
        import importlib.metadata as md
        for dist in ("pydantic-ai-slim", "pydantic-ai"):
            try:
                return md.version(dist)
            except Exception:  # noqa: BLE001
                continue
        return "?"

    def supports(self, cap: Capability) -> bool:
        # Introspected, not asserted. The hand-written list this replaced
        # reported CONCURRENCY_BOUND as absent; Agent.max_concurrency exists.
        import inspect
        from pydantic_ai import Agent
        from pydantic_ai.usage import UsageLimits
        a = set(inspect.signature(Agent).parameters)
        u = set(getattr(UsageLimits, "model_fields", {})) | set(
            inspect.signature(UsageLimits).parameters)
        if cap in (Capability.ITERATION_BOUND,):
            return "request_limit" in u
        if cap in (Capability.TOKEN_BOUND,):
            return "total_tokens_limit" in u
        if cap in (Capability.CONTEXT_BOUND,):
            return "per_request_input_tokens_limit" in u
        if cap in (Capability.CONCURRENCY_BOUND,):
            return "max_concurrency" in a
        if cap in (Capability.EXTERNAL_CANCEL,):
            return True          # harness-driven, as for every system
        return False

    def _model(self):
        from pydantic_ai.providers.openai import OpenAIProvider
        import pydantic_ai.models.openai as _m
        # Renamed OpenAIModel -> OpenAIChatModel in recent releases.
        cls = getattr(_m, "OpenAIChatModel", None) or getattr(_m, "OpenAIModel")
        return cls(self.model,
                   provider=OpenAIProvider(base_url=self.base_url,
                                           api_key=self.api_key))

    def make_agent(self, tool_fn: ToolFn, *, limit=None, deadline_s=None,
                   token_limit=None, context_limit=None,
                   concurrency=None, rpm=None) -> AgentHandle:
        # Found by runtime introspection (broods/audit_caps.py), not by the
        # hand-written list this adapter previously used: Agent accepts
        # max_concurrency. An earlier revision reported L0 here in error.
        import inspect as _i
        from pydantic_ai import Agent as _A
        if concurrency is not None and "max_concurrency" not in _i.signature(_A).parameters:
            raise Unsupported("pydantic_ai: no concurrency bound")
        if rpm is not None:
            raise Unsupported("pydantic_ai: no request-rate bound")
        if deadline_s is not None:
            raise Unsupported("pydantic_ai: no native per-run wall-clock bound")
        from pydantic_ai import Agent

        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent."""
            return tool_fn(task)

        self.agents_built += 1
        akw = {}
        if concurrency is not None:
            akw["max_concurrency"] = concurrency
        agent = Agent(self._model(), system_prompt=SYSTEM_PROMPT, tools=[delegate], **akw)
        return AgentHandle(obj=(agent, limit, token_limit, context_limit))

    def _limits(self, limit: int | None, token_limit: int | None = None,
                context_limit: int | None = None):
        from pydantic_ai.usage import UsageLimits
        kw = {}
        if limit:
            kw["request_limit"] = limit          # the step analogue
        if token_limit:
            kw["total_tokens_limit"] = token_limit   # the aggregate budget
        if context_limit:
            # Caps a SINGLE request's context rather than cumulative spend.
            kw["per_request_input_tokens_limit"] = context_limit
        return UsageLimits(**kw) if kw else None

    def invoke(self, handle: AgentHandle, task: str) -> None:
        agent, limit, tok, ctx = handle.obj
        try:
            res = _drive(agent.run(task, usage_limits=self._limits(limit, tok, ctx)))
        except Exception as e:  # noqa: BLE001
            # Preserve usage across the failure path: a run that hit its budget
            # is exactly when an operator most wants the number.
            self._last_usage = (getattr(e, "usage", None)
                                or _usage_of(getattr(e, "result", None)))
            raise
        self._last_usage = _usage_of(res)

    def report_usage(self, handle: AgentHandle) -> dict | None:
        u = getattr(self, "_last_usage", None)
        if u is None:
            return None
        return {"requests": getattr(u, "requests", None),
                "total_tokens": getattr(u, "total_tokens", None)}

    def start(self, handle: AgentHandle, task: str) -> RunHandle:
        agent, limit, tok, ctx = handle.obj
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        holder: dict[str, Any] = {}

        def _spin():
            asyncio.set_event_loop(loop)
            holder["task"] = loop.create_task(
                agent.run(task, usage_limits=self._limits(limit, tok, ctx)))
            ready.set()
            try:
                loop.run_until_complete(holder["task"])
            except (asyncio.CancelledError, Exception):
                pass

        th = threading.Thread(target=_spin, daemon=True)
        th.start()
        ready.wait(timeout=5.0)
        return RunHandle(obj=(loop, holder["task"], th), started_at=time.time())

    def cancel(self, run: RunHandle) -> float:
        loop, task, _th = run.obj
        loop.call_soon_threadsafe(task.cancel)
        return time.time()

    def join(self, run: RunHandle, timeout: float = 30.0) -> None:
        _loop, _task, th = run.obj
        th.join(timeout=timeout)


if __name__ == "__main__":
    main(PydanticAIAdapter)
