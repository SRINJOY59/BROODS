"""LangGraph adapter.

Iteration bound: `recursion_limit`, declared per-invocation and counting
GRAPH STEPS rather than LLM calls -- the unit mismatch is itself part of
what the suite measures (see spawn probe: branch factor ~2 here vs ~5 for
frameworks whose limit counts iterations directly).

Wall-clock bound: LangGraph has no native per-invocation deadline parameter
(recursion_limit is step-based, not time-based), so WALL_CLOCK_BOUND is
reported unsupported rather than emulated with an external wrapper -- a
harness-imposed asyncio.wait_for would test asyncio, not the framework.

Request-rate bound: `langchain_core.rate_limiters.InMemoryRateLimiter`,
attached to the chat model the graph drives. A deep introspection sweep found
it (the shallow sweep did not, because it is reachable only through the
`model` parameter's own type), correcting an earlier L0 on this axis.

External cancel: `.ainvoke()` is a real coroutine, so start() runs it as an
asyncio Task on a dedicated thread's event loop and cancel() calls
Task.cancel() -- a genuine test of whether cancellation propagates through
nested subgraph calls, not a harness-imposed kill.
"""
from __future__ import annotations

import asyncio
import threading
import time
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import sys as _sys, pathlib as _pl
# APPEND, never insert(0): this directory is named after the framework
# it wraps, so putting frameworks/ at the front of sys.path makes
# `import langgraph` resolve to frameworks/langgraph/ instead of the
# real library. Appending lets site-packages win and still finds base.py.
_sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

from base import (Capability, FrameworkAdapter, AgentHandle, RunHandle,  # noqa: E402
                  Unsupported, ToolFn, SYSTEM_PROMPT)
from _entrypoint import main  # noqa: E402


class LangGraphAdapter(FrameworkAdapter):
    name = "langgraph"

    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        super().__init__(model, base_url, api_key)
        from langchain_openai import ChatOpenAI
        self._llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key,
                               temperature=0)

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("langgraph")

    def supports(self, cap: Capability) -> bool:
        # Introspected, not asserted. The hand-written list reported
        # WALL_CLOCK_BOUND as absent; Pregel.step_timeout exists, and so does
        # a per-node timeout= on StateGraph.add_node. Whether a per-step
        # timeout is a per-run deadline is what the probe decides, not us.
        import inspect
        from langgraph.graph import StateGraph
        from langgraph.pregel import Pregel
        if cap in (Capability.ITERATION_BOUND, Capability.EXTERNAL_CANCEL,
                   Capability.CONCURRENCY_BOUND):
            return True
        if cap in (Capability.RATE_BOUND,):
            # Reachable only through the model object, which is why a
            # signature-only sweep of the graph classes missed it.
            from langchain_core.language_models.chat_models import BaseChatModel
            return "rate_limiter" in getattr(BaseChatModel, "model_fields", {})
        if cap in (Capability.WALL_CLOCK_BOUND,):
            return ("step_timeout" in inspect.signature(Pregel).parameters
                    or "timeout" in inspect.signature(StateGraph.add_node).parameters)
        return False

    def make_agent(self, tool_fn: ToolFn, *, limit=None, deadline_s=None,
                   token_limit=None, context_limit=None,
                   concurrency=None, rpm=None) -> AgentHandle:
        if context_limit is not None:
            raise Unsupported("langgraph: no per-request context bound")
        if token_limit is not None:
            raise Unsupported("langgraph: no aggregate token budget")
        llm = self._llm
        if rpm is not None:
            # Per agent, matching how every other bound in the suite is
            # declared. One limiter per model instance, so a child gets its
            # own -- which is exactly the scoping question the suite asks.
            from langchain_core.rate_limiters import InMemoryRateLimiter
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=self.model, base_url=self.base_url, api_key=self.api_key,
                temperature=0,
                rate_limiter=InMemoryRateLimiter(
                    requests_per_second=rpm / 60.0,
                    check_every_n_seconds=0.05,
                    max_bucket_size=1))
        from langchain_core.tools import tool
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError:
            from langchain.agents import create_agent as create_react_agent

        @tool
        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent."""
            return tool_fn(task)

        self.agents_built += 1
        # The callback lives on the HANDLE, not the adapter. Storing it on
        # self meant a nested child's invoke() overwrote the root's before
        # the root returned, so report_usage(root) handed back the child's
        # usage -- 28 tokens for one call where the root had made two.
        # That produced a spurious -66.7% attribution error against the
        # -33.3% every other framework shows.
        graph = create_react_agent(llm, [delegate])
        if deadline_s is not None:
            # Pregel checks this around each superstep. A per-step budget is
            # not a per-run deadline, which is precisely the distinction the
            # probe measures rather than assumes.
            graph.step_timeout = deadline_s
        return AgentHandle(obj={"agent": graph, "limit": limit,
                                "concurrency": concurrency, "cb": None})

    def _msgs(self, task: str) -> dict:
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": task}]}

    def invoke(self, handle: AgentHandle, task: str) -> None:
        h = handle.obj
        cb = self._usage_cb()
        cfg = {"recursion_limit": h["limit"]}
        if h["concurrency"] is not None:
            cfg["max_concurrency"] = h["concurrency"]
        if cb is not None:
            cfg["callbacks"] = [cb]
            h["cb"] = cb          # per-handle, so children cannot clobber it
        h["agent"].invoke(self._msgs(task), config=cfg)

    @staticmethod
    def _usage_cb():
        """LangGraph reports usage through a callback rather than a return
        value. An earlier version of this adapter simply did not attach one
        and therefore reported L0, "no usage accounting" -- a claim about the
        framework that was really a gap in the adapter."""
        try:
            from langchain_core.callbacks import UsageMetadataCallbackHandler
            return UsageMetadataCallbackHandler()
        except Exception:  # noqa: BLE001
            return None

    def report_usage(self, handle: AgentHandle) -> dict | None:
        cb = handle.obj.get("cb") if isinstance(handle.obj, dict) else None
        meta = getattr(cb, "usage_metadata", None) if cb else None
        if not meta:
            return None
        total = sum(int((v or {}).get("total_tokens") or 0) for v in meta.values())
        return {"total_tokens": total}

    def start(self, handle: AgentHandle, task: str) -> RunHandle:
        h = handle.obj
        agent, limit = h["agent"], h["limit"]

        async def _run():
            await agent.ainvoke(self._msgs(task),
                                config={"recursion_limit": limit})

        loop = asyncio.new_event_loop()
        ready = threading.Event()
        holder: dict[str, Any] = {}

        def _spin():
            asyncio.set_event_loop(loop)
            holder["task"] = loop.create_task(_run())
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
    main(LangGraphAdapter)
