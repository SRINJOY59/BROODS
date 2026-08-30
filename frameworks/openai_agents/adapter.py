"""OpenAI Agents SDK adapter.

  iteration bound : Runner.run(max_turns=...)
  wall clock      : none native
  external cancel : Runner.run() is a coroutine, so a real asyncio Task

Tracing is disabled explicitly: it issues its own HTTP calls to OpenAI's
tracing endpoint, which would both fail against the local proxy and pollute
the call counts the proxy exists to measure.
"""
from __future__ import annotations

import asyncio
import threading
import time
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import sys as _sys, pathlib as _pl
# APPEND, never insert(0) -- see the note in any sibling adapter.
_sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

from base import (Capability, FrameworkAdapter, AgentHandle, RunHandle,  # noqa: E402
                  Unsupported, ToolFn, SYSTEM_PROMPT)
from _entrypoint import main  # noqa: E402


def _drive(coro) -> Any:
    """Run a coroutine whether or not a loop is already spinning here.

    The recursive probe calls a child agent from inside a tool, and that tool
    is already executing on the parent's event loop. Runner.run_sync() calls
    asyncio.run() internally, which is illegal while a loop is running -- so
    nested invocations get their own thread and their own loop.
    """
    import concurrent.futures
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


class OpenAIAgentsAdapter(FrameworkAdapter):
    name = "openai_agents"

    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        super().__init__(model, base_url, api_key)
        from agents import set_default_openai_api, set_tracing_disabled
        set_tracing_disabled(True)   # would otherwise emit its own HTTP traffic
        # Defaults to the Responses API (/responses); the proxy speaks
        # Chat Completions, so without this every call 404s and the
        # probe records zero traffic.
        set_default_openai_api("chat_completions")

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("openai-agents")

    def supports(self, cap: Capability) -> bool:
        return cap in (Capability.ITERATION_BOUND, Capability.EXTERNAL_CANCEL)

    def make_agent(self, tool_fn: ToolFn, *, limit=None, deadline_s=None,
                   token_limit=None, context_limit=None,
                   concurrency=None, rpm=None) -> AgentHandle:
        if context_limit is not None:
            raise Unsupported("openai_agents: no per-request context bound")
        if concurrency is not None:
            raise Unsupported("openai_agents: no concurrency bound")
        if rpm is not None:
            raise Unsupported("openai_agents: no request-rate bound")
        if token_limit is not None:
            raise Unsupported("openai_agents: no aggregate token budget")
        if deadline_s is not None:
            raise Unsupported("openai_agents: no native per-run wall-clock bound")
        from agents import Agent, function_tool

        @function_tool
        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent."""
            return tool_fn(task)

        # A per-agent client, not the global default: a shared AsyncOpenAI
        # binds to whichever event loop first uses it, and a nested agent
        # driven on its own loop then blocks on it forever. Same deadlock
        # AutoGen exhibited, same fix.
        from agents import OpenAIChatCompletionsModel
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

        self.agents_built += 1
        agent = Agent(name=f"agent_{self.agents_built}", instructions=SYSTEM_PROMPT,
                      tools=[delegate],
                      model=OpenAIChatCompletionsModel(model=self.model,
                                                       openai_client=client))
        return AgentHandle(obj=(agent, limit))

    def invoke(self, handle: AgentHandle, task: str) -> None:
        from agents import Runner
        agent, limit = handle.obj
        self._last_result = _drive(Runner.run(agent, task, max_turns=limit or 10))

    def report_usage(self, handle: AgentHandle) -> dict | None:
        res = getattr(self, "_last_result", None)
        u = getattr(getattr(res, "context_wrapper", None), "usage", None)
        t = getattr(u, "total_tokens", None)
        return None if t is None else {"total_tokens": int(t)}

    def start(self, handle: AgentHandle, task: str) -> RunHandle:
        from agents import Runner
        agent, limit = handle.obj
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        holder: dict[str, Any] = {}

        def _spin():
            asyncio.set_event_loop(loop)
            holder["task"] = loop.create_task(
                Runner.run(agent, task, max_turns=limit or 10))
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
    main(OpenAIAgentsAdapter)
