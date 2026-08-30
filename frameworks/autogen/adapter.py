"""AutoGen (AgentChat) adapter.

Iteration bound: `max_tool_iterations` on AssistantAgent.

Wall-clock bound: AutoGen's TimeoutTermination is a Team-level termination
condition; a bare AssistantAgent.run() takes no deadline parameter of its
own. Reported unsupported for the same reason LangGraph is -- wrapping the
call in an external timer would test the harness, not the framework.

External cancel: AssistantAgent.run(task=...) returns a coroutine, so
start() drives it as a real asyncio Task and cancel() calls Task.cancel().
Each agent gets its OWN model client (see _new_client): sharing one client
across nested agents deadlocks, because the client binds to the event loop
that created it and a nested agent driven on its own loop then blocks
forever waiting on the parent's -- discovered empirically while building
the spawn probe, not a hypothetical.
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


def _drive(coro) -> Any:
    """Run a coroutine whether or not a loop is already spinning on this thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


class AutoGenAdapter(FrameworkAdapter):
    name = "autogen"

    def _new_client(self):
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        return OpenAIChatCompletionClient(
            model=self.model, base_url=self.base_url, api_key=self.api_key,
            model_info={"vision": False, "function_calling": True,
                       "json_output": False, "family": "unknown",
                       "structured_output": False})

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("autogen-agentchat")

    def supports(self, cap: Capability) -> bool:
        return cap in (Capability.ITERATION_BOUND, Capability.EXTERNAL_CANCEL,
                       Capability.WALL_CLOCK_BOUND, Capability.TOKEN_BOUND)

    def make_agent(self, tool_fn: ToolFn, *, limit=None, deadline_s=None,
                   token_limit=None, context_limit=None,
                   concurrency=None, rpm=None) -> AgentHandle:
        if context_limit is not None:
            raise Unsupported("autogen: no per-request context bound")
        if concurrency is not None:
            raise Unsupported("autogen: no concurrency bound")
        if rpm is not None:
            raise Unsupported("autogen: no request-rate bound")
        from autogen_agentchat.agents import AssistantAgent

        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent."""
            return tool_fn(task)

        self.agents_built += 1
        kwargs: dict[str, Any] = dict(
            name=f"agent_{self.agents_built}", model_client=self._new_client(),
            tools=[delegate], system_message=SYSTEM_PROMPT)
        if limit is not None:
            import inspect
            sig = inspect.signature(AssistantAgent.__init__)
            if "max_tool_iterations" not in sig.parameters:
                raise Unsupported("AssistantAgent has no max_tool_iterations "
                                  "in this build")
            kwargs["max_tool_iterations"] = limit
        agent = AssistantAgent(**kwargs)

        # A deadline or a token budget is expressed as a Team termination
        # condition, so wrap the single agent in a one-member team.
        conds = []
        if deadline_s is not None:
            from autogen_agentchat.conditions import TimeoutTermination
            conds.append(TimeoutTermination(timeout_seconds=deadline_s))
        if token_limit is not None:
            from autogen_agentchat.conditions import TokenUsageTermination
            conds.append(TokenUsageTermination(max_total_token=token_limit))
        if conds:
            from autogen_agentchat.teams import RoundRobinGroupChat
            cond = conds[0]
            for c in conds[1:]:
                cond = cond | c
            return AgentHandle(obj=RoundRobinGroupChat([agent],
                                                       termination_condition=cond))
        return AgentHandle(obj=agent)

    def invoke(self, handle: AgentHandle, task: str) -> None:
        self._last_result = _drive(handle.obj.run(task=task))

    def report_usage(self, handle: AgentHandle) -> dict | None:
        res = getattr(self, "_last_result", None)
        msgs = getattr(res, "messages", None)
        if not msgs:
            return None
        total = 0
        seen = False
        for m in msgs:
            u = getattr(m, "models_usage", None)
            if u is not None:
                seen = True
                total += int(getattr(u, "prompt_tokens", 0) or 0)
                total += int(getattr(u, "completion_tokens", 0) or 0)
        return {"total_tokens": total} if seen else None

    def start(self, handle: AgentHandle, task: str) -> RunHandle:
        agent = handle.obj
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        holder: dict[str, Any] = {}

        def _spin():
            asyncio.set_event_loop(loop)
            holder["task"] = loop.create_task(agent.run(task=task))
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
    main(AutoGenAdapter)
