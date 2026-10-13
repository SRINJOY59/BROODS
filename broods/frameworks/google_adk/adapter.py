"""Google Agent Development Kit (ADK) adapter.

  iteration bound : RunConfig(max_llm_calls=L)   -- counted in MODEL CALLS,
                    per invocation (default 500)
  wall clock      : LlmAgent(timeout=s)  -- a workflow-node timeout
                    ("the node is cancelled"); measured, not assumed, for a
                    standalone agent run
  external cancel : Runner.run_async is an async generator -> asyncio Task

The model is reached through LiteLLM ("openai/<model>" with api_base), which
is ADK's documented route to any OpenAI-compatible endpoint. ADK's own
AgentTool passes the caller's RunConfig to the nested run, but its source
states "the count itself is still per-invocation, so the ceiling bounds the
nested run rather than being shared with the caller's".
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
import warnings
from typing import Any

warnings.filterwarnings("ignore")
os.environ.setdefault("LITELLM_LOG", "ERROR")

import sys as _sys, pathlib as _pl
# APPEND, never insert(0) -- see the note in any sibling adapter.
_sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

from base import (Capability, FrameworkAdapter, AgentHandle, RunHandle,  # noqa: E402
                  Unsupported, ToolFn, SYSTEM_PROMPT)
from _entrypoint import main  # noqa: E402


def _drive(coro) -> Any:
    """Run a coroutine here, or on a fresh loop in a worker thread when a
    loop is already running (a child agent invoked from inside a tool)."""
    import concurrent.futures
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


class GoogleADKAdapter(FrameworkAdapter):
    name = "google_adk"

    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        super().__init__(model, base_url, api_key)
        # Import LiteLLM now, outside any timed window. ADK imports it lazily
        # on first model use, and a cold import costs 7.5-9.5 s on the test
        # machine -- enough to exhaust a 5 s deadline before the first request
        # exists, which made the deadline probe measure Python's import time.
        import litellm  # noqa: F401
        from google.adk.models.lite_llm import LiteLlm  # noqa: F401

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("google-adk")

    def supports(self, cap: Capability) -> bool:
        return cap in (Capability.ITERATION_BOUND, Capability.WALL_CLOCK_BOUND,
                       Capability.EXTERNAL_CANCEL)

    def make_agent(self, tool_fn: ToolFn, *, limit=None, deadline_s=None,
                   token_limit=None, context_limit=None,
                   concurrency=None, rpm=None) -> AgentHandle:
        if token_limit is not None:
            raise Unsupported("google_adk: no aggregate token budget")
        if context_limit is not None:
            raise Unsupported("google_adk: no per-request context bound")
        if concurrency is not None:
            raise Unsupported("google_adk: no concurrency bound")
        if rpm is not None:
            raise Unsupported("google_adk: no request-rate bound")
        from google.adk.agents import LlmAgent
        from google.adk.models.lite_llm import LiteLlm

        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent.

            Args:
                task: the subtask to delegate
            """
            return tool_fn(task)

        self.agents_built += 1
        kw = {"timeout": float(deadline_s)} if deadline_s is not None else {}
        agent = LlmAgent(
            name=f"agent_{self.agents_built}",
            model=LiteLlm(model=f"openai/{self.model}", api_base=self.base_url,
                          api_key=self.api_key),
            instruction=SYSTEM_PROMPT, tools=[delegate], **kw)
        return AgentHandle(obj={"agent": agent, "limit": limit, "tokens": 0})

    async def _run(self, h: dict, task: str) -> None:
        from google.adk.agents.run_config import RunConfig
        from google.adk.runners import InMemoryRunner
        from google.genai import types
        runner = InMemoryRunner(agent=h["agent"], app_name="broods")
        sess = await runner.session_service.create_session(app_name="broods",
                                                           user_id="u")
        cfg = RunConfig(max_llm_calls=h["limit"]) if h["limit"] else RunConfig()
        msg = types.Content(role="user", parts=[types.Part(text=task)])
        async for ev in runner.run_async(user_id="u", session_id=sess.id,
                                         new_message=msg, run_config=cfg):
            um = getattr(ev, "usage_metadata", None)
            if um is not None and getattr(um, "total_token_count", None):
                h["tokens"] += int(um.total_token_count)

    def invoke(self, handle: AgentHandle, task: str) -> None:
        _drive(self._run(handle.obj, task))

    def report_usage(self, handle: AgentHandle) -> dict | None:
        return {"total_tokens": handle.obj["tokens"]}

    def start(self, handle: AgentHandle, task: str) -> RunHandle:
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        holder: dict[str, Any] = {}

        def _spin():
            asyncio.set_event_loop(loop)
            holder["task"] = loop.create_task(self._run(handle.obj, task))
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
        run.obj[2].join(timeout=timeout)


if __name__ == "__main__":
    main(GoogleADKAdapter)
