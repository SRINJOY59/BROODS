"""LlamaIndex adapter.

  iteration bound : NONE -- no such parameter on either agent class
  wall clock      : FunctionAgent(timeout=...)
  request rate    : LLM(rate_limiter=TokenBucketRateLimiter(...))
  external cancel : none exposed

Two corrections over an earlier version of this adapter, both of which had
been producing wrong rows:

FunctionAgent, not ReActAgent. ReActAgent drives the model with text-based
Thought/Action prompting and ignores native `tool_calls`, so the suite's
injector -- which replies with OpenAI-style tool calls -- never caused it to
invoke a tool at all. Every probe that needed the agent to call `delegate`
reported "tool never executed", which reads like a finding and is really an
adapter mismatch. FunctionAgent uses native function calling and responds to
the injector as every other framework does.

`rate_limiter` is a real request-rate bound. It hangs off the LLM object
rather than the agent, so a sweep of the agent constructors alone missed it
and this row was previously reported L0; `llama_index.core.llms.callbacks`
acquires from it before every chat completion.

`timeout` is a real deadline. Both workflow agent classes accept it, so
LlamaIndex was previously reported L0 on the deadline probe when it in fact
exposes a wall-clock bound. The iteration-bound L0 is genuine: neither class
accepts max_iterations, max_steps or max_turns
(checked against 0.14.24's constructor signature at runtime, not assumed).

KNOWN LIMIT -- tool invocation
------------------------------
FunctionAgent sends a correct `tools` array and receives a well-formed
OpenAI `tool_calls` reply, and then does not execute the tool: one request,
zero invocations, verified against a minimal standalone server. Probes that
require the agent to actually call `delegate` (state_isolation,
fault_isolation, accounting, reclamation) therefore report ERR rather than a
level. That is an adapter/compatibility gap on this path, NOT evidence about
LlamaIndex's containment behaviour, and must not be read as one.

Rows that ARE valid here: spawn (L0 -- no iteration bound), deadline
(timeout is enforced), token_budget (L0), cancellation (L0). Those depend
only on construction and on a single model call, neither of which needs a
tool to fire.
"""
from __future__ import annotations

import inspect
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import sys as _sys, pathlib as _pl
# APPEND, never insert(0) -- see the note in any sibling adapter.
_sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

from base import (Capability, FrameworkAdapter, AgentHandle,  # noqa: E402
                  Unsupported, ToolFn, SYSTEM_PROMPT)
from _entrypoint import main  # noqa: E402

_BOUND_PARAMS = ("max_iterations", "max_steps", "max_turns")


def _drive(coro) -> Any:
    """Run a coroutine whether or not a loop is already spinning here.

    The workflow agents are async-only, and the recursive probe calls
    children from inside a tool that is already running on a loop.
    """
    import asyncio
    import concurrent.futures
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _agent_cls():
    from llama_index.core.agent.workflow import FunctionAgent
    return FunctionAgent


def _params() -> set[str]:
    try:
        return set(inspect.signature(_agent_cls().__init__).parameters)
    except Exception:  # noqa: BLE001
        return set()


class LlamaIndexAdapter(FrameworkAdapter):
    name = "llamaindex"

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("llama-index-core")

    def supports(self, cap: Capability) -> bool:
        p = _params()
        if cap in (Capability.ITERATION_BOUND,):   # == not is; see base.Capability
            return any(b in p for b in _BOUND_PARAMS)
        if cap in (Capability.WALL_CLOCK_BOUND,):
            return "timeout" in p
        if cap in (Capability.RATE_BOUND,):
            try:
                from llama_index.core.rate_limiter import TokenBucketRateLimiter  # noqa: F401
            except Exception:  # noqa: BLE001
                return False
            from llama_index.core.base.llms.base import BaseLLM
            return "rate_limiter" in getattr(BaseLLM, "model_fields", {})
        return False

    def _llm(self, rpm=None):
        """OpenAILike, not OpenAI.

        LlamaIndex's OpenAI class validates the model name against a
        hardcoded catalogue and raises ValueError("Unknown model 'mock'")
        before issuing a single request -- so every probe recorded zero calls
        against a synthetic model name the proxy ignores anyway. Alone among
        the frameworks tested it refuses to talk to an OpenAI-compatible
        endpoint whose model it does not recognise; OpenAILike exists for
        exactly that case.
        """
        from llama_index.llms.openai_like import OpenAILike
        kw = {}
        if rpm is not None:
            from llama_index.core.rate_limiter import TokenBucketRateLimiter
            # One limiter per agent, as every other bound in the suite is
            # declared. LlamaIndex documents sharing one instance across LLM
            # objects; that variant is discussed in the paper, not scored here.
            kw["rate_limiter"] = TokenBucketRateLimiter(requests_per_minute=rpm)
        return OpenAILike(model=self.model, api_base=self.base_url,
                          api_key=self.api_key, is_chat_model=True,
                          is_function_calling_model=True, context_window=8192,
                          **kw)

    def make_agent(self, tool_fn: ToolFn, *, limit=None, deadline_s=None,
                   token_limit=None, context_limit=None,
                   concurrency=None, rpm=None) -> AgentHandle:
        if context_limit is not None:
            raise Unsupported("llamaindex: no per-request context bound")
        if concurrency is not None:
            raise Unsupported("llamaindex: no concurrency bound")
        if token_limit is not None:
            raise Unsupported("llamaindex: no aggregate token budget")
        from llama_index.core.tools import FunctionTool

        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent."""
            return tool_fn(task)

        p = _params()
        bound = next((b for b in _BOUND_PARAMS if b in p), None)
        if limit is not None and bound is None:
            raise Unsupported(
                f"llamaindex {self.version()}: agent classes expose no "
                f"iteration bound (none of {', '.join(_BOUND_PARAMS)})")
        if deadline_s is not None and "timeout" not in p:
            raise Unsupported("llamaindex: no per-run wall-clock bound")

        llm = self._llm(rpm=rpm)
        kwargs: dict[str, Any] = {
            "name": f"agent_{self.agents_built}",
            "description": "research agent",
            "tools": [FunctionTool.from_defaults(
                fn=delegate, name="delegate",
                description="Delegate a subtask to a helper agent.")],
            "llm": llm,
            "system_prompt": SYSTEM_PROMPT,
        }
        if limit is not None:
            kwargs[bound] = limit
        if deadline_s is not None:
            kwargs["timeout"] = deadline_s

        self.agents_built += 1
        # Construction is DEFERRED into the event loop. The workflow agents
        # allocate asyncio primitives in __init__, so building one from
        # ordinary synchronous code raises "no running event loop" before any
        # request is ever made -- which surfaced as a RuntimeError with zero
        # LLM calls on every probe. The handle carries a thunk instead, and
        # the agent is created inside the same loop that runs it.
        return AgentHandle(obj=lambda: _agent_cls()(**kwargs))

    def invoke(self, handle: AgentHandle, task: str) -> None:
        build = handle.obj

        async def _build_and_run():
            return await build().run(user_msg=task)

        _drive(_build_and_run())


if __name__ == "__main__":
    main(LlamaIndexAdapter)
