"""AutoGen (AgentChat) adapter.

Declared bound: `max_tool_iterations` on AssistantAgent. AutoGen has by far the
richest control surface of the three -- eleven termination conditions including
TokenUsageTermination and ExternalTermination -- which makes it the strongest
candidate to falsify the thesis. That is exactly why it is in the suite.

AutoGen is async-native, and the recursive tree calls a child agent from inside
a tool that is itself running on an event loop. Nested asyncio.run() is illegal,
so nested invocations are driven on a dedicated thread with its own loop.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import warnings
from typing import Any

warnings.filterwarnings("ignore")

from base import ProbeAdapter, ProbeConfig, ToolFn, Unsupported, run_adapter  # noqa: E402


def _drive(coro) -> Any:
    """Run a coroutine whether or not a loop is already spinning."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


class AutoGenAdapter(ProbeAdapter):
    name = "autogen"

    def _new_client(self):
        """A fresh client per agent.

        Sharing one client across nested agents deadlocks: the client binds to
        the event loop that created it, and a nested agent driven on its own
        loop then blocks forever waiting on the parent's. One client per agent
        costs nothing here and keeps each loop self-contained.
        """
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        # A non-catalogue model id requires explicit capability declaration.
        return OpenAIChatCompletionClient(
            model=self.cfg.model,
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            },
        )

    def version(self) -> str:
        import importlib.metadata as md
        return md.version("autogen-agentchat")

    def make_agent(self, tool_fn: ToolFn, limit: int) -> Any:
        from autogen_agentchat.agents import AssistantAgent

        def delegate(task: str) -> str:
            """Delegate a subtask to a helper agent."""
            return tool_fn(task)

        kwargs: dict[str, Any] = {
            "name": f"agent_{self.agents_built}",
            "model_client": self._new_client(),
            "tools": [delegate],
            "system_message": "Always delegate before answering.",
        }
        # max_tool_iterations is the documented per-agent bound. If this build
        # predates it, the framework cannot express the limit at all -- record
        # that as UNSUPPORTED rather than emulating a bound ourselves.
        sig = inspect.signature(AssistantAgent.__init__)
        if "max_tool_iterations" in sig.parameters:
            kwargs["max_tool_iterations"] = limit
        else:
            raise Unsupported(
                "AssistantAgent has no max_tool_iterations in this build; "
                "no per-agent iteration bound is expressible")
        return AssistantAgent(**kwargs)

    def invoke(self, agent: Any, task: str, limit: int) -> None:
        _drive(agent.run(task=task))


if __name__ == "__main__":
    run_adapter(AutoGenAdapter)
