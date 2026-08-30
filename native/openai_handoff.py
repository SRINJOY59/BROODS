"""Does Runner(max_turns=L) bound a chain of OpenAI Agents SDK handoffs?

A handoff transfers control to another agent. Unlike agent-as-tool it is not a
nested run, so the aggregate could plausibly stay inside the caller's budget --
which is the question. Two agents hand off to each other indefinitely; only
max_turns can stop the pair.

Run:  .venvs/openai_agents/Scripts/python broods/native/openai_handoff.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from broods.native import mini_proxy  # noqa: E402

PORT = 8145
BASE = f"http://127.0.0.1:{PORT}/v1"
CAP = 200

os.environ["OPENAI_API_KEY"] = "unused"
os.environ["OPENAI_BASE_URL"] = BASE

mini_proxy.serve(PORT)
time.sleep(0.3)

from agents import (Agent, ModelSettings, OpenAIChatCompletionsModel,  # noqa: E402
                    Runner, function_tool, set_tracing_disabled)
from openai import AsyncOpenAI  # noqa: E402

set_tracing_disabled(True)
client = AsyncOpenAI(base_url=BASE, api_key="unused")
MODEL = OpenAIChatCompletionsModel(model="mock", openai_client=client)


@function_tool
def lookup(task: str) -> str:
    """Look a fact up."""
    return "partial result; keep going"


def solo(_L: int) -> Agent:
    return Agent(name="solo", instructions="You never conclude on your own.",
                 tools=[lookup], model=MODEL)


def handoff_pair(_L: int) -> Agent:
    a = Agent(name="alpha", instructions="Always hand off to beta.",
              tools=[lookup], model=MODEL)
    b = Agent(name="beta", instructions="Always hand off to alpha.",
              tools=[lookup], model=MODEL)
    a.handoffs = [b]
    b.handoffs = [a]
    return a


def as_tool_pair(_L: int) -> Agent:
    """The other promoted pattern, for contrast: a nested run, not a transfer."""
    child = Agent(name="child", instructions="You never conclude on your own.",
                  tools=[lookup], model=MODEL)
    return Agent(name="root", instructions="Always call the child agent.",
                 tools=[child.as_tool(tool_name="child_agent",
                                      tool_description="Delegate a subtask.")],
                 model=MODEL)


CASES = [
    ("single agent        ", solo,         ""),
    ("handoff chain       ", handoff_pair, "transfer"),
    ("agent-as-tool       ", as_tool_pair, "child_agent"),
]

for L in (3, 5):
    for label, build, prefer in CASES:
        mini_proxy.reset(cap=CAP, prefer=prefer)
        t0 = time.time()
        try:
            Runner.run_sync(build(L), "Research a topic.", max_turns=L)
            outcome = "COMPLETED"
        except Exception as e:  # noqa: BLE001
            outcome = type(e).__name__
        print(json.dumps({
            "case": f"{label} L={L}",
            "outcome": outcome,
            "llm_calls": mini_proxy.STATE["calls"],
            "declared_limit": L,
            "capped": mini_proxy.STATE["capped"],
            "duration_s": round(time.time() - t0, 1),
        }), flush=True)
