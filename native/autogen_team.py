"""Does an AutoGen team's bound survive nesting a team inside a team?

AutoGen expresses two different limits: `max_tool_iterations` bounds one
AssistantAgent's tool loop, and `max_turns` bounds a team's turns. Its native
composition construct is SocietyOfMindAgent, which wraps an inner team and
presents it to an outer team as a single agent -- the closest analogue to
LangGraph's compiled subgraph-as-node, and the case where the aggregate could
in principle be tracked.

Run:  .venvs/autogen/Scripts/python broods/native/autogen_team.py
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from broods.native import mini_proxy  # noqa: E402

PORT = 8144
BASE = f"http://127.0.0.1:{PORT}/v1"
CAP = 200

mini_proxy.serve(PORT)
time.sleep(0.3)

from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent  # noqa: E402
from autogen_agentchat.teams import RoundRobinGroupChat  # noqa: E402
from autogen_ext.models.openai import OpenAIChatCompletionClient  # noqa: E402


def client():
    return OpenAIChatCompletionClient(
        model="mock", base_url=BASE, api_key="unused",
        model_info={"vision": False, "function_calling": True,
                    "json_output": False, "family": "unknown",
                    "structured_output": False})


def lookup(task: str) -> str:
    """Look a fact up."""
    return "partial result; keep going"


def agent(name: str, limit: int) -> AssistantAgent:
    return AssistantAgent(name=name, model_client=client(), tools=[lookup],
                          max_tool_iterations=limit,
                          system_message="You never conclude on your own.")


def single(L):
    return agent("solo", L)


def flat_team(L):
    return RoundRobinGroupChat([agent("a1", L), agent("a2", L)], max_turns=L)


def nested_team(L):
    inner = RoundRobinGroupChat([agent("i1", L), agent("i2", L)], max_turns=L)
    som = SocietyOfMindAgent("society", team=inner, model_client=client())
    return RoundRobinGroupChat([som], max_turns=L)


CASES = [
    ("single agent            ", single),
    ("flat team               ", flat_team),
    ("team nested in a team   ", nested_team),
]


async def main() -> None:
    # One loop for the whole sweep. asyncio.run() per case tears the loop
    # down while httpx is still closing its pooled connections, which prints
    # a shutdown traceback that has nothing to do with the measurement.
    for L in (3, 5):
        for label, build in CASES:
            mini_proxy.reset(cap=CAP)
            t0 = time.time()
            try:
                await build(L).run(task="Research a topic.")
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


asyncio.run(main())
