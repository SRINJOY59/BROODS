"""Does smolagents' max_steps bound a manager plus its managed_agents?

`managed_agents` is smolagents' native hierarchy: a manager agent is given
other agents and calls them by name. Each agent carries its own max_steps.

Run:  .venvs/smolagents/Scripts/python broods/native/smolagents_managed.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from broods.native import mini_proxy  # noqa: E402

PORT = 8146
BASE = f"http://127.0.0.1:{PORT}/v1"
CAP = 200

mini_proxy.serve(PORT)
time.sleep(0.3)

from smolagents import OpenAIServerModel, ToolCallingAgent, tool  # noqa: E402


def model():
    return OpenAIServerModel(model_id="mock", api_base=BASE, api_key="unused")


@tool
def lookup(task: str) -> str:
    """Look a fact up.

    Args:
        task: what to look up.
    """
    return "partial result; keep going"


def solo(L: int):
    return ToolCallingAgent(tools=[lookup], model=model(), max_steps=L)


def managed(L: int):
    worker = ToolCallingAgent(tools=[lookup], model=model(), max_steps=L,
                              name="worker",
                              description="Does a subtask and reports back.")
    return ToolCallingAgent(tools=[], model=model(), max_steps=L,
                            managed_agents=[worker])


CASES = [("single agent        ", solo, ""),
         ("manager + managed   ", managed, "worker")]

for L in (3, 5):
    for label, build, prefer in CASES:
        mini_proxy.reset(cap=CAP, prefer=prefer)
        t0 = time.time()
        try:
            build(L).run("Research a topic.")
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
