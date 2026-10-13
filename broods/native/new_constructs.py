"""Native multi-agent constructs of the three frameworks added later
(Google ADK, Strands, Agno), measured exactly like Table `native`: every
agent declares L, the endpoint never lets anything stop voluntarily, and
calls are counted at the socket.

    <venv python> broods/native/new_constructs.py {google_adk|strands|agno}

Each framework ships two ways for one agent to use another, and the pair is
the point: one runs the child inside the caller's run, the other starts a
new run.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
os.environ.setdefault("AGNO_TELEMETRY", "false")
os.environ.setdefault("LITELLM_LOG", "ERROR")

from broods.native import mini_proxy  # noqa: E402

PORT = 8146
BASE = f"http://127.0.0.1:{PORT}/v1"
CAP = 250
mini_proxy.serve(PORT)
time.sleep(0.3)


def emit(case: str, L: int, fn) -> None:
    prefer, coworker = "", ""
    if isinstance(fn, tuple):
        fn, prefer, *rest = fn
        coworker = rest[0] if rest else ""
    mini_proxy.reset(cap=CAP, prefer=prefer, coworker=coworker)
    t0 = time.time()
    try:
        fn(L)
        outcome = "COMPLETED"
    except Exception as e:  # noqa: BLE001
        outcome = type(e).__name__
    print(json.dumps({"case": case, "L": L, "outcome": outcome,
                      "llm_calls": mini_proxy.STATE["calls"],
                      "capped": mini_proxy.STATE["capped"],
                      "duration_s": round(time.time() - t0, 1)}), flush=True)


def lookup(task: str) -> str:
    """Look a fact up.

    Args:
        task: what to look up
    """
    return "partial result; keep going"


# ------------------------------------------------------------------ ADK
def adk_cases():
    from google.adk.agents import LlmAgent
    from google.adk.agents.run_config import RunConfig
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    from google.adk.tools.agent_tool import AgentTool
    from google.genai import types

    def model():
        return LiteLlm(model="openai/mock", api_base=BASE, api_key="unused")

    def run(agent, L):
        async def go():
            r = InMemoryRunner(agent=agent, app_name="n")
            s = await r.session_service.create_session(app_name="n", user_id="u")
            msg = types.Content(role="user", parts=[types.Part(text="Research.")])
            async for _ in r.run_async(user_id="u", session_id=s.id, new_message=msg,
                                       run_config=RunConfig(max_llm_calls=L)):
                pass
        asyncio.run(go())

    def single(L):
        run(LlmAgent(name="solo", model=model(), instruction="Never conclude.",
                     tools=[lookup]), L)

    def agent_tool(L):
        child = LlmAgent(name="child", model=model(), instruction="Never conclude.",
                         tools=[lookup])
        run(LlmAgent(name="root", model=model(), instruction="Always call child.",
                     tools=[AgentTool(child)]), L)

    def transfer(L):
        a = LlmAgent(name="alpha", model=model(), instruction="Transfer to beta.",
                     tools=[lookup])
        b = LlmAgent(name="beta", model=model(), instruction="Transfer to alpha.",
                     tools=[lookup])
        root = LlmAgent(name="root", model=model(), instruction="Transfer.",
                        sub_agents=[a, b])
        run(root, L)

    return [("single agent", single), ("AgentTool (nested run)", (agent_tool, "child")),
            ("sub_agents transfer", (transfer, "transfer"))]


# -------------------------------------------------------------- Strands
def strands_cases():
    from strands import Agent, tool
    from strands.models.openai import OpenAIModel
    from strands.multiagent import Swarm

    lk = tool(lookup)

    def model():
        return OpenAIModel(client_args={"base_url": BASE, "api_key": "unused"},
                           model_id="mock")

    def single(L):
        Agent(model=model(), tools=[lk], callback_handler=None)(
            "Research.", limits={"turns": L})

    def as_tool(L):
        child = Agent(model=model(), tools=[lk], callback_handler=None,
                      name="child", description="Delegate a subtask.")
        Agent(model=model(), tools=[child.as_tool()], callback_handler=None)(
            "Research.", limits={"turns": L})

    def swarm(L):
        a = Agent(model=model(), tools=[lk], callback_handler=None, name="alpha",
                  description="researcher")
        b = Agent(model=model(), tools=[lk], callback_handler=None, name="beta",
                  description="researcher")
        Swarm([a, b], max_handoffs=L, max_iterations=L)("Research.")

    return [("single agent", single), ("agent.as_tool() (nested run)", (as_tool, "child")),
            ("Swarm(max_handoffs=L, max_iterations=L)", (swarm, "handoff", "alpha|beta"))]


# ------------------------------------------------------------------ Agno
def agno_cases():
    from agno.agent import Agent
    from agno.models.openai.like import OpenAILike
    from agno.team import Team

    def model():
        return OpenAILike(id="mock", base_url=BASE, api_key="unused")

    def single(L):
        Agent(model=model(), tools=[lookup], tool_call_limit=L,
              telemetry=False).run("Research.")

    def team(L):
        m = Agent(name="member", model=model(), tools=[lookup], tool_call_limit=L,
                  telemetry=False)
        Team(members=[m], model=model(), max_iterations=L, tool_call_limit=L,
             telemetry=False).run("Research.")

    return [("single agent (tool_call_limit)", single),
            ("Team(max_iterations=L)", (team, "member"))]


if __name__ == "__main__":
    which = sys.argv[1]
    cases = {"google_adk": adk_cases, "strands": strands_cases,
             "agno": agno_cases}[which]()
    if len(sys.argv) > 2:                    # run only cases whose label matches
        cases = [c for c in cases if sys.argv[2].lower() in c[0].lower()]
    for L in (3, 5):
        for label, fn in cases:
            emit(label, L, fn)
