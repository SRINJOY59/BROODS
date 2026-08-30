"""Does CrewAI's max_iter bound a HIERARCHICAL crew, or only one agent?

The paper measures agent-as-tool delegation, where a bound declared on the
root does not constrain the children the root creates. This tests CrewAI's own
multi-agent construct instead: Process.hierarchical, where a manager agent
delegates to workers through tools CrewAI generates itself. If any framework
tracks the work it started, this is where it would.

Two managers are tried, because they are two different claims:
  manager_llm    -- CrewAI builds the manager; can a bound even be declared?
  manager_agent  -- we build it, so max_iter is declared on every agent

Run:  .venvs/crewai/Scripts/python broods/native/crewai_hier.py
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

PORT = 8143
BASE = f"http://127.0.0.1:{PORT}/v1"
CAP = 200

os.environ["OPENAI_API_KEY"] = "unused"
os.environ["OPENAI_API_BASE"] = BASE
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

mini_proxy.serve(PORT)
time.sleep(0.3)

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402

WORKER_ROLE = "Worker"


def _llm():
    return LLM(model="openai/mock", base_url=BASE, api_key="unused",
               temperature=0)


@tool("lookup")
def lookup(task: str) -> str:
    """Look a fact up."""
    return "partial result; keep going"


def worker(limit: int, n: int = 1) -> Agent:
    return Agent(role=WORKER_ROLE if n == 1 else f"{WORKER_ROLE}{n}",
                 goal="Do the assigned subtask.",
                 backstory="You never conclude on your own.",
                 tools=[lookup], llm=_llm(), max_iter=limit,
                 allow_delegation=False, verbose=False)


def sequential(limit: int) -> Crew:
    a = worker(limit)
    t = Task(description="Research a topic.",
             expected_output="A short note.", agent=a)
    return Crew(agents=[a], tasks=[t], process=Process.sequential,
                verbose=False)


def hierarchical(limit: int, explicit_manager: bool) -> Crew:
    ws = [worker(limit, 1), worker(limit, 2)]
    # In hierarchical mode the task carries no agent: the manager assigns it.
    t = Task(description="Research a topic.", expected_output="A short note.")
    kw = dict(agents=ws, tasks=[t], process=Process.hierarchical, verbose=False)
    if explicit_manager:
        kw["manager_agent"] = Agent(
            role="Manager", goal="Delegate the work and assemble the answer.",
            backstory="You always delegate.", llm=_llm(), max_iter=limit,
            allow_delegation=True, verbose=False)
    else:
        kw["manager_llm"] = _llm()
    return Crew(**kw)


CASES = [
    ("sequential  1 agent      ", lambda L: sequential(L)),
    ("hierarchical manager_llm ", lambda L: hierarchical(L, False)),
    ("hierarchical manager_agent", lambda L: hierarchical(L, True)),
]

for L in (3, 5):
    for label, build in CASES:
        mini_proxy.reset(cap=CAP, coworker=WORKER_ROLE)
        t0 = time.time()
        try:
            crew = build(L)
            crew.kickoff()
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
