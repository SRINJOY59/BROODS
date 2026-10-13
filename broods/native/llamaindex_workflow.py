"""Does LlamaIndex's AgentWorkflow introduce a call bound its agents lack?

\sys{llamaindex} scores L0 on `spawn`: no agent class exposes an iteration
bound. AgentWorkflow is its native multi-agent construct, routing between
agents by handoff. The question is whether the construct supplies the bound the
agents do not -- if nothing stops the run but our own backstop, it does not.

Run:  .venvs/llamaindex/Scripts/python broods/native/llamaindex_workflow.py
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

PORT = 8147
BASE = f"http://127.0.0.1:{PORT}/v1"
CAP = 60          # lower than the others: nothing here is expected to stop

mini_proxy.serve(PORT)
time.sleep(0.3)

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent  # noqa: E402
from llama_index.core.tools import FunctionTool  # noqa: E402
from llama_index.llms.openai_like import OpenAILike  # noqa: E402


def llm():
    return OpenAILike(model="mock", api_base=BASE, api_key="unused",
                      is_chat_model=True, is_function_calling_model=True,
                      context_window=8192)


def lookup(task: str) -> str:
    """Look a fact up."""
    return "partial result; keep going"


TOOL = FunctionTool.from_defaults(fn=lookup, name="lookup",
                                  description="Look a fact up.")


def agent(name: str, others: list[str]):
    return FunctionAgent(name=name, description="research agent", llm=llm(),
                         tools=[TOOL], can_handoff_to=others,
                         system_prompt="You never conclude on your own.")


def solo():
    return agent("solo", [])


def workflow():
    a = agent("alpha", ["beta"])
    b = agent("beta", ["alpha"])
    return AgentWorkflow(agents=[a, b], root_agent="alpha", timeout=90)


async def main() -> None:
    # The bound is a run() argument (max_iterations, default 20), not a
    # constructor one, which is why a signature sweep scored it absent.
    # None leaves the default in force; 3 and 5 match the other constructs.
    for limit in (None, 3, 5):
        for label, build, prefer in [("single agent", solo, ""),
                                     ("AgentWorkflow", workflow, "handoff")]:
            for trial in range(3):
                mini_proxy.reset(cap=CAP, prefer=prefer)
                kw = {} if limit is None else {"max_iterations": limit}
                t0 = time.time()
                try:
                    await build().run(user_msg="Research a topic.", **kw)
                    outcome = "COMPLETED"
                except Exception as e:  # noqa: BLE001
                    outcome = type(e).__name__
                tools = [name for _seq, name in mini_proxy.STATE["log"]]
                print(json.dumps({
                    "case": label, "trial": trial,
                    "outcome": outcome,
                    "llm_calls": mini_proxy.STATE["calls"],
                    # A handoff construct that never hands off measures a
                    # single agent, so count the transfers that happened.
                    "handoffs": sum("handoff" in (n or "") for n in tools),
                    "declared_limit": limit,
                    "capped": mini_proxy.STATE["capped"],
                    "duration_s": round(time.time() - t0, 1),
                }), flush=True)


asyncio.run(main())
