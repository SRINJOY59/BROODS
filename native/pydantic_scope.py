"""Can the scope patch be retrofitted into a real framework?

The positive control of the paper (baseline_scoped) is our own 40-line loop
with a shared budget threaded through the delegation tree. The obvious
objection is that we implemented the abstraction in code we control, which
proves it is implementable but not that any shipped framework could carry it.

This tests the retrofit directly. PydanticAI's `Agent.run_sync` accepts BOTH
`usage_limits` (the bound) and `usage` (the accumulator the bound is checked
against). The matrix measures the default: every nested run allocates a fresh
RunUsage, so `request_limit` binds each agent. Here the SAME RunUsage object is
threaded into every descendant, so `request_limit` binds the whole subtree.

Three arms at each depth, all with the same declared limit L:

  per_agent  -- a fresh UsageLimits per agent  (the matrix condition)
  shared     -- one RunUsage threaded through  (the retrofit)
  none       -- no bound at all                (the runaway, for scale)

Calls are counted at the socket by mini_proxy, not self-reported.

Run:  .venvs/pydantic_ai/Scripts/python broods/native/pydantic_scope.py
"""
from __future__ import annotations

import asyncio
import concurrent.futures
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
CAP = 400
MODEL = "mock"

mini_proxy.serve(PORT)
time.sleep(0.3)

# A stale proxy left listening on PORT by an earlier run would answer these
# requests while THIS process's counter stayed at zero -- silently reporting
# a perfectly contained tree. That is precisely the false-pass failure mode
# of Appendix A, so assert the socket we count is the socket being used.
_probe = __import__("urllib.request").request
try:
    with _probe.urlopen(f"http://127.0.0.1:{PORT}/__stats", timeout=3) as _r:
        if json.loads(_r.read())["calls"] != mini_proxy.STATE["calls"]:
            raise SystemExit(
                f"port {PORT} is served by a different process; "
                f"kill it before running (counts would be wrong)")
except OSError as _e:
    raise SystemExit(f"mini_proxy not reachable on {PORT}: {_e}")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.openai import OpenAIChatModel  # noqa: E402
from pydantic_ai.providers.openai import OpenAIProvider  # noqa: E402
from pydantic_ai.usage import RunUsage, UsageLimits  # noqa: E402

def new_model():
    """A fresh model, and so a fresh httpx client, per agent.

    Not incidental. A delegating tool runs its child on another event loop
    (see _drive), and an AsyncClient is bound to the loop that created it, so
    sharing one model across the tree deadlocks the child's first request.
    The production adapter constructs a model per agent for the same reason.
    """
    return OpenAIChatModel(
        MODEL, provider=OpenAIProvider(base_url=BASE, api_key="unused")
    )

SYSTEM = ("You are a research agent. Break the task into subtasks and "
          "delegate each one using the delegate tool before answering.")


def _drive(coro):
    """Run a coroutine even though a loop is already spinning in this thread.

    Identical to the production adapter: PydanticAI raises UserError on a
    nested synchronous run, so a delegating tool must hand the child to a
    fresh loop on another thread. This is the executor boundary that a
    scope object has to survive, and testing the retrofit without it would
    test a topology the matrix never ran.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def build(depth: int, limit: int, shared: RunUsage | None,
          limits: UsageLimits | None):
    """One agent whose `delegate` tool constructs and runs a child.

    Mirrors the spawn probe's tree exactly: same topology, same per-node
    declaration. The ONLY variable is whether `usage` is shared downward.
    """
    agent = Agent(new_model(), system_prompt=SYSTEM)

    if depth == 0:
        # A leaf still has a tool, exactly as in the spawn probe: without one
        # the injector has nothing to call and the agent stops after a single
        # reply, which would measure the mock rather than the bound.
        @agent.tool_plain
        def delegate(task: str) -> str:  # noqa: ANN202
            return "leaf complete"
    else:
        @agent.tool_plain
        def delegate(task: str) -> str:  # noqa: ANN202,F811
            child = build(depth - 1, limit, shared, limits)
            try:
                # The retrofit is the `usage=` argument. With `usage=shared`
                # the child's requests land in the parent's accumulator, so
                # the declared request_limit is a budget for the subtree.
                _drive(child.run(task, usage_limits=limits, usage=shared))
            except Exception as e:  # noqa: BLE001
                # A conscientious tool catches child failures. That habit is
                # what lets a parent continue after a child was correctly
                # stopped -- and is why per-agent bounds do not compose.
                return f"child stopped: {type(e).__name__}"
            return "child complete"

    return agent


def run_arm(arm: str, depth: int, limit: int) -> dict:
    mini_proxy.reset(cap=CAP)
    if arm == "none":
        # NOT unbounded: UsageLimits carries a default request_limit, so the
        # honest "no declared bound" arm is that default, reported as such.
        limits, shared = None, None
    elif arm == "per_agent":
        limits, shared = UsageLimits(request_limit=limit), None
    elif arm == "shared":
        limits, shared = UsageLimits(request_limit=limit), RunUsage()
    else:
        raise ValueError(arm)

    root = build(depth, limit, shared, limits)
    t0 = time.time()
    try:
        _drive(root.run("Research a topic.", usage_limits=limits, usage=shared))
        outcome = "COMPLETED"
    except Exception as e:  # noqa: BLE001
        outcome = type(e).__name__
    return {
        "arm": arm, "depth": depth, "declared_limit": limit,
        "llm_calls": mini_proxy.STATE["calls"],
        "capped": mini_proxy.STATE["capped"],
        "outcome": outcome,
        "duration_s": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    # Depths 0--1 only. Each delegation needs its own thread and event loop
    # (see _drive), so a depth-2 per-agent tree nests 31 pools and does not
    # terminate in reasonable time -- itself a finding, reported rather than
    # worked around. Depth 0 vs 1 already separates enforcement from
    # composition, which is the whole question here.
    depths = [int(x) for x in (sys.argv[1:] or ["0", "1"])]
    for depth in depths:
        for arm in ("per_agent", "shared"):
            for trial in range(3):     # determinism, as for the native table
                r = run_arm(arm, depth, 5)
                r["trial"] = trial
                print(json.dumps(r), flush=True)
