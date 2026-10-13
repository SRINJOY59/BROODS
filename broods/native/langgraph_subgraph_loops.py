"""Does LangGraph's recursion_limit compose when parent AND child both loop?

langgraph_subgraph.py adds a never-terminating child graph as a node of its
parent. There the child exhausts recursion_limit and raises
GraphRecursionError, which aborts the parent before it can loop again. That
run cannot tell a shared budget from a child exception that kills its caller:
both give "composed == alone".

This version separates the two. The child finishes NORMALLY after K model
calls, well inside its own limit, so no exception reaches the parent. The
parent then loops back into the child with no exit of its own. If the step
budget is shared across the subgraph boundary, total calls stay near L. If
each graph counts only its own supersteps, every parent step buys K more calls
and the total multiplies.

Calls are counted at the socket by mini_proxy, not self-reported.

Run:  .venvs/langgraph/Scripts/python broods/native/langgraph_subgraph_loops.py
"""
import json
import pathlib
import sys
import time
import urllib.request
import warnings
from datetime import datetime, timezone
from typing import TypedDict

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from broods.native import mini_proxy  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

PORT = 8143
BASE = f"http://127.0.0.1:{PORT}/v1"
K = 2            # model calls per child invocation; below every limit tested
LIMITS = (3, 5, 8)
TRIALS = 3


class S(TypedDict):
    i: int       # calls made so far; the child stops at each multiple of K


def _chat() -> None:
    body = json.dumps({"model": "mock",
                       "messages": [{"role": "user", "content": "go"}]}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer x",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def child_graph():
    """Loops on itself, one model call per step, and ends after K steps."""
    g = StateGraph(S)

    def step(s: S):
        _chat()
        return {"i": s["i"] + 1}

    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_conditional_edges("step",
                            lambda s: END if s["i"] % K == 0 else "step",
                            {"step": "step", END: END})
    return g.compile()


def parent_looping_over_child():
    """The child graph as a node, and a parent that re-enters it forever."""
    g = StateGraph(S)
    g.add_node("sub", child_graph())
    g.add_edge(START, "sub")
    g.add_conditional_edges("sub", lambda s: "sub", {"sub": "sub"})
    return g.compile()


def run(graph, limit: int) -> dict:
    mini_proxy.reset(cap=250)
    try:
        graph.invoke({"i": 0}, config={"recursion_limit": limit})
        outcome = "COMPLETED"
    except Exception as e:  # noqa: BLE001 -- the stop reason is the datum
        outcome = type(e).__name__
    return {"llm_calls": mini_proxy.STATE["calls"],
            "capped": mini_proxy.STATE["capped"], "outcome": outcome}


def main() -> None:
    mini_proxy.serve(PORT)
    time.sleep(0.3)
    rows = []
    for L in LIMITS:
        for case, build in (("child alone", child_graph),
                            ("parent loops over child", parent_looping_over_child)):
            trials = [run(build(), L) for _ in range(TRIALS)]
            row = {"case": case, "declared_limit": L, "child_calls_per_invocation": K,
                   "llm_calls": [t["llm_calls"] for t in trials],
                   "outcomes": sorted({t["outcome"] for t in trials}),
                   "capped": any(t["capped"] for t in trials)}
            rows.append(row)
            print(json.dumps(row), flush=True)

    out = ROOT / "results" / "native"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    import importlib.metadata as md
    path = out / f"langgraph_subgraph_loops_{stamp}.json"
    path.write_text(json.dumps({"utc": stamp, "langgraph": md.version("langgraph"),
                                "K": K, "trials": TRIALS, "rows": rows}, indent=2))
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
