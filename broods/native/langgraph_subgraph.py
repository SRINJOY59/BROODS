"""Does LangGraph's recursion_limit propagate into a subgraph?

The paper measures agent-as-tool delegation. This tests the native construct:
a child graph added as a NODE of the parent graph, which is the case where a
framework plausibly does track work it created itself.
"""
import json, pathlib, sys, time, urllib.request, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from typing import TypedDict

from broods.native import mini_proxy
from langgraph.graph import StateGraph, START, END

PORT = 8141
BASE = f"http://127.0.0.1:{PORT}/v1"
mini_proxy.serve(PORT)
time.sleep(0.3)
calls = {"n": 0}

def _chat():
    calls["n"] += 1
    body = json.dumps({"model": "mock", "messages": [{"role": "user", "content": "go"}]}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer x",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()

class S(TypedDict):
    n: int

def child_graph():
    g = StateGraph(S)
    def step(s: S):
        _chat()
        return {"n": s["n"] + 1}
    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_conditional_edges("step", lambda s: "step", {"step": "step"})
    return g.compile()

def parent_with_subgraph():
    sub = child_graph()
    g = StateGraph(S)
    g.add_node("sub", sub)                 # child compiled graph AS A NODE
    g.add_edge(START, "sub")
    g.add_conditional_edges("sub", lambda s: "sub", {"sub": "sub"})
    return g.compile()

cases = []
for L in (3, 5, 8):
    cases.append((f"child alone      L={L}", child_graph(),          {"recursion_limit": L}))
    cases.append((f"parent+subgraph  L={L}", parent_with_subgraph(), {"recursion_limit": L}))
for label, graph, kw in cases:
    calls["n"] = 0
    try:
        graph.invoke({"n": 0}, config=kw)
        outcome = "COMPLETED"
    except Exception as e:
        outcome = type(e).__name__
    print(json.dumps({"case": label, "outcome": outcome,
                      "llm_calls": calls["n"], "declared_limit": kw["recursion_limit"]}),
          flush=True)
