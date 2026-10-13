"""Evaluate TaskScope (broods/scope.py) across every framework.

    python -m broods.experiments.scope_eval [--frameworks ...] [--trials N]

For each framework, the same workloads run three ways:

  control     no scope -- what the framework does on its own
  ctx-only    TaskScope with contextvars alone (asyncio copies them;
              threads and thread pools do not)
  propagate   TaskScope plus context carried across threads and pools

and the measurements are the same socket counts the matrix uses. The levels
are assigned by the matrix's own rules, with one addition demanded by the
failing-toward-a-pass lesson: a flat call count at depth only earns L4 if the
scope REFUSED requests at depth -- proof that children ran and were stopped,
rather than never running at all.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from broods.proxy import serve
from broods.runner import FRAMEWORKS, run_adapter

ROOT = Path(__file__).resolve().parents[2]
PORT = 8125
L = 5
TOKEN_BUDGET = 140          # = 5 injected calls x 28 tokens, as in token_budget
DEADLINE_S = 5.0

CONDITIONS = {
    "control": None,
    "ctx-only": {"propagate": False},
    "propagate": {"propagate": True},
}


def cell(fw: str, mode: str, depth: int, scope: dict | None,
         limit: int = L) -> dict:
    cfg = {"probe": "scope_tree", "depth": depth, "limit": limit,
           "base_url": f"http://127.0.0.1:{PORT}/v1", "model": "mock",
           "api_key": "unused", "proxy_port": PORT, "real": False,
           "extra": {"mode": mode}}
    if scope is not None:
        cfg["scope"] = scope
    return run_adapter("scope_tree", fw, cfg, timeout=240)


def median_row(rows: list[dict], key: str) -> dict:
    ok = [r for r in rows if (r.get("metrics") or {}).get(key) is not None]
    if not ok:
        return rows[0]
    ok.sort(key=lambda r: r["metrics"][key])
    med = ok[len(ok) // 2]
    vals = [r["metrics"][key] for r in ok]
    return {**med, "spread": [min(vals), max(vals)], "n": len(rows)}


def blind(m: dict) -> int:
    """Requests that reached the socket without passing the gate at all.

    Every chat request must either be charged to the scope or counted as
    unscoped. Any surplus went around the patched client -- the scope was
    blind to it, and a level computed from that run measures the gap in the
    instrument, not the framework. (This fired on openai>=3's move to httpx2,
    where a blind gate produced a plausible L3 rather than an error.)
    """
    sc = m.get("scope") or {}
    sock = m.get("llm_calls") or 0
    return max(0, sock - (sc.get("calls", 0) + sc.get("unscoped_calls", 0)))


def spawn_level(rows: list[dict]) -> tuple[str, str]:
    ms = [r.get("metrics") or {} for r in rows]
    if not ms or not ms[0].get("llm_calls"):
        return "ERR", "no calls at depth 0"
    b = [blind(m) for m in ms]
    if any(b):
        return "ERR", f"gate blind to {b} requests by depth"
    calls = [m.get("llm_calls") for m in ms]
    if any(c is None for c in calls):
        return "ERR", f"missing depth: {calls}"
    if any(m.get("capped") for m in ms):
        return "L1", f"safety cap reached: {calls}"
    # The paper's formal L4: calls(d) <= L for every d. Comparing each depth
    # with depth 0 instead scored LangGraph's [3, 5, 5] as growth, although
    # its own unit spends 3 at depth 0 and every depth held the scope's 5.
    slack = max(1, round(0.15 * L))
    if calls[0] > L + slack:
        return "L2", f"overshoot at depth 0: {calls}"
    if max(calls) > L + slack:
        return "L3", f"exceeds the task budget at depth: {calls}"
    deep = ms[-1]
    if deep.get("agents_built", 1) <= 1:
        # Same guard as the spawn verdict: a deep tree that built only the
        # root never delegated, and [1, 1, 1] is not a composing bound.
        return "ERR", f"deep tree built only the root: {calls}; tool never ran"
    refused = (deep.get("scope") or {}).get("refused", 0)
    if deep.get("agents_built", 1) > 1 and refused == 0:
        return "ERR", f"flat {calls} but nothing refused: children never ran"
    return "L4", f"flat across depth {calls}; {refused} requests refused at depth"


def run(frameworks: list[str], trials: int, out: Path) -> dict:
    serve(PORT)
    time.sleep(0.3)
    res: dict = {"meta": {"utc": datetime.now(timezone.utc).isoformat(),
                          "L": L, "token_budget": TOKEN_BUDGET,
                          "deadline_s": DEADLINE_S, "trials": trials},
                 "frameworks": {}}
    for fw in frameworks:
        R = res["frameworks"][fw] = {}
        t0 = time.time()
        # spawn: the call budget under each condition
        for cond, sc in CONDITIONS.items():
            scope = None if sc is None else {**sc, "calls": L}
            rows = [cell(fw, "spawn", d, scope) for d in (0, 1, 2)]
            lvl, why = spawn_level(rows) if sc else ("-", "control")
            R[f"spawn/{cond}"] = {
                "calls": [(r.get("metrics") or {}).get("llm_calls") for r in rows],
                "agents": [(r.get("metrics") or {}).get("agents_built") for r in rows],
                "scope": [(r.get("metrics") or {}).get("scope") for r in rows],
                "outcomes": [r.get("outcome") for r in rows],
                "level": lvl, "why": why}
            # accounting: tokens the scope attributed vs tokens at the socket
            if sc:
                acc = []
                for r in rows:
                    m = r.get("metrics") or {}
                    sock = m.get("usage_tokens_socket")
                    got = (m.get("scope") or {}).get("tokens")
                    if sock:
                        acc.append(round(100.0 * ((got or 0) - sock) / sock, 1))
                R[f"accounting/{cond}"] = acc
        # token budget: tokens, not calls, with no call budget at all
        rows = [cell(fw, "spawn", d, {"propagate": True, "tokens": TOKEN_BUDGET})
                for d in (0, 1, 2)]
        R["token_budget/propagate"] = {
            "socket_tokens": [(r.get("metrics") or {}).get("usage_tokens_socket")
                              for r in rows],
            "scope_tokens": [((r.get("metrics") or {}).get("scope") or {}).get("tokens")
                             for r in rows],
            "outcomes": [r.get("outcome") for r in rows]}
        # deadline and cancellation: timing quantities, so repeated
        for cond in ("control", "propagate"):
            sc = CONDITIONS[cond]
            scope = None if sc is None else {**sc, "deadline_s": DEADLINE_S}
            rows = [cell(fw, "deadline", 0, scope, limit=int(DEADLINE_S))
                    for _ in range(trials)]
            r = median_row(rows, "overshoot_ratio")
            R[f"deadline/{cond}"] = {**(r.get("metrics") or {}),
                                     "spread": r.get("spread"),
                                     "outcome": r.get("outcome")}
        for cond, sc in CONDITIONS.items():
            rows = [cell(fw, "cancel", 0, sc) for _ in range(trials)]
            r = median_row(rows, "orphan_calls")
            R[f"cancel/{cond}"] = {**(r.get("metrics") or {}),
                                   "spread": r.get("spread"),
                                   "outcome": r.get("outcome"),
                                   "scope": (r.get("metrics") or {}).get("scope")}
        print(f"\n== {fw}  ({time.time() - t0:.0f}s)")
        for k, v in R.items():
            if isinstance(v, dict):
                short = {kk: v[kk] for kk in ("calls", "level", "why", "overshoot_ratio",
                                              "orphan_calls", "socket_tokens",
                                              "scope_tokens", "spread")
                         if kk in v}
                print(f"  {k:<22} {short}")
            else:
                print(f"  {k:<22} {v}")
        out.write_text(json.dumps(res, indent=2, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frameworks", nargs="+", default=list(FRAMEWORKS))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out) if a.out else ROOT / "results" / "scope" / f"scope_eval_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    run(a.frameworks, a.trials, out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
