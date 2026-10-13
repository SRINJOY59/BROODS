"""Collect every headline number of the extended study from result files.

    python -m broods.experiments.summarize      -> results/SUMMARY.json + .md

Nothing in the abstract or the paper should be typed from memory; this reads
the latest bundle of each experiment and prints the figures they support.
Missing experiments are reported as missing, not filled in.
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"


def latest(pattern: str) -> Path | None:
    xs = sorted(glob.glob(str(R / pattern)))
    return Path(xs[-1]) if xs else None


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def matrix() -> dict:
    p = latest("full/matrix_*.json")
    if not p:
        return {"missing": True}
    d = json.loads(p.read_text())
    v = d["verdicts"]
    fws = list(next(iter(v.values())).keys())
    counts: dict[str, int] = {}
    cells = 0
    for probe, row in v.items():
        for fw, x in row.items():
            counts[x["level"]] = counts.get(x["level"], 0) + 1
            cells += 1
    l4 = [(p_, fw) for p_, row in v.items() for fw, x in row.items() if x["level"] == "L4"]
    return {"bundle": p.name, "commit": d["manifest"].get("git_commit"),
            "dirty": d["manifest"].get("git_dirty"), "trials": d.get("trials"),
            "systems": fws, "cells": cells, "levels": counts, "L4_cells": l4,
            "levels_by_system": {fw: {p_: v[p_][fw]["level"] for p_ in v} for fw in fws}}


def scope() -> dict:
    p = latest("scope/scope_eval_*.json")
    if not p:
        return {"missing": True}
    d = json.loads(p.read_text())["frameworks"]
    out = {"bundle": p.name, "frameworks": {}}
    for fw, Rw in d.items():
        out["frameworks"][fw] = {
            "control_calls": Rw["spawn/control"]["calls"],
            "ctx_only": [Rw["spawn/ctx-only"]["calls"], Rw["spawn/ctx-only"]["level"]],
            "propagate": [Rw["spawn/propagate"]["calls"], Rw["spawn/propagate"]["level"]],
            "accounting_err_pct": Rw.get("accounting/propagate"),
            "token_socket": Rw["token_budget/propagate"]["socket_tokens"],
            "deadline_control_x": Rw["deadline/control"].get("overshoot_ratio"),
            "deadline_scope_x": Rw["deadline/propagate"].get("overshoot_ratio"),
            "orphans_control": Rw["cancel/control"].get("orphan_calls"),
            "orphans_ctx_only": Rw["cancel/ctx-only"].get("orphan_calls"),
            "orphans_scope": Rw["cancel/propagate"].get("orphan_calls"),
        }
    f = out["frameworks"]
    ok = [fw for fw, x in f.items() if x["propagate"][1] == "L4"]
    out["L4_with_propagation"] = ok
    out["ctx_only_fails"] = [fw for fw, x in f.items()
                             if x["propagate"][1] == "L4" and x["ctx_only"][1] != "L4"]
    ds = [x["deadline_scope_x"] for fw, x in f.items() if fw in ok and x["deadline_scope_x"]]
    out["deadline_scope_range"] = [min(ds), max(ds)] if ds else None
    dc = [x["deadline_control_x"] for fw, x in f.items() if x["deadline_control_x"]]
    out["deadline_control_range"] = [min(dc), max(dc)] if dc else None
    return out


def protocols() -> dict:
    p = latest("protocols/protocols_*.json")
    if not p:
        return {"missing": True}
    rows = json.loads(p.read_text())["rows"]
    out: dict = {"bundle": p.name, "cells": {}}
    for proto in sorted({r["proto"] for r in rows}):
        for mode in ("timeout", "cancel"):
            rs = [r for r in rows if r["proto"] == proto and r["mode"] == mode]
            if not rs:
                continue
            h = lambda k, i: [r["hops"][str(i)][k] if str(i) in r["hops"] else r["hops"][i][k]
                              for r in rs]
            out["cells"][f"{proto}/{mode}"] = {
                "n": len(rs),
                "deadline_seen_hop1": all(h("deadline_seen", 1)),
                "deadline_seen_hop2": all(h("deadline_seen", 2)),
                "orphans_hop1": [min(h("orphan_calls", 1)), max(h("orphan_calls", 1))],
                "orphans_hop2": [min(h("orphan_calls", 2)), max(h("orphan_calls", 2))],
                "cancel_delivered_hop2": all(h("cancel_delivered", 2)),
            }
    return out


def branching() -> dict:
    p = R / "live" / "branching_summary.json"
    return json.loads(p.read_text()) if p.exists() else {"missing": True}


def audit() -> dict:
    p = R / "audit" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {"missing": True}


def versions() -> dict:
    p = R / "versions" / "versions.jsonl"
    if not p.exists():
        return {"missing": True}
    import importlib
    spawn = importlib.import_module("broods.probes.spawn.probe").SpawnProbe(None)
    from broods.probes.base import ProbeResult
    rows = []
    for r in jl(p):
        lvl = r.get("level")
        # Re-score with the CURRENT verdict: rows written before the
        # root-only guard existed carried a false L4 (crewai 0.126.0).
        if r.get("status") == "OK" and r.get("spawn"):
            res = [ProbeResult(probe="spawn", framework=r["framework"], depth=s["depth"],
                               outcome=s["outcome"],
                               metrics={"llm_calls": s["calls"], "agents_built": s["agents"],
                                        "declared_limit": 5, "capped": False})
                   for s in r["spawn"] if s.get("calls") is not None]
            if res:
                lvl = spawn.safe_verdict(res)[0]
        rows.append({"framework": r["framework"], "version": r["version"],
                     "released": r["released"], "status": r["status"], "level": lvl,
                     "calls": [s.get("calls") for s in r.get("spawn", [])],
                     "default_bound": r.get("default_bound"),
                     "native": [(c["case"].strip(), c["llm_calls"])
                                for c in r.get("native", []) if isinstance(c, dict)]
                     if isinstance(r.get("native"), list) else None})
    measured = [r for r in rows if r["level"] in ("L0", "L1", "L2", "L3", "L4")]
    return {"rows": rows, "measured": len(measured),
            "any_L4": [f"{r['framework']} {r['version']}" for r in measured
                       if r["level"] == "L4"]}


def newfw() -> dict:
    out = {}
    p = latest("ts/ts_spawn_*.json")
    if p:
        out["typescript"] = {fw: {"level": v["level"],
                                  "calls": [r["metrics"]["llm_calls"] for r in v["rows"]],
                                  "version": v["rows"][0]["version"]}
                             for fw, v in json.loads(p.read_text()).items()}
    nat = {}
    for fw in ("google_adk", "strands", "agno"):
        q = R / "native" / f"new_{fw}.jsonl"
        if q.exists():
            nat[fw] = [(r["case"], r["L"], r["llm_calls"], r["capped"]) for r in jl(q)]
    out["native_constructs"] = nat
    return out


def main() -> None:
    S = {"matrix": matrix(), "taskscope": scope(), "protocols": protocols(),
         "branching": branching(), "audit": audit(), "versions": versions(),
         "new_frameworks": newfw()}
    (R / "SUMMARY.json").write_text(json.dumps(S, indent=2, default=str))
    lines = ["# Extended study: headline numbers (generated)\n"]
    for k, v in S.items():
        lines.append(f"## {k}\n\n```json\n{json.dumps(v, indent=1, default=str)[:6000]}\n```\n")
    (R / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: ("missing" if isinstance(v, dict) and v.get("missing") else "ok")
                      for k, v in S.items()}, indent=1))


if __name__ == "__main__":
    main()
