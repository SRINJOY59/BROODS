"""Print the conformance matrix from one or more run_all bundles.

Levels are recomputed from each bundle's raw rows with the current verdict
rules, never copied from the verdicts stored in the bundle. A later bundle
replaces every (probe, system) cell it contains, so a targeted re-run can be
layered over a full matrix.

    python -m broods.table results/full/matrix_A.json
    python -m broods.table A.json B.json C.json --latex rows.tex
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from broods.probes import REGISTRY
from broods.probes.base import ProbeResult

SYSTEMS = ["baseline", "langgraph", "crewai", "autogen", "pydantic_ai",
           "openai_agents", "smolagents", "llamaindex"]
NAMES = {"baseline": "baseline", "langgraph": "LangGraph", "crewai": "CrewAI",
         "autogen": "AutoGen", "pydantic_ai": "PydanticAI",
         "openai_agents": "OpenAI Agents", "smolagents": "smolagents",
         "llamaindex": "LlamaIndex"}
PROBES = ["spawn", "deadline", "token_budget", "context_bound", "concurrency",
          "rate_limit", "cancellation", "fault_isolation", "state_isolation",
          "accounting", "reclamation"]
LATEX = {"L0": r"\lz", "L1": r"\lo", "L2": r"\lt", "L3": r"\lth", "L4": r"\lf",
         "N/A": r"\na", "ERR": r"\err"}
FIELDS = set(ProbeResult.__dataclass_fields__)


def collect(bundles: list[Path]) -> dict[tuple[str, str], list[dict]]:
    """Rows per (probe, system), later bundles replacing earlier ones."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for path in bundles:
        fresh: dict[tuple[str, str], list[dict]] = {}
        for r in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            fresh.setdefault((r["probe"], r["framework"]), []).append(r)
        cells.update(fresh)
    return cells


def score(rows: list[dict], probe: str) -> tuple[str, str]:
    rows = sorted(rows, key=lambda r: r["depth"])
    typed = [ProbeResult(**{k: v for k, v in r.items() if k in FIELDS}) for r in rows]
    return REGISTRY[probe](None).safe_verdict(typed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundles", nargs="+", type=Path)
    ap.add_argument("--latex", type=Path, help="also write LaTeX table rows here")
    ap.add_argument("--why", action="store_true", help="print each cell's rationale")
    a = ap.parse_args()

    cells = collect(a.bundles)
    systems = [s for s in SYSTEMS if any((p, s) in cells for p in PROBES)]
    level = {(p, s): score(cells[(p, s)], p) if (p, s) in cells else ("--", "not run")
             for p in PROBES for s in systems}

    print("| system | " + " | ".join(PROBES) + " |")
    print("|---" * (len(PROBES) + 1) + "|")
    for s in systems:
        print(f"| {NAMES[s]} | " + " | ".join(level[(p, s)][0] for p in PROBES) + " |")
    counts = Counter(level[(p, s)][0] for p in PROBES for s in systems)
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))

    if a.why:
        for (p, s), (lv, why) in sorted(level.items()):
            print(f"{p:<16}{s:<15}{lv:<5}{why}")

    if a.latex:
        out = []
        for s in systems:
            row = " & ".join(LATEX.get(level[(p, s)][0], "--") for p in PROBES)
            lz = sum(level[(p, s)][0] == "L0" for p in PROBES)
            out.append(f"{NAMES[s]} & {row} & {lz} \\\\")
        a.latex.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"\nwrote {a.latex}")


if __name__ == "__main__":
    main()
