"""Fit and validate a growth model over collected spawn-probe results.

Turns a table of measurements into a claim that can be falsified: if
aggregate cost really is governed by delegation structure rather than by
any framework's implementation, then one two-parameter model should fit
every framework, and its fitted branch factor should be predictable from
the declared limit and the framework's counting unit.

Model
-----
A tree where each agent makes `a` calls and spawns `b` children has

    agents(d) = (b^(d+1) - 1) / (b - 1)
    calls(d)  = a * agents(d)

so `a` is read off depth 0 and `b` from the agent counts. Both are
recovered from data, not assumed; the residual column is what says whether
the model deserves to be believed.

    python -m broods.analyze
    python -m broods.analyze --holdout      # fit on d<=1, predict d=2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(results_dir: Path, probe: str = "spawn",
         bundle: Path | None = None) -> list[dict]:
    """Collect spawn cells, either from a run_all bundle or loose per-probe files.

    Prefer the bundle: it is one revision's worth of results, whereas the loose
    files accumulate across every version of the code written during
    development and silently mix pre-fix with post-fix numbers.
    """
    rows = []
    if bundle is not None:
        d = json.loads(bundle.read_text())
        for r in d["rows"]:
            if r.get("probe") != probe:
                continue
            m = r.get("metrics") or {}
            if not r.get("supported") or "llm_calls" not in m or m.get("capped"):
                continue
            rows.append({"framework": r["framework"], "depth": r["depth"],
                         "limit": m["declared_limit"], "calls": m["llm_calls"],
                         "agents": m.get("agents_built", 0),
                         "tokens": m.get("prompt_tokens_est", 0)})
        return rows

    for f in sorted(results_dir.glob(f"{probe}_*.json")):
        d = json.loads(f.read_text())
        # Only deterministic fault-injected runs are comparable. Real-model
        # runs let the model decide whether to delegate, so their call
        # counts measure model behaviour, not framework containment.
        # model=="mock" is recorded in every results file, including those
        # written before an explicit scenario field existed.
        if d.get("real") or d.get("model") != "mock":
            continue
        for r in d["rows"]:
            m = r.get("metrics") or {}
            if not r.get("supported") or "llm_calls" not in m or m.get("capped"):
                continue
            rows.append({"framework": r["framework"], "depth": r["depth"],
                         "limit": m["declared_limit"], "calls": m["llm_calls"],
                         "agents": m.get("agents_built", 0),
                         "tokens": m.get("prompt_tokens_est", 0)})
    return rows


def geometric_agents(b: float, d: int) -> float:
    """Nodes in a complete b-ary tree of depth d."""
    if abs(b - 1.0) < 1e-9:
        return d + 1.0
    return (b ** (d + 1) - 1.0) / (b - 1.0)


def fit_branch(rows: list[dict]) -> float | None:
    """Recover b from consecutive agent counts (a_{d+1}-1)/a_d -> b."""
    by_d = {r["depth"]: r["agents"] for r in rows if r["agents"]}
    ratios = [(by_d[d + 1] - 1) / by_d[d]
              for d in sorted(by_d) if d + 1 in by_d and by_d[d]]
    return sum(ratios) / len(ratios) if ratios else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--bundle", default=None,
                    help="a results/full/matrix_*.json from run_all; "
                        "'latest' picks the newest")
    ap.add_argument("--holdout", action="store_true",
                    help="fit on shallow depths only, then predict the deepest")
    a = ap.parse_args()

    bundle = None
    if a.bundle:
        if a.bundle == "latest":
            found = sorted((ROOT / "results" / "full").glob("matrix_*.json"))
            if not found:
                raise SystemExit("no run_all bundle found")
            bundle = found[-1]
        else:
            bundle = Path(a.bundle)
        print(f"source: {bundle.name}\n")
    else:
        print("source: loose per-probe files (mixed revisions -- prefer "
              "--bundle latest)\n")

    rows = load(Path(a.results), bundle=bundle)
    if not rows:
        raise SystemExit("no synthetic spawn results found")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["framework"], r["limit"])].append(r)

    print(f"{'framework':<11}{'limit':>6}{'a':>6}{'b':>7}{'d':>3}"
          f"{'observed':>10}{'predicted':>11}{'err%':>7}")
    print("-" * 61)

    errs, held = [], []
    for (fw, lim), g in sorted(groups.items()):
        g.sort(key=lambda r: r["depth"])
        if len(g) < 2:
            continue
        fit_rows = [r for r in g if r["depth"] <= max(x["depth"] for x in g) - 1] \
            if a.holdout else g
        b = fit_branch(fit_rows) or fit_branch(g)
        base = next((r for r in fit_rows if r["depth"] == 0), None)
        if b is None or base is None:
            continue
        a_calls = base["calls"]                      # calls per agent, from d=0

        for r in g:
            pred = a_calls * geometric_agents(b, r["depth"])
            err = 100.0 * (r["calls"] - pred) / r["calls"] if r["calls"] else 0.0
            is_held = a.holdout and r["depth"] > max(x["depth"] for x in fit_rows)
            (held if is_held else errs).append(abs(err))
            print(f"{fw:<11}{lim:>6}{a_calls:>6}{b:>7.2f}{r['depth']:>3}"
                  f"{r['calls']:>10}{pred:>11.0f}{err:>+7.1f}"
                  f"{'  <-- held out' if is_held else ''}")

    print("-" * 61)
    if errs:
        print(f"in-sample mean |error|  {sum(errs)/len(errs):.1f}%  (n={len(errs)})")
    if held:
        print(f"HELD-OUT mean |error|   {sum(held)/len(held):.1f}%  (n={len(held)})")

    # Calls understate cost: context grows inside each agent's loop too.
    print("\ntoken growth vs call growth (synthetic, depth 0 -> max):")
    for (fw, lim), g in sorted(groups.items()):
        g.sort(key=lambda r: r["depth"])
        if len(g) < 2 or not g[0]["tokens"]:
            continue
        cg = g[-1]["calls"] / g[0]["calls"]
        tg = g[-1]["tokens"] / g[0]["tokens"]
        print(f"  {fw:<11} limit={lim:<3} calls x{cg:>6.1f}   tokens x{tg:>7.1f}"
              f"   tokens/calls = {tg/cg:.2f}")


if __name__ == "__main__":
    main()
