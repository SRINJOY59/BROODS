"""Regenerate the entire matrix in one reproducible invocation.

Every number in the write-up should come from a single run of a single
revision. During development results accumulate across dozens of files
written by different versions of the code, and quoting from that pile is how
a table ends up mixing pre-fix and post-fix numbers.

This writes ONE timestamped bundle containing:

    manifest   git revision, python version, per-framework package versions
    rows       every cell, with full metrics
    verdicts   probe x framework -> level + rationale
    matrix.md  the table, generated rather than transcribed

    python -m broods.run_all
    python -m broods.run_all --probes spawn accounting --quick
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from broods.probes import REGISTRY
from broods.probes.base import ProbeResult
from broods.proxy import serve
from broods.proxy_client import ProxyClient
from broods.runner import FRAMEWORKS, _load_dotenv, _venv_python, run_adapter

ROOT = Path(__file__).resolve().parents[1]

# Depths worth running per probe. Several probes measure a property that a
# delegation tree does not change -- state_isolation builds its own fan-out of
# victims, fault_isolation its own siblings, cancellation its own concurrent
# children -- so running them deeper repeats one measurement rather than
# adding an axis.
DEPTHS = {
    "spawn": [0, 1, 2],
    "token_budget": [0, 1],
    "accounting": [0, 1],
    "reclamation": [0, 1],
    "deadline": [0],
    "cancellation": [0],
    "fault_isolation": [0],
    "state_isolation": [0],
    "context_bound": [0, 1],
    # depth 0 only: the fan-out happens at the root reply, so a deeper
    # tree repeats the same measurement rather than testing composition.
    # Nested parallelism would need the injector to fan out at every
    # level; noted as future work rather than faked with a duplicate row.
    "concurrency": [0],
    "rate_limit": [0],
}


def manifest(frameworks: list[str]) -> dict:
    """Everything needed to reproduce or date this run."""
    def _git(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                               text=True, timeout=10)
            # A non-zero exit still yields empty stdout, and returning the
            # argument list as if it were output produced a manifest reading
            # `git_commit: "HEAD"` -- provenance that identifies nothing.
            return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
        except Exception:  # noqa: BLE001
            return None

    versions = {}
    for fw in frameworks:
        exe = _venv_python(fw)
        if not exe.exists():
            versions[fw] = "NO_VENV"
            continue
        try:
            out = subprocess.run(
                [str(exe), "-c",
                 "import importlib.metadata as m,json;"
                 "print(json.dumps(sorted(f'{d.metadata[\"Name\"]}=={d.version}' "
                 "for d in m.distributions())))"],
                capture_output=True, text=True, timeout=60)
            versions[fw] = json.loads(out.stdout or "[]")
        except Exception as e:  # noqa: BLE001
            versions[fw] = f"ERROR: {e}"

    commit = _git("rev-parse", "HEAD")
    if not commit:
        print("  WARNING: no git commit recorded -- results are not "
              "traceable to a revision", file=sys.stderr)
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "packages": versions,
    }


def to_markdown(verdicts: dict, probes: list[str], frameworks: list[str]) -> str:
    head = "| system | " + " | ".join(probes) + " |"
    sep = "|" + "---|" * (len(probes) + 1)
    lines = [head, sep]
    for fw in frameworks:
        cells = []
        for p in probes:
            v = verdicts.get(p, {}).get(fw)
            cells.append(v["level"] if v else "—")
        lines.append(f"| {fw} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="+", default=list(REGISTRY))
    ap.add_argument("--frameworks", nargs="+", default=list(FRAMEWORKS))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--trials", type=int, default=1,
                    help="repeats per cell; the median cell is kept and the "
                        "spread recorded. Anything quoted should use >1.")
    ap.add_argument("--real-probes", nargs="*", default=None,
                    help="after the synthetic matrix, re-run these probes "
                        "against a live provider and tag the rows real=true, "
                        "so the bundle itself evidences that findings are not "
                        "mock artifacts")
    ap.add_argument("--real-model", default="nvidia/nemotron-3-super-120b-a12b:free")
    ap.add_argument("--quick", action="store_true",
                    help="depth 0 only -- smoke test, not a publishable run")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    serve(a.port)
    time.sleep(0.3)
    base_url = f"http://127.0.0.1:{a.port}/v1"
    proxy = ProxyClient(a.port)

    print(f"broods full matrix -- {len(a.probes)} probes x "
          f"{len(a.frameworks)} systems\n")
    man = manifest(a.frameworks)
    print(f"  commit {man['git_commit'] or '?'}"
          f"{' (dirty)' if man['git_dirty'] else ''}   python {man['python']}\n")

    rows: list[dict] = []
    verdicts: dict[str, dict] = {}
    t_start = time.time()

    # Stream every cell to disk as it completes. Previously nothing was
    # written until the final line, so a single crash discarded a
    # multi-hour matrix; now a failed run is resumable from the JSONL.
    outdir = Path(a.out) / "full"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    live = (outdir / f"cells_{stamp}.jsonl").open("w", encoding="utf-8")

    for probe_name in a.probes:
        probe = REGISTRY[probe_name](proxy)
        depths = [0] if a.quick else DEPTHS.get(probe_name, [0])
        verdicts[probe_name] = {}
        print(f"{probe_name}  (depths {depths})")

        for fw in a.frameworks:
            per_fw: list[dict] = []
            for d in depths:
                trials = [run_adapter(probe_name, fw, {
                    "probe": probe_name, "depth": d, "limit": a.limit,
                    "base_url": base_url, "model": "mock", "api_key": "unused",
                    "proxy_port": a.port, "real": False, "cap": a.cap,
                }, timeout=a.timeout) for _ in range(a.trials)]

                # Keep the MEDIAN trial and record the spread. A single sample
                # is an anecdote for any probe with real variance.
                km = probe.key_metric
                usable = [t for t in trials
                          if t["supported"] and (t.get("metrics") or {}).get(km) is not None]
                if usable:
                    usable.sort(key=lambda t: t["metrics"][km])
                    res = dict(usable[len(usable) // 2])
                    vals = [t["metrics"][km] for t in usable]
                    res["n_trials"] = len(trials)
                    res["spread"] = {km: [min(vals), max(vals)]}
                else:
                    res = dict(trials[0])
                    res["n_trials"] = len(trials)
                res["probe"], res["depth"] = probe_name, d
                rows.append(res)
                per_fw.append(res)
                live.write(json.dumps(res) + "\n")
                live.flush()
                if not res["supported"]:
                    break

            fields = set(ProbeResult.__dataclass_fields__)
            typed = [ProbeResult(**{k: v for k, v in r.items() if k in fields})
                     for r in per_fw]
            level, why = probe.safe_verdict(typed)
            verdicts[probe_name][fw] = {"level": level, "rationale": why}
            print(f"    {fw:<15}{level:<5} {why[:70]}")
        print()

    # ---- real-model validation ------------------------------------------
    # The synthetic matrix is deterministic by design, but "these are not mock
    # artifacts" is a claim the bundle should carry evidence for rather than
    # assert. These rows are tagged real=true and kept separate.
    real_rows: list[dict] = []
    if a.real_probes:
        key = os.environ.get("OPENROUTER_API_KEY", "unused")
        if key == "unused":
            print("  SKIPPING real validation: no OPENROUTER_API_KEY", file=sys.stderr)
        else:
            print(f"real-model validation ({a.real_model})")
            for probe_name in a.real_probes:
                for fw in a.frameworks:
                    res = run_adapter(probe_name, fw, {
                        "probe": probe_name, "depth": 0, "limit": a.limit,
                        "base_url": base_url, "model": a.real_model,
                        "api_key": key, "proxy_port": a.port,
                        "real": True, "cap": a.cap,
                    }, timeout=a.timeout)
                    res["probe"], res["depth"], res["real"] = probe_name, 0, True
                    real_rows.append(res)
                    print(f"    {probe_name}/{fw:<15}{res['outcome']:<18}"
                          f"{res.get('metrics', {})}")
            print()

    live.close()

    bundle = {"utc": stamp, "manifest": man, "limit": a.limit,
              "quick": a.quick, "trials": a.trials,
              # what actually ran, not the module constant -- a --quick
              # bundle previously claimed spawn:[0,1,2] having run only d0
              "depths": {p: ([0] if a.quick else DEPTHS.get(p, [0]))
                         for p in a.probes},
              "real_model": a.real_model if a.real_probes else None,
              "verdicts": verdicts, "rows": rows, "real_rows": real_rows}
    (outdir / f"matrix_{stamp}.json").write_text(json.dumps(bundle, indent=2))

    md = to_markdown(verdicts, a.probes, a.frameworks)
    (outdir / f"matrix_{stamp}.md").write_text(
        f"# broods matrix\n\n"
        f"commit `{man['git_commit'] or '?'}`"
        f"{' **(dirty tree)**' if man['git_dirty'] else ''} · "
        f"python {man['python']} · {stamp}\n\n{md}\n")

    print(md)
    print(f"\n{len(rows)} cells in {time.time() - t_start:.0f}s -> "
          f"{outdir / f'matrix_{stamp}.json'}")


if __name__ == "__main__":
    main()
