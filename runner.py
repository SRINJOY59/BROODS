"""Orchestrator: runs one probe x framework x depth matrix and emits results.

Only stdlib is imported here. The runner never imports a framework or a
probe's dependencies -- each adapter runs in its own venv's interpreter as a
subprocess, which is what keeps three mutually incompatible dependency trees
from colliding. It DOES import broods.probes (stdlib-only) to reuse each
probe's verdict() logic on the results it collects.

    python -m broods.runner --probe spawn --depths 0 1 2
    python -m broods.runner --probe deadline --frameworks crewai langgraph
    python -m broods.runner --probe cancellation --depths 0 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from broods.probes import REGISTRY
from broods.probes.base import ProbeResult
from broods.proxy import serve
from broods.proxy_client import ProxyClient

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS_DIR = ROOT / "broods" / "frameworks"
FRAMEWORKS = ("baseline", "langgraph", "crewai", "autogen",
              "pydantic_ai", "openai_agents", "llamaindex", "smolagents")


def _load_dotenv(path: Path) -> None:
    """Minimal stdlib .env loader -- no point pulling in a dependency for
    three lines. Existing environment variables always win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _venv_python(name: str) -> Path:
    # baseline gets its own EMPTY venv rather than the ambient interpreter.
    # It is stdlib-only either way, so the measurements were never affected --
    # but running it on the harness's own environment made the manifest list
    # 312 packages including LangGraph for what is supposed to be the
    # dependency-free control, which is not a claim a reader should have to
    # take on trust.
    v = ROOT / ".venvs" / name
    win, nix = v / "Scripts" / "python.exe", v / "bin" / "python"
    return win if win.exists() else nix if nix.exists() else win


def run_adapter(probe: str, framework: str, cfg: dict, timeout: float) -> dict:
    """Execute one adapter+probe in its own interpreter; return the JSON result.

    Adapters can be chatty on stdout, so the result is the last parseable
    JSON line. Every failure path still yields a row -- a missing venv or a
    hung adapter is data about the run, not a reason to abort the matrix.
    """
    fail = {"probe": probe, "framework": framework, "version": "?",
            "supported": True, "depth": cfg.get("depth", 0), "duration_s": 0.0,
            "outcome": "?", "error": None, "metrics": {}}
    exe = _venv_python(framework)
    script = FRAMEWORKS_DIR / framework / "adapter.py"
    if not exe.exists():
        return {**fail, "supported": False, "outcome": "NO_VENV",
                "error": f"interpreter missing: {exe}"}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        result_path = Path(tf.name)
    cfg = {**cfg, "result_path": str(result_path)}
    try:
        p = subprocess.run([str(exe), str(script)], input=json.dumps(cfg),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        result_path.unlink(missing_ok=True)
        return {**fail, "outcome": "HARNESS_TIMEOUT", "duration_s": timeout,
                "error": "adapter exceeded harness timeout"}
    # File first -- see _entrypoint for why stdout alone is not reliable.
    try:
        if result_path.stat().st_size:
            d = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(d, dict) and "framework" in d:
                return {**fail, **d}
    except Exception:  # noqa: BLE001
        pass
    finally:
        result_path.unlink(missing_ok=True)

    for line in reversed((p.stdout or "").strip().splitlines()):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "framework" in d:
            return {**fail, **d}
    # Report BOTH streams: a framework that dies quietly writes nothing to
    # stderr, and "no stderr" alone gives nothing to debug from.
    tail = (f"rc={p.returncode} "
            f"stderr={(p.stderr or '')[-300:]!r} "
            f"stdout={(p.stdout or '')[-300:]!r}")
    return {**fail, "outcome": "NO_RESULT", "error": tail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="spawn", choices=list(REGISTRY))
    ap.add_argument("--frameworks", nargs="+", default=list(FRAMEWORKS))
    ap.add_argument("--depths", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--cap", type=int, default=None,
                    help="proxy safety backstop; we build runaways on purpose")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--real", action="store_true",
                    help="passthrough to OpenRouter instead of the injector "
                        "(spawn probe only -- other probes need the injector)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--trials", type=int, default=1,
                    help="repeats per cell; real-model runs vary, so a single "
                        "trial is not a measurement. Reports median + range.")
    ap.add_argument("--hang-s", type=float, default=None, dest="hang_s",
                    help="deadline probe: injected per-call latency in seconds")
    ap.add_argument("--task", default=None,
                    help="override the root task string; the counting unit "
                         "should be invariant to it (default: 'Research a topic.')")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    model = a.model or ("nvidia/nemotron-3-super-120b-a12b:free" if a.real else "mock")
    key = os.environ.get("OPENROUTER_API_KEY", "unused")
    if a.real and key == "unused":
        raise SystemExit("--real needs OPENROUTER_API_KEY in .env or the environment")

    serve(a.port)
    time.sleep(0.3)
    base_url = f"http://127.0.0.1:{a.port}/v1"
    print(f"proxy {base_url}   probe={a.probe}   limit={a.limit}\n")

    hdr = f"{'framework':<12}{'ver':>10}{'d':>3}  outcome            metrics"
    print(hdr); print("-" * len(hdr))

    rows: list[dict] = []
    by_fw: dict[str, list[dict]] = {}
    probe_instance = REGISTRY[a.probe](ProxyClient(a.port))

    key_metric = probe_instance.key_metric

    for fw in a.frameworks:
        by_fw[fw] = []
        for d in a.depths:
            cfg = {"probe": a.probe, "depth": d, "limit": a.limit,
                   "base_url": base_url, "model": model, "api_key": key,
                   "proxy_port": a.port, "real": a.real}
            if a.task is not None:
                cfg["task"] = a.task
            if a.hang_s is not None:
                cfg["hang_s"] = a.hang_s
            if a.cap is not None:
                cfg["cap"] = a.cap

            trials = [run_adapter(a.probe, fw, cfg, timeout=a.timeout)
                      for _ in range(a.trials)]

            # Report the MEDIAN trial, not the first. Real-model latency and
            # model nondeterminism both vary run to run, so a single sample
            # is an anecdote; the spread belongs in the record alongside it.
            # `.get(km) is not None`, not `km in metrics`: accounting reports
            # attribution_error_pct=None when a framework exposes no usage at
            # all, and sorting several Nones raises TypeError, aborting the
            # run. run_all.py was fixed in c3d137d; this call site was not.
            usable = [t for t in trials
                      if t["supported"]
                      and (t.get("metrics") or {}).get(key_metric) is not None]
            if usable:
                usable.sort(key=lambda t: t["metrics"][key_metric])
                res = usable[len(usable) // 2]
                vals = [t["metrics"][key_metric] for t in usable]
                res = {**res, "n_trials": len(trials),
                       f"{key_metric}_median": res["metrics"][key_metric],
                       f"{key_metric}_min": min(vals),
                       f"{key_metric}_max": max(vals)}
            else:
                res = {**trials[0], "n_trials": len(trials)}
            if a.trials > 1:
                res["all_trials"] = [t.get("metrics") for t in trials]

            rows.append(res); by_fw[fw].append(res)

            spread = ""
            if usable and a.trials > 1:
                lo, hi = res[f"{key_metric}_min"], res[f"{key_metric}_max"]
                spread = f"   [{key_metric} {lo}-{hi} over {a.trials}]"
            print(f"{fw:<12}{res['version']:>10}{d:>3}  {res['outcome']:<18} "
                  f"{res.get('metrics', {})}{spread}")
            if res.get("error") and not res["supported"]:
                print(f"{'':>12}   reason: {res['error'][:100]}")
            if not res["supported"]:
                break        # nothing to measure at deeper depths

    print()
    verdicts = {}
    fields = set(ProbeResult.__dataclass_fields__)
    for fw in a.frameworks:
        # Rows carry runner-added aggregate keys (n_trials, *_median, ...)
        # that ProbeResult does not declare; strip them for reconstruction.
        results = [ProbeResult(**{k: v for k, v in r.items() if k in fields})
                   for r in by_fw[fw]]
        lvl, why = probe_instance.safe_verdict(results)
        verdicts[fw] = {"level": lvl, "rationale": why}
        print(f"  {fw:<12} {lvl}   {why}")

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = outdir / f"{a.probe}_{stamp}.json"
    path.write_text(json.dumps({"probe": a.probe, "model": model, "limit": a.limit,
                                "real": a.real,
                                "utc": stamp, "verdicts": verdicts, "rows": rows},
                               indent=2))
    print(f"\n{len(rows)} cells -> {path}")


if __name__ == "__main__":
    main()
