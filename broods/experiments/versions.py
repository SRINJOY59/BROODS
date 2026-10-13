"""Longitudinal sweep: has any release of any framework ever composed?

    python -m broods.experiments.versions [--frameworks ...]

For ~6 releases per framework spread from its first stable minor to the
latest (results/versions_plan.json), install that release in a throwaway
venv, run the spawn probe at depths 0-2 through the UNCHANGED adapter, record
the default iteration bound the release ships, and -- for the OpenAI Agents
SDK -- rerun the handoff-vs-as_tool experiment. Venvs are deleted after each
release and the uv cache after each framework: the machine this ran on had
6 GB free, so the sweep also refuses to start a release below MIN_FREE_GB.

An adapter written for today's API may not run an old release. That is
recorded as ADAPTER_INCOMPATIBLE, never as a level: an old release we could
not drive says nothing about containment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from broods.probes import REGISTRY
from broods.probes.base import ProbeResult
from broods.proxy import serve
from broods.runner import run_adapter

ROOT = Path(__file__).resolve().parents[2]
# Outside the repository, which sits in a synced OneDrive folder.
import tempfile as _tf
HIST = Path(_tf.gettempdir()) / "broods" / "venvs_hist"
OUT = ROOT / "results" / "versions"
PORT = 8133
MIN_FREE_GB = 2.0

SPECS = {
    "openai_agents": lambda v: [f"openai-agents=={v}"],
    "crewai": lambda v: [f"crewai=={v}"],
    "smolagents": lambda v: [f"smolagents=={v}", "openai"],
    "pydantic_ai": lambda v: [f"pydantic-ai-slim[openai]=={v}"],
    "autogen": lambda v: [f"autogen-agentchat=={v}", f"autogen-ext[openai]=={v}"],
    "langgraph": lambda v: [f"langgraph=={v}", "langchain-openai"],
}

DEFAULT_PROBE = {
    "openai_agents": "from agents.run import DEFAULT_MAX_TURNS as d; print(d)",
    "crewai": "from crewai import Agent; print(Agent.model_fields['max_iter'].default)",
    "smolagents": ("import inspect, smolagents as s; print(inspect.signature("
                   "s.MultiStepAgent.__init__).parameters['max_steps'].default)"),
    "pydantic_ai": "from pydantic_ai.usage import UsageLimits; print(UsageLimits().request_limit)",
    "autogen": ("import inspect; from autogen_agentchat.agents import AssistantAgent; "
                "p=inspect.signature(AssistantAgent.__init__).parameters; "
                "print(p['max_tool_iterations'].default if 'max_tool_iterations' in p else 'absent')"),
    "langgraph": ("try:\n from langgraph._internal._config import DEFAULT_RECURSION_LIMIT as d\n"
                  "except Exception:\n from langchain_core.runnables.config import "
                  "DEFAULT_RECURSION_LIMIT as d\nprint(d)"),
}


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


ABORT_FREE_GB = 1.2


def sh(args, env=None, timeout=900) -> subprocess.CompletedProcess:
    """subprocess.run with a disk watchdog. The between-release guard was not
    enough: one crewai install took the disk from 2.7 GB to under 1 GB while
    running, so an install is killed mid-flight below ABORT_FREE_GB."""
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env={**os.environ, **(env or {})})
    t0 = time.time()
    while p.poll() is None:
        if free_gb() < ABORT_FREE_GB or time.time() - t0 > timeout:
            p.kill()
            out, err = p.communicate()
            return subprocess.CompletedProcess(args, 99, out,
                                               (err or "") + " ABORTED: low disk or timeout")
        time.sleep(0.5)
    out, err = p.communicate()
    return subprocess.CompletedProcess(args, p.returncode, out, err)


def one(fw: str, ver: str, date: str, cache: Path) -> dict:
    row = {"framework": fw, "version": ver, "released": date}
    venv = HIST / f"{fw}-{ver}"
    shutil.rmtree(venv, ignore_errors=True)
    env = {"UV_CACHE_DIR": str(cache)}
    t0 = time.time()
    r = sh(["uv", "venv", str(venv), "--python", "3.11", "-q"], env)
    exe = venv / "Scripts" / "python.exe"
    if not exe.exists():
        exe = venv / "bin" / "python"
    # Resolve dependencies as the index stood two weeks after the release.
    # Installing an old release beside TODAY's openai package failed on a
    # schema change (openai-agents 0.0.19: InputTokensDetails.cache_write_tokens)
    # -- a dependency-drift error, not a property of the release.
    import datetime as _dt
    cutoff = (_dt.date.fromisoformat(date) + _dt.timedelta(days=14)).isoformat()
    row["resolved_as_of"] = cutoff
    r = sh(["uv", "pip", "install", "-q", "--python", str(exe),
            "--exclude-newer", cutoff, *SPECS[fw](ver)],
           {**env, "VIRTUAL_ENV": str(venv)})
    row["install_s"] = round(time.time() - t0, 1)
    if r.returncode:
        row["status"] = "INSTALL_FAILED"
        row["error"] = (r.stderr or r.stdout)[-400:]
        shutil.rmtree(venv, ignore_errors=True)
        return row
    try:
        d = sh([str(exe), "-c", DEFAULT_PROBE[fw]], timeout=120)
        row["default_bound"] = (d.stdout.strip().splitlines() or ["?"])[-1] \
            if d.returncode == 0 else f"ERR: {(d.stderr or '')[-160:]}"

        rows = []
        for depth in (0, 1, 2):
            cfg = {"probe": "spawn", "depth": depth, "limit": 5,
                   "base_url": f"http://127.0.0.1:{PORT}/v1", "model": "mock",
                   "api_key": "unused", "proxy_port": PORT, "real": False}
            res = run_adapter("spawn", fw, cfg, timeout=240, exe=exe)
            rows.append(res)
            if not res.get("supported") or not res.get("metrics"):
                break
        row["spawn"] = [{"depth": x.get("depth"), "outcome": x.get("outcome"),
                         "calls": (x.get("metrics") or {}).get("llm_calls"),
                         "agents": (x.get("metrics") or {}).get("agents_built"),
                         "error": (x.get("error") or "")[:200] or None}
                        for x in rows]
        fields = set(ProbeResult.__dataclass_fields__)
        results = [ProbeResult(**{k: v for k, v in x.items() if k in fields})
                   for x in rows]
        if all(x.get("outcome") in ("ADAPTER_ERROR", "NO_RESULT", "HARNESS_TIMEOUT")
               for x in rows):
            row["status"] = "ADAPTER_INCOMPATIBLE"
        else:
            lvl, why = REGISTRY["spawn"](None).safe_verdict(results)
            row["status"], row["level"], row["why"] = "OK", lvl, why

        if fw == "openai_agents":
            n = sh([str(exe), str(ROOT / "broods" / "native" / "openai_handoff.py")],
                   timeout=600)
            cases = []
            for line in n.stdout.splitlines():
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            row["native"] = cases or {"error": (n.stderr or "")[-300:]}
    finally:
        shutil.rmtree(venv, ignore_errors=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frameworks", nargs="+", default=list(SPECS))
    a = ap.parse_args()
    plan = json.loads((ROOT / "results" / "versions_plan.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "versions.jsonl"
    done = set()
    if out.exists():
        done = {(r["framework"], r["version"]) for r in
                map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    serve(PORT)
    time.sleep(0.3)
    HIST.mkdir(parents=True, exist_ok=True)
    for fw in a.frameworks:
        cache = HIST / f"_uvcache_{fw}"
        for ver, date in plan[fw]:
            if (fw, ver) in done:
                continue
            if free_gb() < MIN_FREE_GB:
                print(f"STOP: only {free_gb():.1f} GB free", flush=True)
                return
            t0 = time.time()
            row = one(fw, ver, date, cache)
            # Per release, not per framework: crewai's cache alone reached
            # ~1.6 GB and pushed the disk below the guard mid-framework.
            shutil.rmtree(cache, ignore_errors=True)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            calls = [s.get("calls") for s in row.get("spawn", [])]
            print(f"{fw:<14} {ver:<9} {date}  {row['status']:<21} "
                  f"level={row.get('level', '-'):<4} calls={calls} "
                  f"default={row.get('default_bound')}  ({time.time() - t0:.0f}s, "
                  f"{free_gb():.1f} GB free)", flush=True)
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
