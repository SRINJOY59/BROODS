"""Run the spawn probe against the TypeScript frameworks (Vercel AI SDK, Mastra).

    python -m broods.experiments.ts_spawn [--node-dir DIR]

The agent side is broods/ts/spawn.mjs, run by node from a directory holding
its node_modules (kept outside the repository, which sits in a synced
folder). This side arms the same proxy, counts calls at the socket, and
scores with the unchanged SpawnProbe verdict -- so the TypeScript rows are
measured and levelled exactly as the Python ones are.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from broods.probes.base import ProbeResult
from broods.probes.spawn.probe import SpawnProbe
from broods.proxy import serve
from broods.proxy_client import ProxyClient

ROOT = Path(__file__).resolve().parents[2]
PORT = 8148


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-dir", default=str(Path(tempfile.gettempdir()) / "broods" / "ts"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()
    nd = Path(a.node_dir)
    shutil.copy(ROOT / "broods" / "ts" / "spawn.mjs", nd / "spawn.mjs")
    serve(PORT)
    time.sleep(0.3)
    pc = ProxyClient(PORT)
    out = {}
    for fw in ("vercel", "mastra"):
        rows = []
        for depth in (0, 1, 2):
            trials = []
            for _ in range(a.trials):
                pc.reset(scenario="always_delegate", cap=1000)
                t0 = time.time()
                p = subprocess.run(["node", "spawn.mjs", fw, str(depth), str(a.limit),
                                    pc.base_url], cwd=nd, capture_output=True,
                                   text=True, timeout=600)
                st = pc.stats()
                try:
                    js = json.loads(p.stdout.strip().splitlines()[-1])
                except Exception:  # noqa: BLE001
                    js = {"outcome": "NO_RESULT", "error": (p.stderr or "")[-300:],
                          "version": "?", "agents_built": 0}
                trials.append(ProbeResult(
                    probe="spawn", framework=fw, version=js.get("version", "?"),
                    depth=depth, outcome=js.get("outcome", "?"),
                    duration_s=round(time.time() - t0, 3), error=js.get("error"),
                    metrics={"llm_calls": st["calls"], "agents_built": js.get("agents_built"),
                             "declared_limit": a.limit, "capped": st["capped"],
                             "prompt_tokens_est": st["prompt_tokens_est"]}))
            trials.sort(key=lambda r: r.metrics["llm_calls"])
            med = trials[len(trials) // 2]
            med.metrics["llm_calls_range"] = [trials[0].metrics["llm_calls"],
                                              trials[-1].metrics["llm_calls"]]
            rows.append(med)
            print(f"{fw:<8} {med.version:<9} d{depth}  {med.outcome:<14} "
                  f"calls={med.metrics['llm_calls']} agents={med.metrics['agents_built']} "
                  f"{(med.error or '')[:80]}", flush=True)
        lvl, why = SpawnProbe(None).safe_verdict(rows)
        print(f"  {fw}: {lvl}  {why}\n", flush=True)
        out[fw] = {"level": lvl, "why": why,
                   "rows": [r.__dict__ for r in rows]}
    dest = ROOT / "results" / "ts" / f"ts_spawn_{time.strftime('%Y%m%dT%H%M%S')}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
