"""Shared subprocess entrypoint for every framework adapter script.

Each `frameworks/<name>_adapter.py` defines its FrameworkAdapter subclass and
ends with `main(MyAdapter)`. Separated from base.py so base.py stays a pure
interface with zero I/O and no sys.path manipulation.
"""
from __future__ import annotations

import json
import pathlib
import sys
import traceback
from dataclasses import asdict
from pathlib import Path


def main(adapter_cls) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
    from broods.probes import REGISTRY
    from broods.probes.base import ProbeResult
    from broods.proxy_client import ProxyClient

    cfg = json.loads(sys.stdin.read() or "{}")
    try:
        proxy = ProxyClient(cfg["proxy_port"])
        fw = adapter_cls(cfg["model"], cfg["base_url"], cfg["api_key"])
        probe = REGISTRY[cfg["probe"]](proxy)
        extra = {k: cfg[k] for k in ("hang_s", "cap") if k in cfg}
        result = probe.run(fw, depth=cfg.get("depth", 0), limit=cfg.get("limit", 5),
                           task=cfg.get("task", "Research a topic."),
                           real=cfg.get("real", False), **extra)
    except Exception as e:  # noqa: BLE001 -- import failure, bad env, etc.
        # Unsupported is a designed signal meaning "this framework cannot
        # express what the probe needs". Reporting it as ADAPTER_ERROR made
        # four honest L0s look like harness crashes.
        if type(e).__name__ == "Unsupported":
            ProbeResult(probe=cfg.get("probe", "?"),
                        framework=getattr(adapter_cls, "name", "unknown"),
                        supported=False, outcome="UNSUPPORTED",
                        error=str(e)).emit_to(cfg.get("result_path"))
            return
        result = ProbeResult(probe=cfg.get("probe", "?"),
                             framework=getattr(adapter_cls, "name", "unknown"),
                             supported=False, outcome="ADAPTER_ERROR",
                             error=f"{type(e).__name__}: {e}",
                             metrics={"traceback": traceback.format_exc()[-500:]})
    payload = json.dumps(asdict(result))
    # Prefer a file handoff. Stdout is shared with whatever the framework
    # decides to print, and smolagents in particular produced rc=0 with an
    # empty captured stdout inside long runs -- a lost result indistinguishable
    # from a crash. A file cannot be swallowed by console handling.
    out = cfg.get("result_path")
    if out:
        try:
            pathlib.Path(out).write_text(payload, encoding="utf-8")
        except Exception:  # noqa: BLE001 -- stdout below is the fallback
            pass
    print(payload, flush=True)
