"""Emit per-framework requirements files so the suite can be rebuilt.

`.venvs/` is gitignored, so a clone of this repository has the code but no
way to know what to install into which environment. The run_all manifest
records exact versions after the fact, which dates a result but does not
let anyone reproduce it. This writes the install path itself.

One file per framework, because the environments are mutually exclusive:
they pin incompatible versions of openai, pydantic and langchain-core, and
a single combined requirements file would not resolve.

    python -m broods.freeze              # freeze live venvs
    python -m broods.freeze --from-bundle latest
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from broods.runner import FRAMEWORKS, _venv_python

ROOT = Path(__file__).resolve().parents[1]
REQS = ROOT / "requirements"

# What each venv is actually installed FROM. The frozen file pins the full
# transitive closure; this is the human-facing top-level intent.
TOP_LEVEL = {
    "baseline": [],                       # stdlib only -- that is the point
    "langgraph": ["langgraph", "langchain-openai"],
    "crewai": ["crewai"],
    "autogen": ["autogen-agentchat", "autogen-ext[openai]"],
    "pydantic_ai": ["pydantic-ai-slim[openai]"],
    "openai_agents": ["openai-agents"],
    "llamaindex": ["llama-index-core", "llama-index-llms-openai"],
    "smolagents": ["smolagents[openai]"],
}


def freeze_live(fw: str) -> list[str] | None:
    exe = _venv_python(fw)
    if not exe.exists():
        return None
    try:
        out = subprocess.run([str(exe), "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=120)
    except Exception:  # noqa: BLE001
        return None
    return [l for l in (out.stdout or "").splitlines() if l and "==" in l] or []


def from_bundle(path: Path) -> dict[str, list[str]]:
    man = json.loads(path.read_text())["manifest"]["packages"]
    return {fw: pkgs for fw, pkgs in man.items() if isinstance(pkgs, list)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-bundle", default=None,
                    help="read versions from a run_all bundle instead of the "
                        "live venvs; 'latest' picks the newest")
    a = ap.parse_args()

    pinned: dict[str, list[str]] = {}
    if a.from_bundle:
        if a.from_bundle == "latest":
            found = sorted((ROOT / "results" / "full").glob("matrix_*.json"))
            if not found:
                raise SystemExit("no run_all bundle found in results/full")
            p = found[-1]
        else:
            p = Path(a.from_bundle)
        pinned = from_bundle(p)
        source = p.name
    else:
        for fw in FRAMEWORKS:
            got = freeze_live(fw)
            if got is not None:
                pinned[fw] = got
        source = "live venvs"

    REQS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    written = []
    for fw in FRAMEWORKS:
        pkgs = pinned.get(fw)
        if pkgs is None:
            continue
        top = TOP_LEVEL.get(fw, [])
        header = [
            f"# broods :: {fw} environment",
            f"# frozen {stamp} from {source}",
            "#",
            f"# install:  python -m venv .venvs/{fw}",
            f"#           .venvs/{fw}/bin/python -m pip install -r "
            f"requirements/{fw}.txt      # POSIX",
            f"#           .venvs/{fw}/Scripts/python.exe -m pip install -r "
            f"requirements/{fw}.txt  # Windows",
            "#",
        ]
        if top:
            header.append(f"# top-level: {' '.join(top)}")
        else:
            header.append("# top-level: none -- the control is stdlib-only, "
                          "and an empty environment is the claim being made")
        header.append("")
        (REQS / f"{fw}.txt").write_text("\n".join(header + sorted(pkgs)) + "\n")
        written.append((fw, len(pkgs)))

    # A single entry point so nobody has to reconstruct eight commands.
    lines = ["#!/usr/bin/env bash",
             "# Rebuild every environment. Run from the repository root.",
             "# Environments are separate because they pin incompatible",
             "# versions of openai, pydantic and langchain-core.",
             "set -e",
             "",
             "# venv puts the interpreter in Scripts/ on Windows and bin/",
             "# elsewhere; resolve it rather than hardcoding either.",
             "vpy() { [ -x \".venvs/$1/Scripts/python.exe\" ] \\",
             "        && echo \".venvs/$1/Scripts/python.exe\" \\",
             "        || echo \".venvs/$1/bin/python\"; }",
             ""]
    for fw, _ in written:
        lines += [f"python -m venv .venvs/{fw}",
                  f'"$(vpy {fw})" -m pip install -q -r requirements/{fw}.txt',
                  ""]
    (REQS / "setup.sh").write_text("\n".join(lines))

    for fw, n in written:
        print(f"  requirements/{fw}.txt   {n} pinned")
    print(f"\n{len(written)} environments -> {REQS}")


if __name__ == "__main__":
    main()
