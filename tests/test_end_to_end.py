"""Run the spawn probe end to end on the two standard-library systems.

The control and its task-scoped variant need no framework, so the current
interpreter can stand in for their virtual environments. This exercises the
whole path: runner, adapter subprocess, probe, proxy and verdict.
"""
import sys
from pathlib import Path

import pytest

from broods.runner import run_adapter


def calls(proxy, system, depth, limit=5):
    cfg = {"probe": "spawn", "depth": depth, "limit": limit,
           "base_url": proxy.base_url, "model": "mock", "api_key": "unused",
           "proxy_port": proxy.port, "real": False, "cap": 1000}
    row = run_adapter("spawn", system, cfg, timeout=120, exe=Path(sys.executable))
    assert row["outcome"] not in ("NO_RESULT", "HARNESS_TIMEOUT"), row["error"]
    return row["metrics"]["llm_calls"], row["metrics"]["agents_built"]


@pytest.mark.parametrize("depth, expected", [(0, (5, 1)), (1, (30, 6)), (2, (155, 31))])
def test_per_run_bound_multiplies(proxy, depth, expected):
    assert calls(proxy, "baseline", depth) == expected


@pytest.mark.parametrize("depth, expected", [(0, (5, 1)), (1, (5, 2)), (2, (5, 3))])
def test_task_scoped_bound_holds(proxy, depth, expected):
    assert calls(proxy, "baseline_scoped", depth) == expected
