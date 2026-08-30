"""P4 -- fault isolation.

When one child agent's call fails, does the framework's own tool-execution
loop still attempt the remaining siblings, or does the whole run abort? The
parent's tool builds and invokes three children ITSELF, without wrapping the
framework.invoke() calls in a try/except -- so whatever happens next is the
framework's own error-handling behaviour, not ours.

Targeting: the failing sibling is identified by a marker planted in its own
task text, not by call-sequence position. Sequence targeting assumed a
strictly ordered topology and mis-fired as soon as anything retried or
interleaved; a marker follows the child itself, and covers its retries for
free because every attempt carries the same text.
"""
from __future__ import annotations

import time

from broods.frameworks.base import Capability
from broods.probes.base import Probe, ProbeResult

N_SIBLINGS = 3
FAIL_INDEX = 1   # zero-based; the second sibling is the one that fails
MARKER = "[sibling={i}]"   # planted in each child's task; identifies it on the wire


class FaultIsolationProbe(Probe):
    name = "fault_isolation"
    capability = None   # emergent behaviour, not a declared/opt-in guarantee
    key_metric = "siblings_completed"

    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, **kw) -> ProbeResult:
        # fail_at is checked before scenario dispatch in the proxy, so it
        # composes with passthrough: the targeted call still gets a 400
        # (4xx deliberately -- the SDK silently retries 5xx; see proxy.py),
        # everything else genuinely round-trips to the real provider.
        # delegate_first, not finish: with `finish` the parent's very first
        # call returns a terminal answer, so its tool never runs and no
        # siblings are ever created -- the probe measured nothing.
        self.proxy.reset(scenario="passthrough" if real else "delegate_first",
                         cap=200, fail_match=MARKER.format(i=FAIL_INDEX),
                         api_key=framework.api_key if real else None)
        # Request an iteration bound only where the framework has one.
        # These scenarios terminate on their own, so the bound is belt-and-
        # braces; demanding it unconditionally made every probe unrunnable
        # against a framework that simply lacks the parameter.
        lim = limit if framework.supports(Capability.ITERATION_BOUND) else None
        attempted, completed = [], []

        def parent_tool(t: str) -> str:
            for i in range(N_SIBLINGS):
                attempted.append(i)
                child = framework.make_agent(lambda _t: "leaf", limit=lim)
                framework.invoke(child, f"{t} {MARKER.format(i=i)}")  # unwrapped: framework decides
                completed.append(i)
            return "all siblings done"

        root = framework.make_agent(parent_tool, limit=lim)
        t0 = time.time()
        try:
            framework.invoke(root, task)
            outcome, error = "COMPLETED", None
        except Exception as e:  # noqa: BLE001
            outcome, error = type(e).__name__, str(e)[:200]

        # Call count exposes client-side retry. Expected without retry:
        # 1 parent + N_SIBLINGS children. Anything above that means the
        # injected 500 was absorbed by the SDK and re-issued, so a
        # "no failure observed" result would be a retry artifact rather
        # than evidence the framework isolates the fault.
        stats = self.proxy.stats()
        expected = 1 + N_SIBLINGS
        return ProbeResult(
            probe=self.name, framework=framework.name, version=framework.version(),
            depth=depth, outcome=outcome, duration_s=round(time.time() - t0, 3),
            error=error, metrics={
                "siblings_attempted": len(attempted), "siblings_completed": len(completed),
                "n_siblings": N_SIBLINGS, "fail_index": FAIL_INDEX,
                "parent_survived": outcome == "COMPLETED",
                "llm_calls": stats["calls"], "calls_expected_no_retry": expected,
                "retries_observed": max(0, stats["calls"] - expected),
            })

    def verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        if not results:
            return "L0", "no data"
        base = results[0]
        m = base.metrics
        if not m.get("siblings_attempted"):
            return "ERR", (f"no siblings attempted ({base.outcome}); "
                           f"probe did not execute")
        if m["siblings_completed"] >= m["n_siblings"]:
            return "L4", "all siblings completed despite one injected failure"
        if m["parent_survived"]:
            return "L3", (f"parent survived, but only "
                          f"{m['siblings_completed']}/{m['n_siblings']} "
                          f"siblings completed after the failure")
        return "L1", (f"one sibling's failure aborted the whole run "
                      f"({m['siblings_completed']}/{m['n_siblings']} completed)")
