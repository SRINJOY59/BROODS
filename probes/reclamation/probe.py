"""P8 -- resource reclamation.

The OS analogue is `wait()` and zombie reaping: when a process exits, the
kernel reclaims its descriptors, memory and thread stacks, and a parent that
never reaps leaves zombies behind. Nothing equivalent exists for agents --
there is no reaper, because there is no process table.

This probe runs a delegation tree to COMPLETION (no cancellation, no faults,
no adversarial condition) and then asks whether the interpreter returns to
the state it started in. That framing is deliberate: P3 shows resources
survive an interrupted run, which is arguably excusable. This asks whether
they survive an entirely normal one, which is not.

Measured after an explicit gc.collect(), so anything still counted is
genuinely reachable rather than merely awaiting collection.
"""
from __future__ import annotations

import gc
import threading
import time

from broods.frameworks.base import Capability
from broods.frameworks.base import Unsupported
from broods.probes.base import Probe, ProbeResult

SETTLE_S = 2.0     # give framework-owned worker threads a chance to wind down


class ReclamationProbe(Probe):
    name = "reclamation"
    capability = None          # emergent: no framework declares a reaper
    key_metric = "threads_leaked"

    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, cap: int = 2000, **kw) -> ProbeResult:
        if real:
            # This probe is synthetic by construction, so honouring
            # real= would be a lie: run_all tags every --real-probes row
            # real=true, and a bundle whose purpose is evidencing live
            # validation must not contain fault-injected runs labelled
            # as live ones.
            raise Unsupported(
                "reclamation does not support real=True; its scenario is fault-injected by construction")
        self.proxy.reset(scenario="always_delegate", cap=cap)

        # Request an iteration bound only where the framework has one.
        # These scenarios terminate on their own, so the bound is belt-and-
        # braces; demanding it unconditionally made every probe unrunnable
        # against a framework that simply lacks the parameter.
        lim = limit if framework.supports(Capability.ITERATION_BOUND) else None
        gc.collect()
        threads_before = threading.active_count()
        objects_before = len(gc.get_objects())

        agents_built = [0]

        def make(d: int):
            agents_built[0] += 1
            if d == 0:
                return framework.make_agent(lambda t: "leaf", limit=lim)

            def delegate(t: str) -> str:
                child = make(d - 1)
                try:
                    framework.invoke(child, t)
                except Exception as e:  # noqa: BLE001
                    return f"child stopped: {type(e).__name__}"
                return "child complete"
            return framework.make_agent(delegate, limit=lim)

        # Sample peak concurrency during the run. Without this, a framework
        # that never spawns a thread scores identically to one that spawns and
        # correctly reaps them -- "0 leaked" would mean "0 created".
        peak = [threads_before]
        stop = threading.Event()

        def _sampler():
            while not stop.is_set():
                peak[0] = max(peak[0], threading.active_count())
                time.sleep(0.05)

        sampler = threading.Thread(target=_sampler, daemon=True)
        sampler.start()

        root = make(depth)
        t0 = time.time()
        try:
            framework.invoke(root, task)
            outcome, error = "COMPLETED", None
        except Exception as e:  # noqa: BLE001
            # Hitting the declared bound is the NORMAL end of this tree, not
            # a failure -- the point is that the run finished on its own terms.
            outcome, error = type(e).__name__, str(e)[:200]

        calls = self.proxy.stats()["calls"]
        stop.set()
        sampler.join(timeout=1.0)

        # Drop every reference the probe itself holds, then let the runtime
        # settle before measuring, so nothing is attributed to our own frames.
        del root
        time.sleep(SETTLE_S)
        gc.collect()
        threads_after = threading.active_count()
        objects_after = len(gc.get_objects())

        return ProbeResult(
            probe=self.name, framework=framework.name, version=framework.version(),
            depth=depth, outcome=outcome, duration_s=round(time.time() - t0, 3),
            error=error, metrics={
                "agents_built": agents_built[0],
                "llm_calls": calls,
                "threads_before": threads_before,
                "threads_after": threads_after,
                "threads_leaked": threads_after - threads_before,
                "threads_peak": peak[0],
                # excludes our own sampler thread
                "threads_spawned": max(0, peak[0] - threads_before - 1),
                "objects_leaked": objects_after - objects_before,
            })

    def verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        if not results:
            return "L0", "no data"
        base = results[0]
        m = base.metrics
        if not m.get("llm_calls"):
            return "ERR", f"no LLM calls observed ({base.outcome})"

        deepest = results[-1]
        leaked = deepest.metrics["threads_leaked"]
        agents = deepest.metrics["agents_built"]
        spawned = deepest.metrics.get("threads_spawned", 0)
        if spawned == 0:
            # Nothing to reclaim. Reporting L4 here would credit a framework
            # for reaping threads it never created.
            return "N/A", (f"executes synchronously; no threads spawned across "
                           f"{agents} agents, so reclamation is not exercised")
        if leaked <= 0:
            return "L4", (f"all {spawned} spawned threads reclaimed after "
                          f"{agents} agents completed at depth {deepest.depth}")
        if leaked < agents:
            return "L2", (f"{leaked} thread(s) left running after {agents} "
                          f"agents completed at depth {deepest.depth}")
        return "L1", (f"{leaked} threads left running after {agents} agents "
                      f"completed normally at depth {deepest.depth}; "
                      f"nothing reaps them")
