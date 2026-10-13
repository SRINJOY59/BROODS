"""P10 -- concurrency bound.

The thread-pool analogue: can an operator cap how many model calls are in
flight simultaneously? Unbounded concurrency is what turns a delegation tree
from an expensive run into a provider-rate-limit incident, and it is the one
resource dimension where the damage is instantaneous rather than cumulative.

LangGraph's `max_concurrency` (RunnableConfig) is the only declared bound of
this kind across the eight systems.

Exercising it requires genuine parallelism. An earlier version reused the
sequential delegation tree, where one tool runs at a time, so peak
concurrency was 1 and the probe honestly reported N/A -- the bound existed
but was never under pressure. Here the injector returns FANOUT tool calls in
a single reply, which is how a model actually asks for parallel work; each
child then issues its own request. A framework that dispatches tools
concurrently will hold several sockets open at once, and one that does not
will show a peak of 1 -- itself a reportable property, since a serial
framework cannot exceed a concurrency bound but also cannot exploit one.

Measured at the proxy as the high-water mark of simultaneously open requests,
which is ground truth: the framework cannot under-report sockets it actually
had open at once.
"""
from __future__ import annotations

import itertools
import time

from broods.frameworks.base import Capability, Unsupported
from broods.probes.base import Probe, ProbeResult

DELAY_S = 0.6      # hold each request open long enough for overlap to be visible


class ConcurrencyProbe(Probe):
    name = "concurrency"
    capability = Capability.CONCURRENCY_BOUND
    key_metric = "peak_inflight"

    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, cap: int = 2000, **kw) -> ProbeResult:
        if not framework.supports(self.capability):
            return ProbeResult(probe=self.name, framework=framework.name,
                               version=framework.version(), depth=depth,
                               supported=False, outcome="UNSUPPORTED",
                               error=f"{framework.name} exposes no concurrency "
                                     f"bound")

        # itertools.count() is atomic under the GIL; `agents[0] += 1` is a
        # read-modify-write and is not. This is the one probe whose whole
        # premise is getting the framework to run that tool concurrently,
        # so it is the one place the race is reachable by design.
        counter = itertools.count(1)
        declared, fanout = 2, 6
        # Latency is required, not incidental: without it each request closes
        # before the next opens and peak concurrency reads as 1 regardless of
        # what the framework does.
        self.proxy.reset(scenario="parallel_delegate", cap=cap,
                         delay_s=DELAY_S, fanout=fanout)
        agents = [0]

        def child_work(t: str) -> str:
            # Each of the fanout tool calls runs a child that issues its own
            # request; overlap between those is what the proxy measures.
            child = framework.make_agent(lambda _t: "leaf", limit=limit,
                                         concurrency=declared)
            agents[0] = next(counter)
            try:
                framework.invoke(child, t)
            except Unsupported:
                raise
            except Exception as e:  # noqa: BLE001
                return f"child stopped: {type(e).__name__}"
            return "branch complete"

        agents[0] = next(counter)
        root = framework.make_agent(child_work, limit=limit,
                                    concurrency=declared)
        t0 = time.time()
        try:
            framework.invoke(root, task)
            outcome, error = "COMPLETED", None
        except Exception as e:  # noqa: BLE001
            outcome, error = type(e).__name__, str(e)[:200]

        st = self.proxy.stats()
        peak = st["peak_inflight"]
        return ProbeResult(
            probe=self.name, framework=framework.name, version=framework.version(),
            depth=depth, outcome=outcome, duration_s=round(time.time() - t0, 3),
            error=error, metrics={
                "declared_concurrency": declared,
                "requested_fanout": fanout,
                "peak_inflight": peak,
                "overshoot_ratio": round(peak / declared, 2) if declared else 0,
                "llm_calls": st["calls"], "agents_built": agents[0],
            })

    def verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        if not results or not results[0].supported:
            why = (results[0].error if results else "no data") or ""
            return "L0", f"concurrency bound not expressible: {why[:60]}"
        base = results[0]
        m = base.metrics
        if not m.get("llm_calls"):
            return "ERR", f"no LLM calls observed ({base.outcome})"
        if m["peak_inflight"] <= 1:
            # The framework serialised tool calls the model asked for in
            # parallel. The bound cannot be violated, but neither is it doing
            # any work -- report the behaviour rather than crediting a pass.
            return "N/A", (f"executes tool calls serially (peak 1 in flight "
                           f"despite {m.get('requested_fanout')} requested at "
                           f"once); bound not exercised")
        r = m["overshoot_ratio"]
        if m.get("requested_fanout") and m["peak_inflight"] >= m["requested_fanout"]:
            # Every tool call the model asked for ran at once: the cap
            # changed nothing.
            return "L1", (f"all {m['peak_inflight']} requested calls ran at once "
                          f"against a declared cap of "
                          f"{m['declared_concurrency']}")
        if r > 1.0:
            return "L2", (f"{m['peak_inflight']} requests in flight against a "
                          f"declared cap of {m['declared_concurrency']}")
        if len(results) < 2:
            return "L3", f"held at {m['peak_inflight']} at depth 0; composition untested"
        deepest = results[-1]
        if deepest.metrics["overshoot_ratio"] > 1.0:
            return "L3", (f"held at depth 0 but reached "
                          f"{deepest.metrics['peak_inflight']} in flight by "
                          f"depth {deepest.depth}")
        return "L4", "concurrency bound holds across delegation"
