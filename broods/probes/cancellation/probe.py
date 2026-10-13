"""P3 -- cancellation and orphans (concurrent fan-out).

Does cancelling a parent stop the sub-agents it already started?

Why the topology matters
-----------------------
An earlier version of this probe reused P1's tree, where a parent delegates
by CALLING the child and blocking until it returns. In that shape a child
only ever runs inside the parent's own in-flight tool call, so there is no
instant at which a child executes independently -- cancellation lands either
before anything spawned or after the whole chain unwound. It measured
`total_calls: 1` at every depth and reported a meaningless "0 orphans".

That is a real finding in itself: **sequential delegation is structurally
immune to orphaning.** It is also why testing the actual question requires
concurrency: the parent must start children and RETURN, leaving them
running. Only then can cancelling the parent leave work behind.

What is measured
----------------
The proxy is an independent observer: it timestamps every request that
physically reached the socket. Any request arriving after the cancellation
timestamp is work that outlived the cancel, and no framework can misreport
it -- the bytes either arrived or they did not.

Scope limitation (stated, not hidden)
-------------------------------------
Children are started through the adapter's own non-blocking `start()`, so
the harness owns the concurrency. This models the common real pattern of
firing off a sub-agent from inside a tool, and asks whether frameworks offer
any way to bind a spawned child to its parent's lifecycle -- the agent
analogue of a process group, where killing the leader reaps the rest. It
does NOT exercise framework-native parallel primitives (AutoGen teams,
CrewAI `async_execution`), where a framework may track children it started
itself. Those are the stronger test and remain future work.
"""
from __future__ import annotations

import time

from broods.frameworks.base import Capability, Unsupported
from broods.probes.base import Probe, ProbeResult

SPAWN_WAIT_S = 20.0  # max wait for the parent to actually fan out
SETTLE_S = 4.0     # after fan-out, let children get several calls in
OBSERVE_S = 6.0    # window after cancel in which orphans reveal themselves
DELAY_S = 1.0      # mock latency, so the race window is real (see history above)
# Children must outlive the observation window but MUST self-terminate soon
# after. An early version used 500, so orphans kept calling for ~8 minutes
# each and accumulated across cells until the suite stalled. The window is
# SETTLE+OBSERVE=10s, so ~30 calls at DELAY_S is ample headroom.
CHILD_LIMIT = 30
FANOUT = 3


class CancellationProbe(Probe):
    name = "cancellation"
    capability = Capability.EXTERNAL_CANCEL
    key_metric = "orphan_calls"

    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, fanout: int = FANOUT, **kw) -> ProbeResult:
        if not framework.supports(self.capability):
            return ProbeResult(probe=self.name, framework=framework.name,
                               version=framework.version(), depth=depth,
                               supported=False, outcome="UNSUPPORTED",
                               error=f"{framework.name} exposes no external "
                                     f"cancellation once a run has started")

        if real:
            self.proxy.reset(scenario="passthrough", cap=5000,
                             api_key=framework.api_key)
        else:
            self.proxy.reset(scenario="always_delegate", cap=5000, delay_s=DELAY_S)

        children: list = []

        def parent_tool(t: str) -> str:
            # Concurrent fan-out: START each child and return immediately.
            # The parent does not await them, so from here on the children
            # are live work the parent no longer holds.
            for i in range(fanout):
                child = framework.make_agent(lambda _t: "leaf", limit=CHILD_LIMIT)
                children.append(framework.start(child, f"{t}-{i}"))
            return f"spawned {fanout} children"

        root = framework.make_agent(parent_tool, limit=limit)
        t0 = time.time()
        try:
            run = framework.start(root, task)
        except Unsupported as e:
            return ProbeResult(probe=self.name, framework=framework.name,
                               version=framework.version(), depth=depth,
                               supported=False, outcome="UNSUPPORTED", error=str(e))

        # Wait for the fan-out to HAPPEN rather than assuming it has. A fixed
        # sleep made the verdict timing-dependent: if the parent had not yet
        # reached its tool, `children` was empty and the probe returned L0
        # ("no children spawned") for a framework that cancels fine. Seen as a
        # LangGraph L0 in a full matrix run that was L1 standalone.
        deadline = time.time() + SPAWN_WAIT_S
        while not children and time.time() < deadline:
            time.sleep(0.1)
        if not children:
            self.proxy.quiesce()
            try:
                framework.cancel(run)
            except Exception:  # noqa: BLE001
                pass
            return ProbeResult(
                probe=self.name, framework=framework.name,
                version=framework.version(), depth=depth, supported=True,
                outcome="NO_FANOUT", duration_s=round(time.time() - t0, 3),
                error=f"parent did not spawn children within {SPAWN_WAIT_S}s",
                metrics={"children_spawned": 0,
                         "total_calls": self.proxy.stats()["calls"]})

        time.sleep(SETTLE_S)              # now let the children accumulate calls
        calls_before = self.proxy.stats()["calls"]

        # Cancel ONLY the parent. Nothing is said to the children -- the
        # question is precisely whether the framework knows they exist.
        t_cancel = framework.cancel(run)
        # Hand the framework's own instant to the proxy rather than letting
        # it stamp one a round-trip later; otherwise orphan requests landing
        # in that gap are miscounted as pre-cancel.
        self.proxy.cancel(t_cancel)
        try:
            framework.join(run, timeout=5.0)
        except Exception:  # noqa: BLE001  -- a cancelled run may raise
            pass

        time.sleep(OBSERVE_S)
        stats = self.proxy.stats()

        # Measurement is complete -- now stop the orphans. Asking the
        # framework is not enough: these children are by definition ones it
        # cannot reach (LangGraph's in particular survive an explicit cancel
        # and previously stalled every subsequent trial). Quiescing the proxy
        # ends them from the other side, since an agent whose next call
        # returns a terminal answer simply finishes.
        self.proxy.quiesce()
        for c in children:
            try:
                framework.cancel(c)
            except Exception:  # noqa: BLE001
                pass
        for c in children:                      # let them observe the stop
            try:
                framework.join(c, timeout=3.0)
            except Exception:  # noqa: BLE001
                pass

        orphans = stats["orphan_calls"]
        return ProbeResult(
            probe=self.name, framework=framework.name, version=framework.version(),
            depth=depth, outcome="ORPHANED" if orphans else "CLEAN",
            duration_s=round(time.time() - t0, 3),
            metrics={
                "children_spawned": len(children),
                "calls_before_cancel": calls_before,
                "orphan_calls": orphans,
                "orphan_lifetime_s": stats["orphan_lifetime_s"],
                "total_calls": stats["calls"],
                "observe_window_s": OBSERVE_S,
            })

    def verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        if not results or not results[0].supported:
            why = (results[0].error if results else "no data") or ""
            return "L0", f"cancellation not expressible: {why[:60]}"
        base = results[0]
        m = base.metrics
        if not m.get("children_spawned"):
            # Not L0: "the parent never fanned out" says nothing about whether
            # cancellation is expressible, which is what L0 asserts.
            return "ERR", (f"no children spawned ({base.outcome}); "
                           f"probe did not exercise fan-out")
        if m["orphan_calls"] == 0:
            return "L4", (f"cancel reached all {m['children_spawned']} children; "
                          f"no calls in {m['observe_window_s']}s after cancel")
        return "L1", (f"{m['orphan_calls']} calls from {m['children_spawned']} "
                      f"orphaned children over {m['orphan_lifetime_s']}s "
                      f"after the parent was cancelled")
