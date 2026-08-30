"""P2 -- deadline.

Does a declared wall-clock bound return control to the caller, or is it only
checked between steps -- so one slow call can blow the budget by an
unbounded amount? Uses the proxy's `hang` scenario: every response is
delayed by HANG_S, standing in for a slow or wedged provider.

`limit` doubles as the declared deadline in seconds for this probe; `depth`
is accepted for interface consistency but unused -- a single hung call
already demonstrates whether the bound is enforced, and a tree would add
delegation-tree noise without adding signal to this particular question.
"""
from __future__ import annotations

import time

from broods.frameworks.base import Capability
from broods.probes.base import Probe, ProbeResult

HANG_S = 20.0


class DeadlineProbe(Probe):
    name = "deadline"
    capability = Capability.WALL_CLOCK_BOUND
    key_metric = "observed_s"

    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, hang_s: float = HANG_S, **kw) -> ProbeResult:
        deadline_s = float(limit)
        if not framework.supports(self.capability):
            return ProbeResult(probe=self.name, framework=framework.name,
                               version=framework.version(), depth=depth,
                               supported=False, outcome="UNSUPPORTED",
                               error=f"{framework.name} exposes no native "
                                     f"wall-clock bound per invocation")

        if real:
            # Genuine model call, with controlled latency layered on top. A
            # real provider's own latency is neither large enough nor
            # steerable enough to test a deadline against -- without the
            # injected delay this would only measure "the model happened to
            # be fast", which says nothing about whether the bound is
            # enforced. delay_s is applied before the upstream dispatch, so
            # the round trip is real; only the wait is synthetic.
            self.proxy.reset(scenario="passthrough", delay_s=hang_s, cap=50,
                             api_key=framework.api_key)
        else:
            self.proxy.reset(scenario="hang", hang_s=hang_s, cap=50)
        agent = framework.make_agent(lambda t: "leaf", deadline_s=deadline_s)

        t0 = time.time()
        try:
            framework.invoke(agent, task)
            outcome, error = "COMPLETED", None
        except Exception as e:  # noqa: BLE001
            outcome, error = type(e).__name__, str(e)[:200]
        dur = time.time() - t0

        # Call count is what distinguishes the two failure modes: one very
        # slow call means the deadline cannot interrupt work already in
        # flight; several calls means it was not even checked between steps.
        stats = self.proxy.stats()
        return ProbeResult(
            probe=self.name, framework=framework.name, version=framework.version(),
            depth=depth, outcome=outcome, duration_s=round(dur, 3), error=error,
            metrics={"declared_deadline_s": deadline_s, "observed_s": round(dur, 3),
                     "overshoot_ratio": round(dur / deadline_s, 2),
                     "llm_calls": stats["calls"],
                     "injected_latency_s": hang_s})

    def verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        if not results or not results[0].supported:
            why = (results[0].error if results else "no data") or ""
            return "L0", f"deadline not expressible: {why[:60]}"
        base = results[0]
        m = base.metrics
        if not m.get("llm_calls"):
            return "ERR", (f"no LLM calls observed ({base.outcome}); "
                           f"probe did not execute")
        r = m["overshoot_ratio"]
        calls = m.get("llm_calls", 0)
        # Report WHAT was observed, not WHY. The internal mechanism is not
        # documented -- CrewAI specifies only "maximum time (in seconds) for
        # task execution" -- so any claim about where the check happens would
        # be inference presented as measurement.
        if r > 3.0:
            return "L1", (f"declared {m['declared_deadline_s']}s, observed "
                          f"{m['observed_s']}s ({r}x) across {calls} LLM calls")
        if r > 1.15:
            return "L2", (f"overshoot {r}x declared deadline "
                          f"across {calls} LLM calls")
        return "L3", f"deadline held within {r}x; composition untested"
