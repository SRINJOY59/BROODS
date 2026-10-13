"""P11 -- request-rate bound.

Distinct from every other bound in the suite: a rate limit constrains
requests per unit time rather than requests in total. An agent can respect a
step count, a token budget and a deadline and still trip a provider's rate
limit, because those bound the integral and this bounds the derivative.

Three systems declare a bound of this kind: CrewAI's `max_rpm`, and -- found
only by the transitive introspection sweep, because both hang off the model
object rather than the agent -- LangChain's `InMemoryRateLimiter` under
LangGraph and LlamaIndex's `TokenBucketRateLimiter`.

The proxy computes the observed rate from its own arrival timestamps, so the
figure is the rate the provider would actually have seen.
"""
from __future__ import annotations

import time

from broods.frameworks.base import Capability, Unsupported
from broods.probes.base import Probe, ProbeResult

DECLARED_RPM = 12       # low enough that an unthrottled agent blows past it
# Requests per run. It must exceed any limiter's initial burst, or the limiter
# never acts and the probe measures nothing: CrewAI's max_rpm admits 12 per
# fixed minute, and LlamaIndex's token bucket starts full. An earlier run used
# 8, under both.
RATE_ITERS = 20


class RateLimitProbe(Probe):
    name = "rate_limit"
    capability = Capability.RATE_BOUND
    key_metric = "observed_rpm"

    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, cap: int = 2000, **kw) -> ProbeResult:
        if not framework.supports(self.capability):
            return ProbeResult(probe=self.name, framework=framework.name,
                               version=framework.version(), depth=depth,
                               supported=False, outcome="UNSUPPORTED",
                               error=f"{framework.name} exposes no request-rate "
                                     f"bound")

        # Enough iterations that an unthrottled agent issues them far faster
        # than DECLARED_RPM; a framework that honours the limit will instead
        # space them out and take proportionally longer.
        iters = max(limit, RATE_ITERS)
        # A framework may declare a rate bound and no iteration bound
        # (LlamaIndex). Passing `limit` there would raise Unsupported for the
        # wrong capability and report absence of the one we are testing.
        bounded = framework.supports(Capability.ITERATION_BOUND)
        lim_kw = {"limit": iters} if bounded else {}
        self.proxy.reset(scenario="always_delegate", cap=cap)
        agents = [0]

        def make(d: int):
            agents[0] += 1
            if d == 0:
                return framework.make_agent(lambda t: "leaf",
                                            rpm=DECLARED_RPM, **lim_kw)

            def delegate(t: str) -> str:
                child = make(d - 1)
                try:
                    framework.invoke(child, t)
                except Unsupported:
                    raise
                except Exception as e:  # noqa: BLE001
                    return f"child stopped: {type(e).__name__}"
                return "child complete"
            return framework.make_agent(delegate, rpm=DECLARED_RPM, **lim_kw)

        root = make(depth)
        t0 = time.time()
        try:
            framework.invoke(root, task)
            outcome, error = "COMPLETED", None
        except Exception as e:  # noqa: BLE001
            outcome, error = type(e).__name__, str(e)[:200]

        st = self.proxy.stats()
        # Elapsed time against the minimum the declared rate implies. Reporting
        # an extrapolated rpm instead is misleading: six requests completing in
        # 50 ms extrapolates to ~7000 rpm, a number that says more about the
        # window than the framework. Spacing n requests at r rpm takes at least
        # (n-1)*60/r seconds, and that comparison is stable at any n.
        n = st["calls"]
        implied_min_s = (max(0, n - 1)) * 60.0 / DECLARED_RPM
        # Floor AFTER rounding: round(1e-9, 3) is 0.0, so flooring first still
        # divided by zero whenever a framework issued fewer than two requests.
        observed_s = max(1e-9, round((st["t_last"] or 0) - (st["t_first"] or 0), 3))
        return ProbeResult(
            probe=self.name, framework=framework.name, version=framework.version(),
            depth=depth, outcome=outcome, duration_s=round(time.time() - t0, 3),
            error=error, metrics={
                "declared_rpm": DECLARED_RPM,
                "observed_rpm": st["observed_rpm"],
                "observed_s": observed_s,
                "implied_min_s": round(implied_min_s, 1),
                "speedup_over_limit": round(implied_min_s / observed_s, 1),
                "llm_calls": n, "agents_built": agents[0],
            })

    def verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        if not results or not results[0].supported:
            why = (results[0].error if results else "no data") or ""
            return "L0", f"request-rate bound not expressible: {why[:60]}"
        base = results[0]
        m = base.metrics
        if (m.get("llm_calls") or 0) < 2:
            # A rate needs at least two arrivals to exist.
            return "ERR", (f"only {m.get('llm_calls')} call(s) observed; "
                           f"rate not measurable")
        fast = m["speedup_over_limit"]
        interval = 60.0 / m["declared_rpm"]
        if m["observed_s"] < interval:
            # The whole run fit inside one declared interval: no request was
            # ever made to wait.
            return "L1", (f"issued {m['llm_calls']} requests in "
                          f"{m['observed_s']}s, under one {interval:.0f}s interval "
                          f"at {m['declared_rpm']} rpm; the limiter delayed none")
        if fast > 1.15:
            return "L2", (f"issued {m['llm_calls']} requests in "
                          f"{m['observed_s']}s; {m['declared_rpm']} rpm implies "
                          f"at least {m['implied_min_s']}s ({fast}x too fast)")
        if len(results) < 2:
            return "L3", (f"rate respected ({m['observed_s']}s vs "
                          f"{m['implied_min_s']}s implied); composition untested")
        deepest = results[-1]
        if deepest.metrics["speedup_over_limit"] > 1.15:
            return "L3", (f"held at depth 0 but ran "
                          f"{deepest.metrics['speedup_over_limit']}x too fast by "
                          f"depth {deepest.depth}")
        return "L4", "request rate holds across delegation"
