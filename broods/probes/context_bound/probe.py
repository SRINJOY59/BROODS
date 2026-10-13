"""P9 -- per-request context bound.

Distinct from token_budget. That probe asks whether CUMULATIVE spend can be
bounded; this asks whether the size of any SINGLE request can be, which is
the quantity with no clean operating-system analogue. Exceeding a memory
limit raises a fault; exceeding a context window silently degrades the
answer, so an unbounded context is a correctness problem rather than only a
cost one.

PydanticAI's `per_request_input_tokens_limit` is the only declared bound of
this kind across the eight systems tested. CrewAI's `respect_context_window`
is related but different in kind -- it truncates automatically rather than
letting the operator state a limit -- so it is noted in the paper rather than
scored here.

Measurement uses the proxy's peak observed prompt size, which is the real
size on the wire rather than anything the framework reports.

The proxy reports each request's own size as its prompt usage here, rather
than the fixed 20 tokens the other probes use. PydanticAI checks this limit
against the usage in each response, so under fixed usage it could never fire
and an earlier run scored a bound that had not been exercised.
"""
from __future__ import annotations

import time

from broods.frameworks.base import Capability, Unsupported
from broods.probes.base import Probe, ProbeResult


class ContextBoundProbe(Probe):
    name = "context_bound"
    capability = Capability.CONTEXT_BOUND
    key_metric = "peak_prompt_tokens"

    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, cap: int = 2000, **kw) -> ProbeResult:
        if not framework.supports(self.capability):
            return ProbeResult(probe=self.name, framework=framework.name,
                               version=framework.version(), depth=depth,
                               supported=False, outcome="UNSUPPORTED",
                               error=f"{framework.name} exposes no per-request "
                                     f"context bound")

        # Small enough that a growing conversation crosses it quickly.
        ctx_limit = 60
        self.proxy.reset(scenario="always_delegate", cap=cap, usage="estimate")
        agents = [0]

        def make(d: int):
            agents[0] += 1
            if d == 0:
                return framework.make_agent(lambda t: "leaf" * 40,
                                            limit=limit, context_limit=ctx_limit)

            def delegate(t: str) -> str:
                child = make(d - 1)
                try:
                    framework.invoke(child, t)
                except Unsupported:
                    raise
                except Exception as e:  # noqa: BLE001
                    return f"child stopped: {type(e).__name__}"
                return "child complete " + "pad" * 40
            return framework.make_agent(delegate, limit=limit,
                                        context_limit=ctx_limit)

        root = make(depth)
        t0 = time.time()
        try:
            framework.invoke(root, task)
            outcome, error = "COMPLETED", None
        except Exception as e:  # noqa: BLE001
            outcome, error = type(e).__name__, str(e)[:200]

        st = self.proxy.stats()
        peak = st["peak_prompt_tokens_est"]
        return ProbeResult(
            probe=self.name, framework=framework.name, version=framework.version(),
            depth=depth, outcome=outcome, duration_s=round(time.time() - t0, 3),
            error=error, metrics={
                "declared_context_limit": ctx_limit,
                "declared_limit": limit,
                "peak_prompt_tokens": peak,
                "overshoot_ratio": round(peak / ctx_limit, 2) if ctx_limit else 0,
                "llm_calls": st["calls"], "agents_built": agents[0],
            })

    def verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        if not results or not results[0].supported:
            why = (results[0].error if results else "no data") or ""
            return "L0", f"per-request context bound not expressible: {why[:60]}"
        base = results[0]
        m = base.metrics
        if not m.get("llm_calls"):
            return "ERR", f"no LLM calls observed ({base.outcome})"
        r = m["overshoot_ratio"]
        lim = m.get("declared_limit")
        if r > 1.15 and lim and m["llm_calls"] >= lim:
            # The run lasted as long as its iteration bound allows, so the
            # context bound never stopped it.
            return "L1", (f"peak request carried {m['peak_prompt_tokens']} tokens "
                          f"against a declared cap of "
                          f"{m['declared_context_limit']} and the run ended at its "
                          f"iteration bound ({m['llm_calls']} calls)")
        if r > 1.15:
            return "L2", (f"peak request carried {m['peak_prompt_tokens']} tokens "
                          f"against a declared cap of "
                          f"{m['declared_context_limit']} ({r}x)")
        if len(results) < 2:
            return "L3", f"held within {r}x at depth 0; composition untested"
        deepest = results[-1]
        if deepest.metrics["overshoot_ratio"] > 1.15:
            return "L3", (f"held at depth 0 but reached "
                          f"{deepest.metrics['overshoot_ratio']}x by depth "
                          f"{deepest.depth}")
        return "L4", f"per-request context stayed within {r}x across delegation"
