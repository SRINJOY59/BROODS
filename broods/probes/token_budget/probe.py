"""P6 -- aggregate token budget.

P1 shows every framework bounds STEPS and that the bound does not compose.
This probe asks the sharper question: can an operator bound the quantity that
actually costs money -- tokens -- and does THAT compose?

The distinction matters because a step counter is a poor proxy for spend.
One step at depth 3 costs more than one at depth 0, because context grows
within each agent's loop. A budget denominated in tokens is the only kind
that maps onto the thing being protected.

(An earlier version of this note claimed tokens outrun calls by ~1.22x. That
figure came from analyze.py's loose-file mode, which stitches rows together
across revisions; on a single-revision bundle with corrected instrumentation
the ratio is ~1.04, and peak per-request context is flat in depth. What
explodes is the number of agents, not the context each one carries.)

Across the eight systems tested, PydanticAI's `UsageLimits(total_tokens_limit=)`
is the only instance found. Everything else scores L0 -- the operator cannot
express the budget at all, and so cannot be protected by it.

Tokens are compared against what the proxy REPORTED to the client (a fixed
28/call), not the proxy's own character estimate, since that reported figure
is what a framework budgets against.
"""
from __future__ import annotations

import time

from broods.frameworks.base import Capability, Unsupported
from broods.probes.base import Probe, ProbeResult

TOKENS_PER_CALL = 28      # what the injector reports in every usage block


class TokenBudgetProbe(Probe):
    name = "token_budget"
    capability = Capability.TOKEN_BOUND
    key_metric = "observed_tokens"

    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, cap: int = 2000, **kw) -> ProbeResult:
        if not framework.supports(self.capability):
            return ProbeResult(probe=self.name, framework=framework.name,
                               version=framework.version(), depth=depth,
                               supported=False, outcome="UNSUPPORTED",
                               error=f"{framework.name} exposes no aggregate "
                                     f"token budget")

        # Budget expressed in tokens, sized so it should permit ~`limit` calls.
        budget = limit * TOKENS_PER_CALL
        if real:
            # This probe is synthetic by construction, so honouring
            # real= would be a lie: run_all tags every --real-probes row
            # real=true, and a bundle whose purpose is evidencing live
            # validation must not contain fault-injected runs labelled
            # as live ones.
            raise Unsupported(
                "token_budget does not support real=True; its scenario is fault-injected by construction")
        self.proxy.reset(scenario="always_delegate", cap=cap)
        agents_built = [0]

        def make(d: int):
            agents_built[0] += 1
            if d == 0:
                return framework.make_agent(lambda t: "leaf", token_limit=budget)

            def delegate(t: str) -> str:
                child = make(d - 1)
                try:
                    framework.invoke(child, t)
                except Unsupported:
                    raise
                except Exception as e:  # noqa: BLE001
                    return f"child stopped: {type(e).__name__}"
                return "child complete"
            return framework.make_agent(delegate, token_limit=budget)

        root = make(depth)
        t0 = time.time()
        try:
            framework.invoke(root, task)
            outcome, error = "COMPLETED", None
        except Exception as e:  # noqa: BLE001
            outcome, error = type(e).__name__, str(e)[:200]

        stats = self.proxy.stats()
        observed = stats["usage_tokens_reported"]
        return ProbeResult(
            probe=self.name, framework=framework.name, version=framework.version(),
            depth=depth, outcome=outcome, duration_s=round(time.time() - t0, 3),
            error=error, metrics={
                "declared_token_budget": budget,
                "observed_tokens": observed,
                "llm_calls": stats["calls"],
                "agents_built": agents_built[0],
                "overshoot_ratio": round(observed / budget, 2) if budget else 0,
                "capped": stats["capped"],
            })

    def verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        if not results or not results[0].supported:
            why = (results[0].error if results else "no data") or ""
            return "L0", f"token budget not expressible: {why[:60]}"
        base = results[0]
        if not base.metrics.get("llm_calls"):
            return "ERR", f"no LLM calls observed ({base.outcome})"
        b = base.metrics["declared_token_budget"]

        capped = [r.depth for r in results if r.metrics.get("capped")]
        if capped:
            return "L1", (f"safety cap hit at depth {capped[0]}; budget {b} "
                          f"tokens not enforced")
        if base.metrics["overshoot_ratio"] > 1.15:
            return "L2", (f"overshoot at depth {base.depth}: budget {b} tokens, "
                          f"spent {base.metrics['observed_tokens']}")

        series = [r.metrics["observed_tokens"] for r in results]
        if len(series) < 2:
            return "L3", "tight at node level; composition untested"
        deepest = results[-1]
        if deepest.metrics.get("agents_built", 1) > 1 and series[-1] <= series[0]:
            return "ERR", (f"{deepest.metrics['agents_built']} agents built but "
                           f"spend did not rise; children never executed")
        if series[-1] > series[0] * 1.5:
            return "L3", (f"budget holds per agent, but total spend grows "
                          f"{series[0]}->{series[-1]} tokens across depth "
                          f"{results[0].depth}->{deepest.depth}")
        return "L4", (f"token budget holds across delegation "
                      f"({series[0]}->{series[-1]} tokens)")
