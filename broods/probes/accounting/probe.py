"""P7 -- cost accounting.

The OS analogue is `rusage`: after a process runs you can ask the kernel
exactly what it consumed, and the answer is authoritative because the kernel
mediated every allocation.

For agents the question splits in two:

  1. Does the framework report usage at all?
  2. If it does, is the figure CORRECT once the work spans delegation?

The second half is what makes this measurable rather than a survey. The
proxy counted every request that physically arrived, so it holds ground
truth the framework cannot talk its way around. A framework that reports
only its own top-level agent's usage will under-report by exactly the
subtree it spawned -- and under-reporting is the dangerous direction, since
it makes a runaway look affordable right up until the invoice arrives.
"""
from __future__ import annotations

import time

from broods.frameworks.base import Capability
from broods.frameworks.base import Unsupported
from broods.probes.base import Probe, ProbeResult

TOKENS_PER_CALL = 28


class AccountingProbe(Probe):
    name = "accounting"
    capability = None          # emergent: nothing is declared or opted into
    key_metric = "attribution_error_pct"

    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, cap: int = 2000, **kw) -> ProbeResult:
        # delegate_first, not always_delegate: the parent delegates once and
        # every agent then finishes normally. Under always_delegate each run
        # ends in the iteration bound, and a framework that raises cannot
        # return a usage object -- so "reports no usage" would have been an
        # artifact of forcing a failure, not a property of the framework.
        if real:
            # This probe is synthetic by construction, so honouring
            # real= would be a lie: run_all tags every --real-probes row
            # real=true, and a bundle whose purpose is evidencing live
            # validation must not contain fault-injected runs labelled
            # as live ones.
            raise Unsupported(
                "accounting does not support real=True; its scenario is fault-injected by construction")
        self.proxy.reset(scenario="delegate_first", cap=cap)
        # Request an iteration bound only where the framework has one.
        # These scenarios terminate on their own, so the bound is belt-and-
        # braces; demanding it unconditionally made every probe unrunnable
        # against a framework that simply lacks the parameter.
        lim = limit if framework.supports(Capability.ITERATION_BOUND) else None
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

        root = make(depth)
        t0 = time.time()
        try:
            framework.invoke(root, task)
            outcome, error = "COMPLETED", None
        except Exception as e:  # noqa: BLE001
            outcome, error = type(e).__name__, str(e)[:200]

        stats = self.proxy.stats()
        truth = stats["usage_tokens_reported"]        # socket-level ground truth
        reported = framework.report_usage(root)       # None if unsupported

        rep_tokens = (reported or {}).get("total_tokens")
        err_pct = (None if not truth or rep_tokens is None
                   else round(100.0 * (rep_tokens - truth) / truth, 1))

        return ProbeResult(
            probe=self.name, framework=framework.name, version=framework.version(),
            depth=depth, outcome=outcome, duration_s=round(time.time() - t0, 3),
            error=error, metrics={
                "reports_usage": reported is not None,
                "reported_tokens": rep_tokens,
                "actual_tokens": truth,
                "llm_calls": stats["calls"],
                "agents_built": agents_built[0],
                "attribution_error_pct": err_pct,
            })

    def verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        if not results:
            return "L0", "no data"
        base = results[0]
        m = base.metrics
        if not m.get("llm_calls"):
            return "ERR", f"no LLM calls observed ({base.outcome})"
        if not m["reports_usage"]:
            return "L0", "framework reports no usage accounting"

        # Accuracy at depth 0 is the easy case; the question is whether the
        # figure survives delegation.
        deep = [r for r in results if r.depth > 0 and
                r.metrics.get("attribution_error_pct") is not None]
        worst = max((abs(r.metrics["attribution_error_pct"]) for r in deep),
                    default=abs(m.get("attribution_error_pct") or 0))
        # Order matters. With `default=` falling back to the depth-0 error,
        # an accurate framework yields worst==0 and the L4 branch fired
        # first, claiming "accurate across delegation" from a run that never
        # delegated. Check for the absence of deep rows FIRST.
        if not deep:
            return "L3", f"accurate at depth 0 ({worst}%); composition untested"
        if worst <= 5:
            return "L4", f"usage accurate within {worst}% across delegation"
        d = deep[-1]
        return "L1", (f"under-reports spend by {abs(d.metrics['attribution_error_pct'])}% "
                      f"at depth {d.depth}: reported {d.metrics['reported_tokens']} "
                      f"vs {d.metrics['actual_tokens']} actually spent")
