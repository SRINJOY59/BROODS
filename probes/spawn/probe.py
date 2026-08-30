"""P1 -- spawn bound.

Does a declared iteration/step limit bound the number of LLM calls under
delegation, or only the calls made by a single node?

Method: every agent in a delegation tree of depth `depth` declares the SAME
limit. The proxy never lets an agent finish on its own (always_delegate), so
any stop that happens is attributable entirely to the framework. The tree's
shape is fixed here, in framework-agnostic code, so no adapter can be
implemented "unidiomatically" in a way that changes what is being measured.
"""
from __future__ import annotations

import time

from broods.frameworks.base import Capability, Unsupported
from broods.probes.base import Probe, ProbeResult


class SpawnProbe(Probe):
    name = "spawn"
    capability = Capability.ITERATION_BOUND
    key_metric = "llm_calls"

    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, cap: int = 1000, **kw) -> ProbeResult:
        if not framework.supports(self.capability):
            return ProbeResult(probe=self.name, framework=framework.name,
                               version=framework.version(), depth=depth,
                               supported=False, outcome="UNSUPPORTED",
                               error=f"{framework.name} exposes no iteration bound")

        if real:
            # Organic behaviour: the model decides whether to delegate. No
            # forced worst case, so these numbers validate rather than
            # replace the deterministic ones -- expect them to differ.
            self.proxy.reset(scenario="passthrough", cap=cap,
                             api_key=framework.api_key)
        else:
            self.proxy.reset(scenario="always_delegate", cap=cap)
        agents_built = [0]

        def make(d: int):
            agents_built[0] += 1
            if d == 0:
                return framework.make_agent(lambda t: "leaf complete", limit=limit)

            def delegate(t: str) -> str:
                child = make(d - 1)
                try:
                    framework.invoke(child, t)
                except Unsupported:
                    raise
                except Exception as e:  # noqa: BLE001
                    # A conscientious tool catches child failures -- that
                    # entirely reasonable habit is what lets the parent
                    # continue after the child was correctly stopped.
                    return f"child stopped: {type(e).__name__}"
                return "child complete"
            return framework.make_agent(delegate, limit=limit)

        root = make(depth)
        t0 = time.time()
        try:
            framework.invoke(root, task)
            outcome, error = "COMPLETED", None
        except Exception as e:  # noqa: BLE001
            outcome, error = type(e).__name__, str(e)[:200]

        stats = self.proxy.stats()
        return ProbeResult(
            probe=self.name, framework=framework.name, version=framework.version(),
            depth=depth, outcome=outcome, duration_s=round(time.time() - t0, 3),
            error=error, metrics={
                "agents_built": agents_built[0], "llm_calls": stats["calls"],
                "declared_limit": limit, "capped": stats["capped"],
                # Cost is driven by tokens, not calls. Tracking both shows
                # that context growth compounds the call growth rather than
                # merely tracking it.
                "prompt_tokens_est": stats["prompt_tokens_est"],
                "peak_prompt_tokens_est": stats["peak_prompt_tokens_est"],
            })

    def verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        if not results or not results[0].supported:
            why = (results[0].error if results else "no data") or ""
            return "L0", f"limit not expressible: {why[:60]}"

        base = results[0]                    # shallowest depth = node-level case
        if not base.metrics.get("llm_calls"):
            # Zero calls means the adapter never reached the proxy. Without
            # this guard an adapter that raises on startup scores L4, because
            # "no calls" is indistinguishable from "perfectly bounded".
            return "ERR", (f"no LLM calls observed ({base.outcome}); "
                           f"probe did not execute")
        lim = base.metrics["declared_limit"]
        if base.metrics["capped"]:
            return "L1", (f"safety cap reached at depth {base.depth}: "
                          f"declared {lim}, observed {base.metrics['llm_calls']}+")
        slack = max(1, round(0.15 * lim))    # tolerate a constant final-answer call
        if base.metrics["llm_calls"] > lim + slack:
            return "L2", (f"overshoot at depth {base.depth}: declared {lim}, "
                          f"observed {base.metrics['llm_calls']}")

        series = [r.metrics["llm_calls"] for r in results]
        if len(series) < 2:
            return "L3", "tight at node level; composition untested"

        # A deeper tree that produced no additional traffic means the children
        # never ran -- an adapter fault, not containment. PydanticAI hit this:
        # nested run_sync raised, the tool reported "child stopped", and the
        # flat call count looked exactly like a perfectly composing budget.
        deepest = results[-1]
        if deepest.metrics.get("agents_built", 1) > 1 and series[-1] <= series[0]:
            return "ERR", (f"{deepest.metrics['agents_built']} agents built at "
                           f"depth {deepest.depth} but calls did not rise above "
                           f"depth {results[0].depth} ({series[0]}); children "
                           f"never executed")
        if series[-1] > series[0] * 1.5:
            return "L3", (f"tight per agent, but aggregate grows "
                          f"{series[0]}->{series[-1]} across depth "
                          f"{results[0].depth}->{results[-1].depth}")
        return "L4", "bound holds and survives delegation"
