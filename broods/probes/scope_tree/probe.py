"""Workloads for evaluating TaskScope (broods/scope.py) on every framework.

The matrix probes each require the framework's OWN mechanism -- deadline
returns UNSUPPORTED on a framework with no timeout, cancellation on one with
no cancel hook. A TaskScope is meant to supply exactly those guarantees from
outside, so its evaluation needs workloads that do not depend on them:

  spawn     the always_delegate tree of the spawn probe. The framework's own
            per-agent bound is declared where one exists (so the comparison
            with the matrix is like for like) and omitted where none does.
  deadline  one agent against a hung provider, no framework deadline.
  cancel    a parent that fans out to children running in harness threads
            and returns; "cancel" is proxy.cancel(), which the entrypoint
            binds to TaskScope.cancel() when a scope is active. Without a
            scope nothing stops the children -- the control condition.

Run with and without a scope (cfg["scope"]); the probe itself is identical.
"""
from __future__ import annotations

import threading
import time

from broods.frameworks.base import Capability, Unsupported
from broods.probes.base import Probe, ProbeResult

HANG_S = 20.0
FANOUT, CHILD_LIMIT = 3, 30
DELAY_S, SPAWN_WAIT_S, SETTLE_S, OBSERVE_S = 1.0, 20.0, 4.0, 6.0


class ScopeTreeProbe(Probe):
    name = "scope_tree"
    key_metric = "llm_calls"

    def _lim(self, framework, n):
        return n if framework.supports(Capability.ITERATION_BOUND) else None

    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, mode: str = "spawn", cap: int = 400,
            **kw) -> ProbeResult:
        base = dict(probe=self.name, framework=framework.name,
                    version=framework.version(), depth=depth)
        t0 = time.time()
        if mode == "spawn":
            m, outcome, err = self._spawn(framework, depth, limit, task, cap)
        elif mode == "deadline":
            m, outcome, err = self._deadline(framework, limit, task)
        elif mode == "cancel":
            m, outcome, err = self._cancel(framework, limit, task)
        else:
            raise ValueError(mode)
        m["mode"] = mode
        return ProbeResult(**base, outcome=outcome, error=err,
                           duration_s=round(time.time() - t0, 3), metrics=m)

    # ---------------------------------------------------------------- spawn
    def _spawn(self, framework, depth, limit, task, cap):
        self.proxy.reset(scenario="always_delegate", cap=cap)
        built = [0]

        def make(d):
            built[0] += 1
            if d == 0:
                return framework.make_agent(lambda t: "leaf complete",
                                            limit=self._lim(framework, limit))

            def delegate(t):
                child = make(d - 1)
                try:
                    framework.invoke(child, t)
                except Unsupported:
                    raise
                except Exception as e:  # noqa: BLE001
                    return f"child stopped: {type(e).__name__}"
                return "child complete"
            return framework.make_agent(delegate, limit=self._lim(framework, limit))

        root = make(depth)
        try:
            framework.invoke(root, task)
            outcome, err = "COMPLETED", None
        except Exception as e:  # noqa: BLE001
            outcome, err = type(e).__name__, str(e)[:200]
        st = self.proxy.stats()
        return ({"llm_calls": st["calls"], "agents_built": built[0],
                 "capped": st["capped"], "declared_limit": limit,
                 "framework_bound_declared": self._lim(framework, limit) is not None,
                 "usage_tokens_socket": st["usage_tokens_reported"]},
                outcome, err)

    # ------------------------------------------------------------- deadline
    def _deadline(self, framework, limit, task):
        self.proxy.reset(scenario="hang", hang_s=HANG_S, cap=50)
        # A task deadline runs from when the TASK starts, which under a scope
        # is scope entry -- before agent construction. Timing from invoke()
        # instead scored PydanticAI's 2.6 s of construction as slack, a 0.37x
        # "overshoot" on a deadline that in fact held at 1.0x.
        from broods import scope as _scope
        sc = _scope.current()
        t_task = sc.t_start if sc is not None and sc.t_start else time.time()
        agent = framework.make_agent(lambda t: "leaf",
                                     limit=self._lim(framework, 5))
        t0 = time.time()
        try:
            framework.invoke(agent, task)
            outcome, err = "COMPLETED", None
        except Exception as e:  # noqa: BLE001
            outcome, err = type(e).__name__, str(e)[:200]
        dur = time.time() - t0
        task_dur = time.time() - t_task
        st = self.proxy.stats()
        return ({"observed_s": round(dur, 3), "declared_deadline_s": float(limit),
                 "observed_since_task_start_s": round(task_dur, 3),
                 # control: framework deadlines run from invoke(); scoped: from
                 # task start. Each is the clock that deadline is defined on.
                 "overshoot_ratio": round((task_dur if sc is not None else dur)
                                          / float(limit), 2),
                 "llm_calls": st["calls"], "injected_latency_s": HANG_S},
                outcome, err)

    # --------------------------------------------------------------- cancel
    def _cancel(self, framework, limit, task):
        self.proxy.reset(scenario="always_delegate", cap=5000, delay_s=DELAY_S)
        threads: list[threading.Thread] = []

        def run_child(i, t):
            child = framework.make_agent(lambda _t: "leaf",
                                         limit=self._lim(framework, CHILD_LIMIT))
            try:
                framework.invoke(child, f"{t}-{i}")
            except Exception:  # noqa: BLE001 -- refused or stopped
                pass

        def parent_tool(t):
            if threads:                       # fan out once only
                return "already spawned"
            for i in range(FANOUT):
                th = threading.Thread(target=run_child, args=(i, t), daemon=True)
                threads.append(th)
                th.start()
            return f"spawned {FANOUT} children"

        root = framework.make_agent(parent_tool, limit=self._lim(framework, limit))

        def run_root():
            try:
                framework.invoke(root, task)
            except Exception:  # noqa: BLE001
                pass
        rt = threading.Thread(target=run_root, daemon=True)
        rt.start()

        end = time.time() + SPAWN_WAIT_S
        while not threads and time.time() < end:
            time.sleep(0.1)
        if not threads:
            self.proxy.quiesce()
            return ({"children_spawned": 0,
                     "total_calls": self.proxy.stats()["calls"]},
                    "NO_FANOUT", "parent did not fan out")
        time.sleep(SETTLE_S)
        before = self.proxy.stats()["calls"]
        self.proxy.cancel(time.time())        # scope-bound when a scope is active
        time.sleep(OBSERVE_S)
        st = self.proxy.stats()
        self.proxy.quiesce()
        for th in threads + [rt]:
            th.join(timeout=5.0)
        return ({"children_spawned": len(threads), "calls_before_cancel": before,
                 "orphan_calls": st["orphan_calls"],
                 "orphan_lifetime_s": st["orphan_lifetime_s"],
                 "total_calls": st["calls"], "observe_window_s": OBSERVE_S},
                "ORPHANED" if st["orphan_calls"] else "CLEAN", None)

    def verdict(self, results):
        return "N/A", "evaluated by experiments/scope_eval.py"
