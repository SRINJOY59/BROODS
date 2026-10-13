"""Probe abstraction -- one containment question, run against any framework.

A Probe owns: which Capability it exercises (or None, for probes that
measure emergent behaviour rather than a declared limit), how to arm the
proxy for its fault, how to build whatever agent topology it needs using
only the FrameworkAdapter interface, and how to turn a depth series of
results into a capability level.

Must stay stdlib-only, for the same reason as frameworks/base.py: probe
modules get imported into whichever framework's venv is currently running.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

LEVELS = {
    "L0": "not expressible -- no API to declare this guarantee",
    "L1": "declared, not enforced -- observed behaviour is unbounded",
    "L2": "enforced with overshoot -- holds, but only after exceeding it",
    "L3": "enforced tightly at the node, but does not survive delegation",
    "L4": "enforced and composes -- still holds across delegation",
}


@dataclass
class ProbeResult:
    probe: str
    framework: str
    version: str = "?"
    depth: int = 0
    supported: bool = True
    outcome: str = "?"
    duration_s: float = 0.0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def emit_to(self, path: str | None) -> None:
        """Write to the runner's result file (and stdout as a fallback)."""
        import json as _json
        import pathlib as _pl
        payload = _json.dumps(asdict(self))
        if path:
            try:
                _pl.Path(path).write_text(payload, encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
        print(payload, flush=True)


class Probe(ABC):
    name: str
    capability: Any = None   # a frameworks.base.Capability, or None
    # Metric the runner takes the median of across repeated trials. Real-model
    # runs vary run to run, so a single trial is not a measurement.
    key_metric: str = ""

    def __init__(self, proxy) -> None:   # proxy: proxy_client.ProxyClient
        self.proxy = proxy

    @abstractmethod
    def run(self, framework, depth: int, limit: int, task: str,
            real: bool = False, **kw) -> ProbeResult:
        """`real=True` routes the proxy's scenario to `passthrough` -- genuine
        network round trips to `framework.api_key`'s provider, with the model
        deciding organically whether to delegate, rather than the injector
        forcing worst-case behaviour. Probes whose scenario never dispatches
        upstream regardless (deadline's `hang`) may ignore this flag."""

    def safe_verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        """verdict() with harness failures filtered out.

        run_adapter's failure rows carry `supported: True` and `metrics: {}`
        (a timeout says nothing about whether the framework supports the
        capability), so `if not supported: break` does not fire and an empty
        row reaches verdict(). Every verdict guards results[0] but indexes
        results[-1] directly, so one timed-out deeper row raised KeyError and
        took down the entire matrix -- exactly the outcome run_adapter's
        docstring promises to avoid.

        Rows with no metrics are dropped and named in the rationale, so a
        partial result is reported as partial rather than silently trimmed.
        """
        # A row is a HARNESS failure only if it claims support yet carries no
        # metrics. An UNSUPPORTED row legitimately has empty metrics -- the
        # framework cannot express the capability, which is the L0 finding and
        # must reach verdict(). Filtering on `not r.metrics` alone turned every
        # honest L0 into ERR, inverting the claim from "cannot express this" to
        # "we failed to measure".
        usable = [r for r in results if r.metrics or not r.supported]
        dropped = [r for r in results if r.metrics is not None
                   and not r.metrics and r.supported]
        if not usable:
            outs = ", ".join(sorted({r.outcome for r in results})) or "no rows"
            return "ERR", f"no usable rows ({outs})"
        level, why = self.verdict(usable)
        if dropped:
            lost = ", ".join(f"d{r.depth}:{r.outcome}" for r in dropped)
            why = f"{why} [incomplete: {lost}]"
        return level, why

    @abstractmethod
    def verdict(self, results: list[ProbeResult]) -> tuple[str, str]:
        """Capability level (see LEVELS) + one-line rationale, computed
        across every depth this probe was run at for one framework."""
