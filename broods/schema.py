"""Core types for the conformance suite.

A probe asks one question about one OS-level guarantee and returns a Verdict.
The distinction between UNSUPPORTED and UNCONTAINED is the point of the whole
suite: "the framework has no way to express this limit" is a different (and
more damning) finding than "the limit exists but does not hold".
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Verdict = Literal[
    "CONTAINED",    # limit was declared and held
    "PARTIAL",      # limit held, but only after overshoot (overdraft / leakage)
    "UNCONTAINED",  # limit was declared and did not hold
    "UNSUPPORTED",  # framework exposes no way to declare this limit at all
    "ERROR",        # probe itself failed; not a finding about the framework
]

CAPABILITIES = {
    "spawn":       "Bounded recursive sub-agent spawning",
    "budget":      "Token/cost quota enforced before spend",
    "timeout":     "Wall-clock deadline that actually terminates",
    "fault":       "Sibling fault isolation",
    "toolfault":   "Tool failure containment",
    "state":       "Cross-agent state isolation",
    "accounting":  "Per-agent cost attribution",
    "kill":        "External termination of a running agent",
    "orphan":      "Cancellation propagates to children",
}


@dataclass
class Limits:
    """What we ask the framework to enforce. None = not requested."""
    max_depth: int | None = None
    max_agents: int | None = None
    max_tokens: int | None = None
    max_steps: int | None = None
    wall_clock_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ProbeResult:
    probe_id: str
    capability: str
    framework: str
    framework_version: str
    verdict: Verdict
    # Quantitative evidence, e.g. {"requested_max_depth": 3, "observed_depth": 47}
    metrics: dict[str, Any] = field(default_factory=dict)
    # One-line human summary that goes straight into the paper.
    finding: str = ""
    # Raw excerpt (stack trace, log tail) for the appendix.
    evidence: str = ""
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


@dataclass
class MockCall:
    """One request seen by the mock model server: the accounting ground truth."""
    seq: int
    agent_hint: str | None      # agent identity, if the framework leaked one
    prompt_tokens: int
    completion_tokens: int
    scenario: str
    t: float
