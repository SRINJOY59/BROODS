"""Framework abstraction -- the X axis of the matrix (probe is the Y axis).

A FrameworkAdapter is a thin, uniform handle onto one agent framework: build
an agent around a single tool, run it, and -- where the framework allows --
start it without blocking and stop it from outside. Every probe is written
against this interface and never imports a specific framework, which is what
lets one probe run unmodified against every framework.

Must stay stdlib-only: adapters live one per venv (langgraph/crewai/autogen
pin incompatible dependencies) and this module is imported by path into
whichever venv is currently running.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

ToolFn = Callable[[str], str]

# One instruction, identical for every framework.
#
# This must not vary between adapters. An earlier version left LangGraph and
# the baseline with no system prompt at all while CrewAI and AutoGen were both
# told "always delegate before answering" -- so a real-model comparison was
# measuring my prompts as much as the frameworks. It showed up exactly as you
# would expect: CrewAI delegated and made 3 calls, the two with no instruction
# answered immediately in 1.
#
# Synthetic runs are unaffected either way (the fault injector never reads the
# prompt), but the text must still be uniform for real-model runs to compare
# anything meaningful. Each adapter places it in that framework's native slot.
SYSTEM_PROMPT = (
    "You are a research agent. Break the task into subtasks and delegate each "
    "one using the delegate tool before answering."
)
# Live experiments (broods/experiments/branching.py) vary the instruction to
# separate "the prompt told it to delegate" from "the model chose to". Read
# once at import, in every venv, so all adapters still see ONE uniform text.
import os as _os
SYSTEM_PROMPT = _os.environ.get("BROODS_SYSTEM_PROMPT") or SYSTEM_PROMPT


class Capability(str, Enum):
    """A guarantee a framework may or may not expose a way to declare.

    Absence is data, not an error -- see Unsupported.

    NOTE: compare with == or `in`, never `is`. Adapters import this module by
    path (as `base`) while probes import it as `broods.frameworks.base`,
    so the two sides hold DIFFERENT class objects for the same enum and
    identity comparison silently returns False. Deriving from str keeps
    value equality working across that boundary.
    """
    ITERATION_BOUND = "iteration_bound"     # max_iter / recursion_limit / max_tool_iterations
    WALL_CLOCK_BOUND = "wall_clock_bound"   # a native per-invocation deadline
    EXTERNAL_CANCEL = "external_cancel"     # stoppable from outside once running
    # A budget denominated in TOKENS rather than steps. Distinct from
    # ITERATION_BOUND because it constrains the quantity that actually costs
    # money: a step counter bounds neither the size of each step nor the
    # number of agents taking them. PydanticAI's total_tokens_limit is the
    # only instance found across the systems tested.
    TOKEN_BOUND = "token_bound"
    # A cap on the size of any SINGLE request's context, as opposed to
    # cumulative spend. This is the quantity with no clean OS analogue:
    # exceeding it degrades correctness rather than raising a fault.
    # PydanticAI's per_request_input_tokens_limit is the only instance found.
    CONTEXT_BOUND = "context_bound"
    # A cap on how many operations may be in flight at once -- the thread-pool
    # analogue. LangGraph's max_concurrency is the only instance found.
    CONCURRENCY_BOUND = "concurrency_bound"
    # A cap on request rate rather than total count. CrewAI's max_rpm is the
    # only instance found.
    RATE_BOUND = "rate_bound"


class Unsupported(Exception):
    """The framework exposes no way to express what the probe needs.

    Raising this (or supports() returning False) produces the L0 row: the
    operator cannot even ask for the guarantee, which is itself the finding.
    """


@dataclass
class AgentHandle:
    """Opaque wrapper around a framework's native agent object.

    Probes never touch `.obj` directly -- they only pass the handle back
    into the same adapter's invoke() / start() / cancel().
    """
    obj: Any


@dataclass
class RunHandle:
    """A started, not-yet-finished invocation.

    `.obj` carries whatever start() used to represent it -- an asyncio Task,
    a thread, a future -- framework-specific and opaque to probes.
    """
    obj: Any
    started_at: float


class FrameworkAdapter(ABC):
    """One agent framework under test."""

    name: str = "unknown"

    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self.model, self.base_url, self.api_key = model, base_url, api_key
        self.agents_built = 0

    # --------------------------------------------------------------- identity
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def supports(self, cap: Capability) -> bool: ...

    # ---------------------------------------------------------------- static
    @abstractmethod
    def make_agent(self, tool_fn: ToolFn, *, limit: int | None = None,
                   deadline_s: float | None = None,
                   token_limit: int | None = None,
                   context_limit: int | None = None,
                   concurrency: int | None = None,
                   rpm: int | None = None) -> AgentHandle:
        """Build an agent whose only tool is `tool_fn`, named `delegate`.

        `limit` must be declared through whatever the framework documents as
        its iteration/step bound; `deadline_s` through its wall-clock bound.
        Pass only the one(s) the probe actually needs. If a value is passed
        for a bound the framework cannot express, raise Unsupported rather
        than silently ignoring it -- a silently-ignored limit would look
        identical to an enforced one in the results.

        `token_limit` is an AGGREGATE budget denominated in tokens rather
        than steps. `context_limit` bounds a SINGLE request's context;
        `concurrency` bounds simultaneous operations; `rpm` bounds request
        rate. Each is exposed by at most one framework in the suite --
        everywhere else these must raise Unsupported.
        """

    @abstractmethod
    def invoke(self, handle: AgentHandle, task: str) -> None:
        """Run to completion. Let the framework's own stop propagate as an
        exception; do not swallow it here."""

    # --------------------------------------------------------- non-blocking
    def start(self, handle: AgentHandle, task: str) -> RunHandle:
        """Begin `invoke` without blocking. Only needed by probes that must
        act on a still-running agent (cancellation)."""
        raise Unsupported(f"{self.name}: no non-blocking invocation")

    def cancel(self, run: RunHandle) -> float:
        """Stop a started run from outside. Returns the cancel timestamp."""
        raise Unsupported(f"{self.name}: no external cancellation")

    def join(self, run: RunHandle, timeout: float = 30.0) -> None:
        """Best-effort wait for a (possibly cancelled) run to actually stop."""
        raise Unsupported(f"{self.name}: no non-blocking invocation")

    # ------------------------------------------------------------ accounting
    def report_usage(self, handle: AgentHandle) -> dict | None:
        """Token usage this framework attributes to `handle`, if it offers any.

        The OS analogue is rusage: after a process exits you can ask what it
        consumed. Return None when the framework exposes no usage accounting
        -- that absence is the measurement, so do not synthesise a figure
        from our own bookkeeping.
        """
        return None
