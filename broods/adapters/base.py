"""Adapter abstraction.

IMPORTANT -- this module must stay stdlib-only.

Adapters execute inside per-framework virtualenvs, because LangGraph, CrewAI
and AutoGen pin mutually incompatible versions of langchain-core, pydantic and
openai. Those venvs cannot import the `agentprobe` package. They can, however,
import a pure-stdlib module by path, which is how every adapter shares this
base class without any of them sharing a dependency tree.

What a subclass must supply is deliberately tiny:

    version()                     which build is under test
    make_agent(tool_fn, limit)    wrap a plain callable as this framework's
                                  agent, with `limit` declared however the
                                  framework expresses it
    invoke(agent, task, limit)    run it; raise if the framework stops it

Everything else -- the recursive delegation tree, timing, error capture, the
JSON envelope -- lives here. That matters for validity as much as for tidiness:
the tree's SHAPE is fixed by the harness rather than by each adapter, so the
usual objection to cross-framework comparison ("you implemented it
unidiomatically for framework X") cannot reach the structure being measured.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

ToolFn = Callable[[str], str]


class Unsupported(Exception):
    """The framework exposes no way to express what this probe needs.

    Raising this is a RESULT, not a bug: it produces the L0 row, meaning the
    operator cannot even ask for the guarantee.
    """


@dataclass
class ProbeConfig:
    probe: str = "spawn"
    depth: int = 0
    limit: int = 5
    base_url: str = "http://127.0.0.1:8099/v1"
    model: str = "mock"
    api_key: str = "unused"
    task: str = "Research a topic."

    @classmethod
    def from_stdin(cls) -> "ProbeConfig":
        raw = json.loads(sys.stdin.read() or "{}")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class Result:
    framework: str
    version: str = "?"
    supported: bool = True
    agents_built: int = 0
    outcome: str = "?"
    duration_s: float = 0.0
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def emit(self) -> None:
        # Frameworks are chatty on stdout, so the runner reads the LAST
        # parseable JSON line. Keeping this single-line is load-bearing.
        print(json.dumps(asdict(self)), flush=True)


class ProbeAdapter(ABC):
    """One framework under test."""

    name: str = "unknown"

    def __init__(self, cfg: ProbeConfig) -> None:
        self.cfg = cfg
        self.agents_built = 0

    # ------------------------------------------------------------ subclass API
    @abstractmethod
    def version(self) -> str:
        """Installed version of the framework under test."""

    @abstractmethod
    def make_agent(self, tool_fn: ToolFn, limit: int) -> Any:
        """Wrap `tool_fn` as a tool named `delegate` on a fresh agent.

        `limit` must be declared through whatever mechanism the framework
        documents for bounding agent execution -- recursion_limit, max_iter,
        max_tool_iterations. If the framework has no such mechanism, raise
        Unsupported rather than emulating one: the absence is the finding.
        """

    @abstractmethod
    def invoke(self, agent: Any, task: str, limit: int) -> None:
        """Run `agent` to completion, letting framework stops propagate."""

    def cancel(self, agent: Any) -> float:
        """Stop a running agent from outside. Override for the P3 probe."""
        raise Unsupported(f"{self.name} exposes no external cancellation")

    # --------------------------------------------------------------- provided
    def build_tree(self, depth: int, limit: int) -> Any:
        """An agent whose delegate tool spawns an agent one level shallower.

        Every agent at every level declares the SAME limit. The probe asks
        whether that limit binds the tree or merely the node.
        """
        self.agents_built += 1

        if depth == 0:
            return self.make_agent(lambda task: "leaf complete", limit)

        def delegate(task: str) -> str:
            child = self.build_tree(depth - 1, limit)
            try:
                self.invoke(child, task, limit)
            except Unsupported:
                raise
            except Exception as e:  # noqa: BLE001
                # A conscientious tool catches child failures. That entirely
                # reasonable habit is precisely what lets the parent continue
                # after the child was correctly stopped.
                return f"child stopped: {type(e).__name__}"
            return "child complete"

        return self.make_agent(delegate, limit)

    def execute(self) -> Result:
        r = Result(framework=self.name)
        try:
            r.version = self.version()
        except Exception:  # noqa: BLE001
            r.version = "?"

        t0 = time.time()
        try:
            root = self.build_tree(self.cfg.depth, self.cfg.limit)
            self.invoke(root, self.cfg.task, self.cfg.limit)
            r.outcome = "COMPLETED"
        except Unsupported as e:
            r.supported = False
            r.outcome = "UNSUPPORTED"
            r.error = str(e)
        except Exception as e:  # noqa: BLE001
            # The expected path: the framework enforced its bound and raised.
            r.outcome = type(e).__name__
            r.error = str(e)[:200]
        r.duration_s = round(time.time() - t0, 3)
        r.agents_built = self.agents_built
        return r


def run_adapter(cls: type[ProbeAdapter]) -> None:
    """Entrypoint: read config on stdin, emit one JSON result on stdout."""
    cfg = ProbeConfig.from_stdin()
    try:
        cls(cfg).execute().emit()
    except Exception as e:  # noqa: BLE001  -- import failure, bad env, etc.
        Result(framework=getattr(cls, "name", "unknown"),
               supported=False, outcome="ADAPTER_ERROR",
               error=f"{type(e).__name__}: {e}",
               extra={"traceback": traceback.format_exc()[-600:]}).emit()


def bootstrap() -> None:
    """Let an adapter import this module by path from any interpreter."""
    import pathlib
    d = str(pathlib.Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
