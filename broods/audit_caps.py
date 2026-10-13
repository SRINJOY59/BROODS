"""Capability audit: does a framework expose a bound we did not look for?

A \\lz{} verdict is a claim of absence, and six of eight adapters asserted it
from a hand-written list. This script replaces assertion with introspection.
For each framework it imports the classes the adapter actually builds on,
enumerates every parameter of their constructors and run methods, and matches
those names against a keyword set per capability.

The sweep is transitive: a parameter whose annotation is itself a class in
the framework's own package is expanded too, to a bounded depth, because a
bound reachable only through a nested configuration object would otherwise
read as absence. Output is JSON on stdout: for each capability, every
parameter name in the framework's own surface that plausibly implements it,
plus ``surface_size`` and the full ``surface`` so that an absence can be
checked by reading rather than by trusting the keyword set. Cross-referencing
against what the adapter sets (Appendix D) yields the candidate false L0s.

Run inside a framework's own venv:
    .venvs/<fw>/Scripts/python broods/audit_caps.py <fw>
"""
from __future__ import annotations

import inspect
import json
import sys
import warnings

warnings.filterwarnings("ignore")

# Capability -> substrings that would name a mechanism implementing it.
# Deliberately broad: a false positive costs one line of manual checking,
# a false negative is exactly the error this script exists to catch.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "ITERATION_BOUND":  ("max_iter", "max_step", "max_turn", "recursion",
                         "max_loop", "max_round", "max_cycle", "iteration_limit",
                         "step_limit", "request_limit", "max_tool_iter"),
    "WALL_CLOCK_BOUND": ("timeout", "deadline", "max_time", "max_execution",
                         "time_limit", "max_duration", "expires"),
    "TOKEN_BOUND":      ("token", "usage", "budget", "max_cost", "spend"),
    "CONTEXT_BOUND":    ("context_window", "max_input", "input_token",
                         "max_prompt", "context_limit", "per_request"),
    "CONCURRENCY_BOUND":("concurren", "parallel", "max_workers", "semaphore",
                         "max_inflight", "max_in_flight"),
    "RATE_BOUND":       ("rpm", "rate_limit", "requests_per", "qps",
                         "throttle", "max_rpm"),
    "EXTERNAL_CANCEL":  ("cancel", "abort", "stop_signal", "interrupt",
                         "shutdown"),
}

# The classes each adapter actually builds on, plus the methods that carry
# per-run options. Import paths are lazy so one missing framework does not
# abort the sweep.
TARGETS: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
    "langgraph": [
        ("langgraph.prebuilt", "create_react_agent", ("__call__",)),
        ("langgraph.graph", "StateGraph", ("compile", "add_node")),
    ],
    "crewai": [
        ("crewai", "Agent", ("kickoff",)),
        ("crewai", "Crew", ("kickoff",)),
        ("crewai", "Task", ()),
    ],
    "autogen": [
        ("autogen_agentchat.agents", "AssistantAgent", ("run", "run_stream")),
        ("autogen_agentchat.teams", "RoundRobinGroupChat", ("run",)),
    ],
    "pydantic_ai": [
        ("pydantic_ai", "Agent", ("run", "run_sync")),
        ("pydantic_ai.usage", "UsageLimits", ()),
    ],
    "openai_agents": [
        ("agents", "Agent", ()),
        ("agents", "Runner", ("run", "run_sync")),
        ("agents", "RunConfig", ()),
    ],
    "smolagents": [
        ("smolagents", "ToolCallingAgent", ("run",)),
        ("smolagents", "CodeAgent", ("run",)),
    ],
    "llamaindex": [
        ("llama_index.core.agent.workflow", "FunctionAgent", ("run",)),
        ("llama_index.core.agent.workflow", "ReActAgent", ("run",)),
    ],
}


# Top-level package each framework's own types live in. Only annotations that
# resolve into these packages are expanded, so we recurse into the framework's
# own configuration objects and not into the standard library or pydantic.
ROOT_PKG: dict[str, tuple[str, ...]] = {
    "langgraph":      ("langgraph", "langchain_core"),
    "crewai":         ("crewai",),
    "autogen":        ("autogen_agentchat", "autogen_core"),
    "pydantic_ai":    ("pydantic_ai",),
    "openai_agents":  ("agents",),
    "smolagents":     ("smolagents",),
    "llamaindex":     ("llama_index",),
}

MAX_DEPTH = 2


def _params_of(obj) -> list[str]:
    try:
        return list(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return []


def _pydantic_fields(cls) -> list[str]:
    """Pydantic models hide their real surface behind **kwargs."""
    for attr in ("model_fields", "__fields__"):
        f = getattr(cls, attr, None)
        if isinstance(f, dict):
            return list(f)
    return []


def _annotations_of(obj) -> dict:
    """Declared types of a callable's parameters or a model's fields."""
    out = {}
    try:
        for name, par in inspect.signature(obj).parameters.items():
            if par.annotation is not inspect.Parameter.empty:
                out[name] = par.annotation
    except (TypeError, ValueError):
        pass
    for attr in ("model_fields", "__fields__"):
        f = getattr(obj, attr, None)
        if isinstance(f, dict):
            for name, fi in f.items():
                ann = getattr(fi, "annotation", None)
                if ann is not None:
                    out[name] = ann
    return out


def _candidate_types(ann, roots: tuple[str, ...]) -> list:
    """Classes inside `ann` (unwrapping Optional/Union/list/etc.) that belong
    to the framework's own package and so are worth expanding."""
    out = []
    stack = [ann]
    while stack:
        a = stack.pop()
        if a is None:
            continue
        args = getattr(a, "__args__", None)
        if args:
            stack.extend(args)
            continue
        if isinstance(a, str):          # unresolved forward reference
            continue
        mod = getattr(a, "__module__", "") or ""
        if inspect.isclass(a) and mod.split(".")[0] in roots:
            out.append(a)
    return out


def surface(fw: str) -> dict:
    found: dict[str, set[str]] = {}
    every: set[str] = set()
    seen: list[str] = []
    errors: list[str] = []
    roots = ROOT_PKG.get(fw, (fw,))

    def record(owner: str, names: dict) -> None:
        for n in names:
            every.add(f"{owner}.{n}")
            low = n.lower()
            for cap, keys in KEYWORDS.items():
                if any(k in low for k in keys):
                    found.setdefault(cap, set()).add(f"{owner}.{n}")

    def expand(cls, owner: str, depth: int, visited: set) -> None:
        key = f"{getattr(cls, '__module__', '')}.{getattr(cls, '__name__', cls)}"
        if key in visited or depth > MAX_DEPTH:
            return
        visited.add(key)
        anns = _annotations_of(cls)
        names = set(_params_of(cls)) | set(_pydantic_fields(cls)) | set(anns)
        record(owner, {n: None for n in names})
        if depth == MAX_DEPTH:
            return
        for pname, ann in anns.items():
            for sub in _candidate_types(ann, roots):
                expand(sub, f"{owner}.{pname}:{sub.__name__}", depth + 1, visited)

    for mod_name, cls_name, methods in TARGETS.get(fw, []):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{mod_name}.{cls_name}: {type(e).__name__}")
            continue
        seen.append(f"{mod_name}.{cls_name}")

        visited: set = set()
        expand(cls, cls_name, 0, visited)
        for m in methods:
            fn = getattr(cls, m, None)
            if fn is None:
                continue
            anns = _annotations_of(fn)
            record(cls_name, {n: None for n in set(_params_of(fn)) | set(anns)})
            for pname, ann in anns.items():
                for sub in _candidate_types(ann, roots):
                    expand(sub, f"{cls_name}.{m}.{pname}:{sub.__name__}", 1, visited)

    return {
        "framework": fw,
        "classes_introspected": seen,
        "import_errors": errors,
        "max_depth": MAX_DEPTH,
        "surface_size": len(every),
        "capabilities": {k: sorted(v) for k, v in sorted(found.items())},
        "surface": sorted(every),
    }


if __name__ == "__main__":
    fw = sys.argv[1] if len(sys.argv) > 1 else ""
    print(json.dumps(surface(fw), indent=1))
