"""The level each probe assigns to representative measurements.

The cases mirror cells of the paper's Table 2, so a change to a verdict rule
that would move a published level fails here first.
"""
import pytest

from broods.probes import REGISTRY
from broods.probes.base import ProbeResult


def level(probe, *rows, supported=True):
    results = [ProbeResult(probe=probe, framework="x", depth=d, supported=supported,
                           metrics=m) for d, m in rows]
    return REGISTRY[probe](None).safe_verdict(results)[0]


def spawn(calls, agents, limit=5, capped=False):
    return {"llm_calls": calls, "agents_built": agents, "declared_limit": limit,
            "capped": capped}


def test_unsupported_is_l0():
    assert level("spawn", (0, {}), supported=False) == "L0"


@pytest.mark.parametrize("series, expected", [
    ([(5, 1), (30, 6), (155, 31)], "L3"),   # per-run bound, total grows
    ([(6, 1), (36, 6)], "L3"),              # one terminal call is within slack
    # A deeper tree with no extra traffic is refused, not credited (the
    # "Refusals" rule of Appendix D): children may simply never have run.
    ([(5, 1), (5, 3)], "ERR"),
    ([(5, 1), (5, 6)], "ERR"),
])
def test_spawn(series, expected):
    rows = [(d, spawn(c, a)) for d, (c, a) in enumerate(series)]
    assert level("spawn", *rows) == expected


def test_spawn_safety_cap_is_l1():
    assert level("spawn", (0, spawn(1001, 1, capped=True))) == "L1"


def token(spent, calls, agents=1, capped=False, budget=140):
    return {"declared_token_budget": budget, "observed_tokens": spent,
            "llm_calls": calls, "agents_built": agents, "capped": capped,
            "overshoot_ratio": round(spent / budget, 2)}


def test_token_cap_at_any_depth_is_l1():
    # CrewAI: max_tokens never fired, and depth 1 hit the safety cap.
    assert level("token_budget", (0, token(728, 26)),
                 (1, token(11256, 402, agents=16, capped=True))) == "L1"


def test_token_one_reply_late_is_l2():
    # PydanticAI: stopped at 168 against 140.
    assert level("token_budget", (0, token(168, 6)), (1, token(1008, 36, 6))) == "L2"


def test_token_per_agent_is_l3():
    # AutoGen: holds per agent, grows 140 -> 840.
    assert level("token_budget", (0, token(140, 5)), (1, token(840, 30, 6))) == "L3"


def conc(peak, fanout=6, cap=2):
    return {"declared_concurrency": cap, "requested_fanout": fanout,
            "peak_inflight": peak, "overshoot_ratio": round(peak / cap, 2),
            "llm_calls": 8}


@pytest.mark.parametrize("peak, expected", [(6, "L1"), (3, "L2"), (2, "L3"), (1, "N/A")])
def test_concurrency(peak, expected):
    assert level("concurrency", (0, conc(peak))) == expected


def rate(n, seconds, rpm=12):
    implied = (n - 1) * 60 / rpm
    seconds = max(1e-9, seconds)    # the probe floors elapsed time the same way
    return {"declared_rpm": rpm, "observed_s": seconds, "implied_min_s": implied,
            "speedup_over_limit": round(implied / seconds, 1), "llm_calls": n}


@pytest.mark.parametrize("n, seconds, expected", [
    (9, 0.18, "L1"),     # the limiter delayed no request
    (21, 60.5, "L2"),    # CrewAI: a fixed window lets a burst through
    (10, 45.2, "L3"),    # LangGraph: a smooth rate
    (1, 0.0, "ERR"),     # a rate needs two arrivals
])
def test_rate(n, seconds, expected):
    assert level("rate_limit", (0, rate(n, seconds))) == expected


def ctx(peak, calls, limit=5, cap=60):
    return {"declared_context_limit": cap, "declared_limit": limit,
            "peak_prompt_tokens": peak, "overshoot_ratio": round(peak / cap, 2),
            "llm_calls": calls}


def test_context_bound_that_never_stopped_the_run_is_l1():
    assert level("context_bound", (0, ctx(447, 5))) == "L1"


def test_context_bound_checked_after_the_response_is_l2():
    assert level("context_bound", (0, ctx(150, 2)), (1, ctx(150, 4))) == "L2"


def cancel(orphans):
    return {"children_spawned": 12, "orphan_calls": orphans,
            "orphan_lifetime_s": 5.0, "observe_window_s": 6.0, "total_calls": 40}


@pytest.mark.parametrize("orphans, expected", [(69, "L1"), (0, "L4")])
def test_cancellation(orphans, expected):
    assert level("cancellation", (0, cancel(orphans))) == expected


def acct(err, reported=56, actual=84):
    return {"reports_usage": True, "reported_tokens": reported,
            "actual_tokens": actual, "llm_calls": 3, "agents_built": 2,
            "attribution_error_pct": err}


@pytest.mark.parametrize("deep_err, expected", [(-33.3, "L1"), (0.0, "L4")])
def test_accounting(deep_err, expected):
    assert level("accounting", (0, acct(0.0, 56, 56)), (1, acct(deep_err))) == expected
