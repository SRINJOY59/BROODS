import json

from broods.proxy import USAGE_COMPLETION, USAGE_TOTAL

TOOLS = [{"type": "function", "function": {"name": "delegate", "parameters": {}}}]
MSGS = [{"role": "user", "content": "Research a topic."}]


def test_always_delegate_answers_with_a_tool_call(proxy, chat):
    proxy.reset(scenario="always_delegate", cap=10)
    reply = chat(MSGS, TOOLS)
    choice = reply["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "delegate"
    assert reply["usage"]["total_tokens"] == USAGE_TOTAL


def test_socket_count_and_reported_usage(proxy, chat):
    proxy.reset(scenario="always_delegate", cap=10)
    for _ in range(3):
        chat(MSGS, TOOLS)
    st = proxy.stats()
    assert st["calls"] == 3
    assert st["usage_tokens_reported"] == 3 * USAGE_TOTAL


def test_estimate_usage_reports_request_size(proxy, chat):
    proxy.reset(scenario="always_delegate", cap=10, usage="estimate")
    reply = chat(MSGS, TOOLS)
    expected = len(json.dumps(MSGS)) // 4
    assert reply["usage"]["prompt_tokens"] == expected
    assert reply["usage"]["total_tokens"] == expected + USAGE_COMPLETION


def test_delegate_first_lets_only_the_root_delegate(proxy, chat):
    proxy.reset(scenario="delegate_first", cap=10)
    first, second = chat(MSGS, TOOLS), chat(MSGS, TOOLS)
    assert first["choices"][0]["finish_reason"] == "tool_calls"
    assert second["choices"][0]["finish_reason"] == "stop"


def test_safety_cap_stops_a_runaway(proxy, chat):
    proxy.reset(scenario="always_delegate", cap=2)
    replies = [chat(MSGS, TOOLS) for _ in range(3)]
    assert replies[-1]["choices"][0]["message"]["content"] == "SAFETY_CAP"
    assert proxy.stats()["capped"] is True
