"""Observer + fault injector.

Runs as a standalone OpenAI-compatible server. Frameworks point their
`base_url` at it and cannot tell the difference. Two jobs:

  observer      counts every request that physically arrived, with timestamps.
                The framework cannot misreport this -- it is socket truth.

  fault injector  serves scripted responses so an agent provably never stops
                on its own. Any stop that happens is therefore attributable
                to the framework, which is the whole point of the measurement.

Control plane (not part of the OpenAI surface):
  POST /__reset  {"scenario": "...", "cap": 200}
  GET  /__stats  -> {"calls": n, "capped": bool, "events": [...], ...}

Run:  python -m broods.proxy --port 8099
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCENARIOS = (
    "always_delegate",  # every reply is a tool call -> agent can never finish
    "delegate_first",   # call #1 delegates, all later calls finish immediately
    "parallel_delegate",# call #1 asks for `fanout` tool calls AT ONCE, rest finish
    "finish",           # every reply is a terminal answer (control condition)
    "hang",             # sleep `hang_s`, then finish -> tests deadline enforcement
    "http_500",         # upstream error -> tests fault isolation
    "malformed",        # tool call with unparseable arguments
    "passthrough",      # forward to a real upstream provider
)

class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        # Must exist before reset(): reset deliberately carries inflight over
        # rather than zeroing it (stale requests still release against this
        # counter), so it reads the attribute it no longer creates.
        self.inflight = 0
        self.reset("always_delegate", 200)

    def reset(self, scenario: str, cap: int, hang_s: float = 20.0,
              fail_at: int | None = None, delay_s: float = 0.0,
              fail_status: int = 400, fail_repeat: int = 1,
              fail_match: str | None = None, fanout: int = 4,
              upstream: str | None = None, api_key: str | None = None) -> None:
        # Held across the whole mutation: a chat request arriving mid-reset
        # would otherwise observe a half-applied configuration.
        with self.lock:
            self._apply(scenario, cap, hang_s, fail_at, delay_s, fail_status,
                        fail_repeat, fail_match, fanout, upstream, api_key)

    def _apply(self, scenario, cap, hang_s, fail_at, delay_s, fail_status,
               fail_repeat, fail_match, fanout, upstream, api_key) -> None:
        self.scenario = scenario
        self.cap = cap
        self.hang_s = hang_s
        self.fail_at = fail_at       # target ONE call seq to fail, else scenario rules
        # 4xx by default, deliberately: the OpenAI SDK every framework wraps
        # retries 5xx automatically (max_retries=2), which silently repairs an
        # injected 500 and makes fault-isolation probes measure the HTTP
        # client's retry policy instead of the framework's behaviour.
        self.fail_status = fail_status
        # Fail this many CONSECUTIVE calls from fail_at. A single-shot
        # fault is silently repaired by client retry (LiteLLM retries even
        # 4xx), which makes a framework look fault-isolating when the
        # fault simply never reached it.
        self.fail_repeat = max(1, fail_repeat)
        # Fail by CONTENT rather than call sequence. Sequence numbers
        # assume a strictly ordered topology and silently mis-target the
        # moment anything retries or interleaves; a marker planted in the
        # child's own task text identifies that child no matter when it
        # calls, and automatically covers its retries too.
        self.fail_match = fail_match
        # How many tool calls a single reply requests. Frameworks that
        # execute tools concurrently will then have several children in
        # flight at once, which is the only way a concurrency bound can
        # come under pressure -- a sequential tree never overlaps.
        self.fanout = max(1, fanout)
        self.delay_s = delay_s       # per-call latency; without this the mock
                                      # responds so fast a cancellation probe
                                      # races nothing -- the tree finishes
                                      # naturally before cancel() ever fires
        self.upstream = upstream
        self.api_key = api_key
        self.events: list[dict] = []
        self.capped = False
        self.t_cancel: float | None = None
        # inflight is deliberately NOT zeroed. Requests from the previous
        # cell may still be in flight and will call release() against this
        # counter; zeroing here let a stale release steal a live request's
        # slot, so peak_inflight undercounted. That flips concurrency's
        # verdict outright, since peak <= 1 reports 'executes serially'.
        self.peak_inflight = self.inflight

    def record(self, body: dict) -> tuple[int, bool]:
        """Log one arriving request. Returns (seq, over_cap)."""
        msgs = body.get("messages", [])
        blob = json.dumps(msgs)
        # Real prompt size as it arrived on the wire. Serialise the WHOLE
        # message, not just `content`: an assistant message carrying a tool
        # call has content=None and its entire payload in `tool_calls`, so
        # summing `content` scored those at zero. Roughly half the messages
        # in a delegation tree are tool calls, so context was undercounted
        # by about half -- corrupting both the token-growth figures and the
        # per-request context bound probe.
        # Original note: call count alone
        # understates the true cost: context also grows within each agent's
        # loop, so tokens scale super-linearly in the number of calls.
        # ~4 chars/token is a coarse but consistent unit -- we only ever
        # compare these figures to each other, never to a provider's bill.
        prompt_chars = len(blob)
        with self.lock:
            seq = len(self.events) + 1
            self.events.append({
                "seq": seq,
                "t": time.time(),
                "n_messages": len(msgs),
                "prompt_chars": prompt_chars,
                "prompt_tokens_est": prompt_chars // 4,
                "after_cancel": self.t_cancel is not None,
            })
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
            over = seq > self.cap
            if over:
                self.capped = True
            return seq, over

    def release(self) -> None:
        """Mark one in-flight request finished (concurrency measurement)."""
        with self.lock:
            self.inflight = max(0, self.inflight - 1)

    def stats(self) -> dict:
        with self.lock:
            ev = list(self.events)
            scenario, capped = self.scenario, self.capped
            peak_inflight, t_cancel = self.peak_inflight, self.t_cancel
        after = [e for e in ev if e["after_cancel"]]
        return {
            "scenario": scenario,
            "calls": len(ev),
            "prompt_tokens_est": sum(e["prompt_tokens_est"] for e in ev),
            # What we TOLD the client it spent (28/call). Frameworks
            # budget against this, so it is the figure a token bound
            # must be compared against -- not our own char estimate.
            "usage_tokens_reported": len(ev) * USAGE_TOTAL,
            "peak_prompt_tokens_est": max((e["prompt_tokens_est"] for e in ev),
                                          default=0),
            "capped": capped,
            "peak_inflight": peak_inflight,
            # Requests per minute over the observed window. Meaningful only
            # when the window is long enough to matter; the rate probe sizes
            # its workload accordingly.
            "observed_rpm": (
                round(len(ev) / max(1e-9, (ev[-1]["t"] - ev[0]["t"])) * 60.0, 1)
                if len(ev) > 1 else 0.0
            ),
            "t_first": ev[0]["t"] if ev else None,
            "t_last": ev[-1]["t"] if ev else None,
            "t_cancel": t_cancel,
            "orphan_calls": len(after),
            "orphan_lifetime_s": (
                round(after[-1]["t"] - t_cancel, 3)
                if after and t_cancel else 0.0
            ),
            "events": ev,
        }


ST = State()


# Synthetic usage reported on every injected reply. Frameworks budget
# against these numbers, so the token probes must compare against the same
# constant -- it was previously duplicated as a literal 28 in stats().
USAGE_PROMPT, USAGE_COMPLETION = 20, 8
USAGE_TOTAL = USAGE_PROMPT + USAGE_COMPLETION


def _completion(seq: int, msg: dict, finish: str) -> bytes:
    return json.dumps({
        "id": f"chatcmpl-probe{seq}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "broods/fault-injector",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": USAGE_PROMPT,
                  "completion_tokens": USAGE_COMPLETION,
                  "total_tokens": USAGE_TOTAL},
    }).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a) -> None:  # keep the console clean
        pass

    def handle_one_request(self) -> None:
        # A cancelled agent drops its socket mid-flight. That is precisely
        # what P3 sets out to cause, so it is expected traffic rather than
        # a fault -- swallow it instead of dumping a traceback per request.
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    # ---------------------------------------------------------- control plane
    def do_GET(self) -> None:
        if self.path.startswith("/__stats"):
            return self._json(ST.stats())
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        if self.path.startswith("/__reset"):
            cfg = json.loads(raw or b"{}")
            fail_at = cfg.get("fail_at")
            ST.reset(cfg.get("scenario", "always_delegate"),
                     int(cfg.get("cap", 200)),
                     float(cfg.get("hang_s", 20.0)),
                     int(fail_at) if fail_at is not None else None,
                     float(cfg.get("delay_s", 0.0)),
                     int(cfg.get("fail_status", 400)),
                     int(cfg.get("fail_repeat", 1)),
                     cfg.get("fail_match"),
                     int(cfg.get("fanout", 4)),
                     cfg.get("upstream"), cfg.get("api_key"))
            return self._json({"ok": True, "scenario": ST.scenario})
        if self.path.startswith("/__quiesce"):
            # Let outstanding agents finish NOW without clearing counters.
            # P3 deliberately creates uncancellable orphans; without this they
            # keep calling for their full CHILD_LIMIT and poison the next cell.
            with ST.lock:
                ST.scenario, ST.delay_s, ST.fail_match = "finish", 0.0, None
            return self._json({"ok": True, "quiesced": True})
        if self.path.startswith("/__cancel"):
            # Accept the caller's timestamp. The proxy's own clock reading is
            # one HTTP round-trip later than the real cancel, and every orphan
            # request arriving in that window was scored as pre-cancel traffic
            # -- a bias that flatters cancellation.
            cfg = json.loads(raw or b"{}")
            with ST.lock:
                ST.t_cancel = float(cfg.get("t_cancel") or time.time())
            return self._json({"ok": True, "t_cancel": ST.t_cancel})
        if "/chat/completions" in self.path:
            return self._chat(raw)
        self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------- data plane
    def _chat(self, raw: bytes) -> None:
        # Parse and record BEFORE the try, so `finally` cannot release a
        # slot that was never acquired -- a malformed body would otherwise
        # decrement inflight without a matching increment and understate
        # observed concurrency.
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "malformed request"}, 400)
        seq, over = ST.record(body)
        try:
            self._chat_inner(body, raw, seq, over)
        finally:
            ST.release()

    def _chat_inner(self, body: dict, raw: bytes, seq: int,
                    over: bool) -> None:
        # One consistent snapshot. /__quiesce can rewrite scenario and
        # delay_s mid-request, which previously let a single request act on
        # a mixture of the old and new configuration.
        with ST.lock:
            sc, fail_match, fail_at = ST.scenario, ST.fail_match, ST.fail_at
            fail_repeat, fail_status = ST.fail_repeat, ST.fail_status
            delay_s, hang_s, fanout = ST.delay_s, ST.hang_s, ST.fanout
            upstream, api_key = ST.upstream, ST.api_key

        if over:  # safety backstop -- we are deliberately building runaways
            return self._raw(_completion(seq, {"role": "assistant",
                                               "content": "SAFETY_CAP"}, "stop"))

        if fail_match is not None and fail_match in json.dumps(
                body.get("messages", [])):
            return self._json({"error": {"message": "injected fault (matched)",
                                         "type": "invalid_request_error"}},
                              fail_status)

        if fail_at is not None and fail_at <= seq < fail_at + fail_repeat:
            return self._json({"error": {"message": "injected fault (targeted)",
                                         "type": "invalid_request_error"}},
                              fail_status)

        if sc == "http_500":
            return self._json({"error": {"message": "injected upstream failure",
                                         "type": "server_error"}}, 500)

        if delay_s and sc != "hang":      # hang already sleeps hang_s
            time.sleep(delay_s)

        if sc == "hang":
            time.sleep(hang_s)
            return self._raw(_completion(seq, {"role": "assistant",
                                               "content": "done"}, "stop"))

        if sc == "parallel_delegate" and seq == 1:
            tools = body.get("tools") or []
            name = "delegate"
            if tools:
                try:
                    name = tools[0]["function"]["name"]
                except Exception:  # noqa: BLE001
                    pass
            calls = [{"id": f"call_{seq}_{i}", "type": "function",
                      "function": {"name": name,
                                   "arguments": json.dumps({"task": f"branch-{i}"})}}
                     for i in range(fanout)]
            return self._raw(_completion(seq, {"role": "assistant",
                                               "content": None,
                                               "tool_calls": calls}, "tool_calls"))

        if sc == "finish" or (sc in ("delegate_first", "parallel_delegate")
                              and seq > 1):
            # delegate_first: only the root delegates. Every agent it spawns
            # completes on its first call, which makes the call sequence
            # deterministic (1 = parent, 2..N = children in creation order)
            # and lets fail_at target one specific child by position.
            return self._raw(_completion(seq, {"role": "assistant",
                                               "content": "done"}, "stop"))

        if sc == "passthrough":
            req = urllib.request.Request(
                upstream or "https://openrouter.ai/api/v1/chat/completions",
                data=raw, headers={"Authorization": f"Bearer {api_key}",
                                   "Content-Type": "application/json"})
            try:
                return self._raw(urllib.request.urlopen(req, timeout=180).read())
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 502)

        # always_delegate / malformed -- call whatever tool the framework
        # registered, so this works on any framework with no adaptation.
        tools = body.get("tools") or []
        name = "delegate"
        if tools:
            try:
                name = tools[0]["function"]["name"]
            except Exception:  # noqa: BLE001
                pass
        args = ("{not valid json" if sc == "malformed"
                else json.dumps({"task": "continue the work"}))
        return self._raw(_completion(seq, {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"call_{seq}", "type": "function",
                            "function": {"name": name, "arguments": args}}],
        }, "tool_calls"))

    # ------------------------------------------------------------------ utils
    def _json(self, obj: dict, code: int = 200) -> None:
        self._raw(json.dumps(obj).encode(), code)

    def _raw(self, payload: bytes, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionAbortedError):
            pass  # client cancelled mid-flight; expected during P3


def serve(port: int = 8099) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    serve(a.port)
    print(f"broods proxy on http://127.0.0.1:{a.port}/v1", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
