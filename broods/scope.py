"""TaskScope -- the missing abstraction, retrofitted from outside.

Every failure in the matrix traces to one absence: no object denotes "a task
and everything it spawns", so there is nothing for a limit to bind, a usage
total to sum over, or a cancellation to reach. This module supplies that
object for ANY framework without touching framework code, by gating the one
thing every framework must do to spend money: send an HTTP request to a
chat-completions endpoint.

    with TaskScope(calls=5, tokens=None, deadline_s=None) as s:
        run_the_whole_agent_tree()
    s.report()   # calls, tokens, refused, ... charged to the task

Mechanism:
  * The current scope lives in a contextvar. Scopes nest; a charge is
    checked against, and debited from, every ancestor (cgroup semantics).
  * install() wraps Client.send and AsyncClient.send in httpx AND httpx2
    (the fork openai>=3 moved to), and urllib's OpenerDirector.open. A chat request is refused BEFORE it leaves
    the process once any enclosing budget is spent, the scope is cancelled,
    or its deadline has passed; otherwise its timeouts are clamped to the
    time remaining, so a deadline interrupts an in-flight call instead of
    waiting for it (the 4x overshoot in the matrix).
  * Usage is read from each response and charged to the scope that issued
    the request -- accounting by construction rather than by each
    framework's own bookkeeping.

The research question is where the contextvar FAILS to follow the work.
asyncio tasks copy it; threads and thread-pool workers do not. With
propagate_threads=True, threading.Thread and ThreadPoolExecutor.submit are
wrapped to carry the creator's context across, the same technique tracing
libraries use. A request made with no scope in context is counted in
UNSCOPED -- attribution that escaped -- rather than silently allowed.

Stdlib only at import; httpx/httpx2 are patched only where the venv has them.
"""
from __future__ import annotations

import contextvars
import io
import json
import threading
import time
from typing import Any

_current: contextvars.ContextVar["TaskScope | None"] = contextvars.ContextVar(
    "broods_task_scope", default=None)
_LOCK = threading.RLock()
UNSCOPED = {"calls": 0}          # chat requests issued with no scope in context
_INSTALLED = {"http": False, "threads": False}


class ScopeRefused(Exception):
    """A chat request was refused by an enclosing TaskScope."""


class ScopeExhausted(ScopeRefused):
    pass


class ScopeCancelled(ScopeRefused):
    pass


class ScopeDeadline(ScopeRefused):
    pass


def _is_chat(url: str) -> bool:
    return "/chat/completions" in url


class TaskScope:
    def __init__(self, calls: int | None = None, tokens: int | None = None,
                 deadline_s: float | None = None, name: str = "task") -> None:
        self.name = name
        self.budget_calls, self.budget_tokens = calls, tokens
        self.deadline_s = deadline_s
        self.parent: TaskScope | None = None
        self.calls = self.tokens = self.refused = 0
        self.refused_by: dict[str, int] = {}
        self.cancelled = False
        self.t_start = self.t_deadline = None
        self._token = None

    # ------------------------------------------------------------ lifecycle
    def __enter__(self) -> "TaskScope":
        self.parent = _current.get()
        self.t_start = time.time()
        if self.deadline_s is not None:
            self.t_deadline = self.t_start + self.deadline_s
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc) -> None:
        _current.reset(self._token)

    def cancel(self) -> None:
        """Cancel this scope and, through the ancestor check, everything
        charged to it: no request issued under it leaves the process again."""
        with _LOCK:
            self.cancelled = True

    # -------------------------------------------------------------- charges
    def _chain(self):
        s = self
        while s is not None:
            yield s
            s = s.parent

    def admit(self) -> float | None:
        """Check every ancestor, then debit one call. Returns seconds left
        before the tightest deadline in the chain, or None if none."""
        now = time.time()
        with _LOCK:
            left = None
            for s in self._chain():
                reason = None
                if s.cancelled:
                    reason, exc = "cancelled", ScopeCancelled
                elif s.t_deadline is not None and now >= s.t_deadline:
                    reason, exc = "deadline", ScopeDeadline
                elif s.budget_calls is not None and s.calls >= s.budget_calls:
                    reason, exc = "calls", ScopeExhausted
                elif s.budget_tokens is not None and s.tokens >= s.budget_tokens:
                    reason, exc = "tokens", ScopeExhausted
                if reason:
                    self.refused += 1
                    self.refused_by[reason] = self.refused_by.get(reason, 0) + 1
                    raise exc(f"TaskScope[{s.name}] refused request: {reason}")
                if s.t_deadline is not None:
                    r = s.t_deadline - now
                    left = r if left is None else min(left, r)
            for s in self._chain():
                s.calls += 1
            return left

    def expired(self) -> bool:
        now = time.time()
        return any(s.t_deadline is not None and now >= s.t_deadline
                   for s in self._chain())

    def note_refusal(self, reason: str) -> None:
        with _LOCK:
            self.refused += 1
            self.refused_by[reason] = self.refused_by.get(reason, 0) + 1

    def charge_tokens(self, n: int) -> None:
        with _LOCK:
            for s in self._chain():
                s.tokens += n

    def report(self) -> dict[str, Any]:
        return {"calls": self.calls, "tokens": self.tokens,
                "refused": self.refused, "refused_by": dict(self.refused_by),
                "budget_calls": self.budget_calls,
                "budget_tokens": self.budget_tokens,
                "deadline_s": self.deadline_s, "cancelled": self.cancelled,
                "unscoped_calls": UNSCOPED["calls"],
                "propagate_threads": _INSTALLED["threads"]}


def current() -> TaskScope | None:
    return _current.get()


# ----------------------------------------------------------------- gating
def _gate(url: str):
    """-> (scope, seconds_left, refusal). A refusal is returned, not raised,
    so each client can surface it in the form that client does not retry."""
    if not _is_chat(url):
        return None, None, None
    s = _current.get()
    if s is None:
        with _LOCK:
            UNSCOPED["calls"] += 1
        return None, None, None
    try:
        return s, s.admit(), None
    except ScopeRefused as e:
        return s, None, e


def _refusal_response(httpx, request, err: Exception):
    # A 400, deliberately. Raising inside send() surfaces as a connection
    # error, and the OpenAI SDK retries those after a backoff sleep -- which
    # put a 1.18x overshoot on a deadline the scope had already enforced.
    # 4xx other than 408/409/429 is not retried by any client here.
    return httpx.Response(400, request=request, json={"error": {
        "message": str(err), "type": "invalid_request_error",
        "code": "task_scope_refused"}})


def _usage_tokens(body: bytes) -> int:
    """total_tokens from a completion, or from the usage chunk of an SSE
    stream. Skipping streamed replies made the scope blind to LlamaIndex's
    and Strands' spend: -100% accounting on a framework that streams."""
    try:
        u = json.loads(body).get("usage") or {}
        return int(u.get("total_tokens") or 0)
    except Exception:  # noqa: BLE001
        pass
    total = 0
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
            continue
        try:
            u = json.loads(line[5:]).get("usage") or {}
        except Exception:  # noqa: BLE001
            continue
        total = max(total, int(u.get("total_tokens") or 0))
    return total


def _clamp_httpx_timeout(request, left: float | None) -> None:
    if left is None:
        return
    left = max(0.001, left)
    t = dict(request.extensions.get("timeout") or {})
    for k in ("connect", "read", "write", "pool"):
        v = t.get(k)
        t[k] = left if v is None else min(v, left)
    request.extensions["timeout"] = t


def _install_http() -> None:
    # httpx AND httpx2: openai>=3 moved to the httpx2 fork, so five of the
    # seven frameworks here send through httpx2 while two still use httpx.
    # Patching only one made the scope silently blind to every request from
    # the others -- zero calls charged, zero refused, -100% accounting.
    import importlib
    for modname in ("httpx", "httpx2"):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        _patch_httpx_like(mod)
    _install_urllib()


def _patch_httpx_like(httpx) -> None:
    o_sync, o_async = httpx.Client.send, httpx.AsyncClient.send

    def send(self, request, *a, **k):
        s, left, refusal = _gate(str(request.url))
        if refusal is not None:
            return _refusal_response(httpx, request, refusal)
        _clamp_httpx_timeout(request, left)
        try:
            resp = o_sync(self, request, *a, **k)
        except Exception:
            # The clamped timeout fired: the deadline interrupted this call.
            if s is not None and s.expired():
                s.note_refusal("deadline_inflight")
                return _refusal_response(httpx, request,
                                         ScopeDeadline("deadline reached in flight"))
            raise
        if s is not None:
            s.charge_tokens(_usage_tokens(resp.read()))
        return resp

    async def asend(self, request, *a, **k):
        s, left, refusal = _gate(str(request.url))
        if refusal is not None:
            return _refusal_response(httpx, request, refusal)
        _clamp_httpx_timeout(request, left)
        try:
            resp = await o_async(self, request, *a, **k)
        except Exception:
            if s is not None and s.expired():
                s.note_refusal("deadline_inflight")
                return _refusal_response(httpx, request,
                                         ScopeDeadline("deadline reached in flight"))
            raise
        if s is not None:
            s.charge_tokens(_usage_tokens(await resp.aread()))
        return resp

    httpx.Client.send, httpx.AsyncClient.send = send, asend


def _install_urllib() -> None:
    import urllib.request
    import urllib.response
    o_open = urllib.request.OpenerDirector.open

    def open_(self, fullurl, data=None, timeout=None, **k):
        url = fullurl if isinstance(fullurl, str) else fullurl.full_url
        s, left, refusal = _gate(url)
        if refusal is not None:
            raise refusal            # urllib has no retry layer to avoid
        if left is not None:
            timeout = left if timeout is None else min(timeout, max(0.001, left))
        args = (self, fullurl, data) if timeout is None else (self, fullurl, data, timeout)
        resp = o_open(*args, **k)
        if s is None:
            return resp
        body = resp.read()
        s.charge_tokens(_usage_tokens(body))
        return urllib.response.addinfourl(io.BytesIO(body), resp.headers,
                                          resp.url, resp.status)

    urllib.request.OpenerDirector.open = open_


def _install_threads() -> None:
    """Carry the creator's context into threads and pool tasks."""
    import concurrent.futures as cf
    # _bootstrap_inner, not run(): it is what calls run(), so subclasses that
    # override run() (as thread pools and many frameworks do) are covered too.
    o_init, o_boot = threading.Thread.__init__, threading.Thread._bootstrap_inner

    def init(self, *a, **k):
        o_init(self, *a, **k)
        self._broods_ctx = contextvars.copy_context()

    def boot(self):
        ctx = getattr(self, "_broods_ctx", None)
        return ctx.run(o_boot, self) if ctx is not None else o_boot(self)

    threading.Thread.__init__, threading.Thread._bootstrap_inner = init, boot

    o_submit = cf.ThreadPoolExecutor.submit

    def submit(self, fn, /, *a, **k):
        ctx = contextvars.copy_context()
        return o_submit(self, ctx.run, fn, *a, **k)

    cf.ThreadPoolExecutor.submit = submit


def install(propagate_threads: bool = True) -> None:
    with _LOCK:
        if not _INSTALLED["http"]:
            _install_http()
            _INSTALLED["http"] = True
        if propagate_threads and not _INSTALLED["threads"]:
            _install_threads()
            _INSTALLED["threads"] = True
