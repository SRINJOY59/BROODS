"""Do deadlines and cancellation cross a protocol boundary?

Inside one process a TaskScope can follow the work (broods/scope.py). Once an
agent delegates to another agent over a protocol -- MCP, A2A -- no in-process
object can: whatever the callee learns about its caller's deadline or
cancellation must travel in the protocol. gRPC is the precedent: it puts the
deadline on the wire (the grpc-timeout header) on every call.

Topology: client -> hop 1 -> hop 2, each hop a separate server process on a
real transport (MCP streamable HTTP, A2A JSON-RPC over HTTP, gRPC over
HTTP/2). Each hop runs a simulated agent -- N_CALLS "model calls" to a local
observer, DELAY_S apart -- and hop 1 delegates to hop 2 concurrently. Every
hop honours whatever cancellation signal its protocol delivers to it, so the
only variable is propagation. The client gives up after GIVE_UP_S either by
timing out or by explicitly cancelling; the observer then counts, per hop,
model calls that ARRIVE after that instant: orphaned work.

    .venvs/protocols/Scripts/python.exe -m broods.experiments.protocols
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N_CALLS, DELAY_S = 40, 0.5
GIVE_UP_S, OBSERVE_S = 3.0, 8.0
OBS_PORT = 8140
DEADLINE_WORDS = ("deadline", "timeout", "expire", "time_remaining", "ttl")


def log(ev: dict) -> None:
    path = os.environ.get("BROODS_PROTO_EVENTS")
    if path:
        # One file per process: concurrent appends from two processes to one
        # file lose lines on Windows, which read as "no cancel delivered".
        with open(f"{path}.{os.getpid()}", "a", encoding="utf-8") as f:
            f.write(json.dumps({**ev, "t": time.time()}, default=str) + "\n")


# ------------------------------------------------------------ observer
class Observer:
    """Stands in for the model provider: counts calls per hop, with times."""

    def __init__(self, port: int) -> None:
        self.records: list[tuple[float, str]] = []
        self.lock = threading.Lock()
        obs = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def handle_one_request(self):
                # A cancelled caller drops its socket mid-call; that is the
                # traffic this experiment provokes, not an error.
                try:
                    super().handle_one_request()
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    self.close_connection = True

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                with obs.lock:
                    obs.records.append((time.time(), self.headers.get("X-Hop", "?")))
                time.sleep(DELAY_S)
                body = json.dumps({"choices": [{"message": {"role": "assistant",
                                                            "content": "ok"}}]}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def reset(self) -> None:
        with self.lock:
            self.records.clear()

    def after(self, t: float) -> dict[str, int]:
        with self.lock:
            out: dict[str, int] = {}
            for ts, hop in self.records:
                if ts > t:
                    out[hop] = out.get(hop, 0) + 1
            return out

    def total(self) -> dict[str, int]:
        return self.after(0.0)


OBS_URL = f"http://127.0.0.1:{OBS_PORT}/v1/chat/completions"


async def agent_async(hop: int) -> int:
    import httpx
    async with httpx.AsyncClient(timeout=30) as c:
        for i in range(N_CALLS):
            await c.post(OBS_URL, headers={"X-Hop": str(hop)}, json={"i": i})
    return N_CALLS


def agent_sync(hop: int, alive) -> int:
    import httpx
    with httpx.Client(timeout=30) as c:
        for i in range(N_CALLS):
            if not alive():
                return i
            c.post(OBS_URL, headers={"X-Hop": str(hop)}, json={"i": i})
    return N_CALLS


def _deadline_seen(blob) -> bool:
    s = json.dumps(blob, default=str).lower()
    return any(w in s for w in DEADLINE_WORDS)


# ================================================================== MCP
def mcp_server(hop: int, port: int, downstream: str | None) -> None:
    import anyio
    from mcp.server.mcpserver import Context, MCPServer
    # Annotations are strings under `from __future__ import annotations`;
    # MCPServer resolves them against module globals.
    globals()["Context"] = Context

    srv = MCPServer(f"hop{hop}")

    async def call_down() -> None:
        from mcp.client.client import Client
        # Hop 1 forwards everything it knows about its caller's deadline --
        # which, over MCP, is nothing: no field carries one.
        async with Client(downstream) as c:
            await c.call_tool("work", {"task": "subtask"})

    @srv.tool()
    async def work(task: str, ctx: Context) -> str:
        rc = ctx.request_context
        params = dict(rc.params or {})
        meta = rc.meta.model_dump() if hasattr(rc.meta, "model_dump") else rc.meta
        log({"proto": "mcp", "hop": hop, "event": "received",
             "param_keys": sorted(params), "meta": meta,
             "deadline_seen": _deadline_seen({"p": params, "m": meta})})
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(agent_async, hop)
                if downstream:
                    tg.start_soon(call_down)
        except BaseException as e:
            log({"proto": "mcp", "hop": hop, "event": "cancelled",
                 "exc": type(e).__name__})
            raise
        log({"proto": "mcp", "hop": hop, "event": "finished"})
        return "done"

    srv.run("streamable-http", host="127.0.0.1", port=port)


async def mcp_client(url: str, mode: str) -> float:
    from mcp.client.client import Client
    async with Client(url) as c:
        if mode == "timeout":
            try:
                await c.call_tool("work", {"task": "root"},
                                  read_timeout_seconds=GIVE_UP_S)
            except Exception:  # noqa: BLE001 -- the timeout is the point
                pass
            return time.time()
        t = asyncio.create_task(c.call_tool("work", {"task": "root"}))
        await asyncio.sleep(GIVE_UP_S)
        tc = time.time()
        t.cancel()
        try:
            await t
        except BaseException:  # noqa: BLE001
            pass
        await asyncio.sleep(0.5)   # let the courtesy cancel flush
        return tc


# ================================================================== A2A
def _a2a_card(port: int):
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface
    return AgentCard(
        name=f"hop-{port}", description="broods protocol probe", version="1",
        supported_interfaces=[AgentInterface(url=f"http://127.0.0.1:{port}/",
                                             protocol_binding="JSONRPC")],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text"], default_output_modes=["text"], skills=[])


async def a2a_send(url: str, text: str, timeout: float | None,
                   return_immediately: bool = False):
    import httpx
    from a2a.client import ClientCallContext, ClientConfig, create_client
    from a2a.helpers.proto_helpers import new_text_message
    from a2a.types import Role, SendMessageConfiguration, SendMessageRequest
    hc = httpx.AsyncClient(timeout=timeout if timeout else 120)
    client = await create_client(url, client_config=ClientConfig(
        streaming=False, httpx_client=hc))
    req = SendMessageRequest(
        message=new_text_message(text, role=Role.ROLE_USER),
        configuration=SendMessageConfiguration(return_immediately=return_immediately))
    ctx = ClientCallContext(timeout=timeout) if timeout else None
    try:
        async for resp in client.send_message(req, context=ctx):
            return client, resp
    except BaseException:
        await hc.aclose()
        raise
    return client, None


def a2a_server(hop: int, port: int, downstream: str | None) -> None:
    import uvicorn
    from starlette.applications import Starlette
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.request_handlers import DefaultRequestHandlerV2
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore, TaskUpdater

    class Exec(AgentExecutor):
        async def execute(self, context, event_queue):
            msg_meta = dict(context.metadata or {})
            log({"proto": "a2a", "hop": hop, "event": "received",
                 "metadata_keys": sorted(msg_meta),
                 "deadline_seen": _deadline_seen(msg_meta)})
            if context.current_task is None:
                from a2a.helpers.proto_helpers import new_task
                from a2a.types import TaskState
                await event_queue.enqueue_event(new_task(
                    context.task_id, context.context_id,
                    TaskState.TASK_STATE_SUBMITTED))
            upd = TaskUpdater(event_queue, context.task_id, context.context_id)
            await upd.start_work()
            try:
                jobs = [agent_async(hop)]
                if downstream:
                    # As over MCP, there is no deadline to forward.
                    jobs.append(a2a_send(downstream, "subtask", None))
                await asyncio.gather(*jobs)
            except BaseException as e:
                log({"proto": "a2a", "hop": hop, "event": "cancelled",
                     "exc": type(e).__name__})
                raise
            log({"proto": "a2a", "hop": hop, "event": "finished"})
            await upd.complete()

        async def cancel(self, context, event_queue):
            log({"proto": "a2a", "hop": hop, "event": "cancel_called"})
            await TaskUpdater(event_queue, context.task_id,
                              context.context_id).cancel()

    card = _a2a_card(port)
    handler = DefaultRequestHandlerV2(Exec(), InMemoryTaskStore(), card)
    app = Starlette(routes=create_agent_card_routes(card)
                    + create_jsonrpc_routes(handler, "/"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


async def a2a_client(url: str, mode: str) -> float:
    if mode == "timeout":
        try:
            await a2a_send(url, "root", GIVE_UP_S)
        except BaseException:  # noqa: BLE001
            pass
        return time.time()
    from a2a.types import CancelTaskRequest
    client, resp = await a2a_send(url, "root", 30, return_immediately=True)
    task_id = resp.task.id if resp is not None and resp.HasField("task") else None
    await asyncio.sleep(GIVE_UP_S)
    tc = time.time()
    if task_id:
        try:
            await client.cancel_task(CancelTaskRequest(id=task_id))
        except Exception as e:  # noqa: BLE001
            log({"proto": "a2a", "hop": 0, "event": "cancel_error",
                 "exc": f"{type(e).__name__}: {e}"[:200]})
    else:
        log({"proto": "a2a", "hop": 0, "event": "no_task_id"})
    return tc


# ================================================================= gRPC
METHOD = "/broods.Agent/Work"


def grpc_server(hop: int, port: int, downstream: str | None,
                forward: bool) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import grpc

    # One long-lived channel per server, as real services hold them. A
    # channel created inside the handler is garbage-collected when the
    # handler returns, and closing it cancels the call -- which made the
    # naive variant look as if cancellation propagated when it was only
    # channel teardown. Futures are retained for the same reason.
    stub = (grpc.insecure_channel(downstream).unary_unary(METHOD)
            if downstream else None)
    live: list = []

    def work(req: bytes, context) -> bytes:
        rem = context.time_remaining()
        log({"proto": "grpc", "hop": hop, "event": "received",
             "time_remaining": rem,
             "deadline_seen": rem is not None and rem < 1e8})
        fut = None
        if downstream:
            fn = stub
            if forward:
                # The idiom gRPC makes possible: pass the remaining deadline
                # on, and cancel the downstream call when this one dies.
                # time_remaining() is ~9.2e18 when the caller set no deadline;
                # passing that on as a timeout fails the call outright.
                left = context.time_remaining()
                fut = fn.future(b"subtask",
                                timeout=left if left is not None and left < 1e8 else None)
                context.add_callback(fut.cancel)
            else:
                fut = fn.future(b"subtask")
            live.append(fut)
        n = agent_sync(hop, context.is_active)
        if n < N_CALLS:
            log({"proto": "grpc", "hop": hop, "event": "cancelled", "calls": n})
            return b"cancelled"
        if fut is not None:
            try:
                fut.result()
            except Exception:  # noqa: BLE001
                pass
        log({"proto": "grpc", "hop": hop, "event": "finished"})
        return b"done"

    server = grpc.server(ThreadPoolExecutor(max_workers=16))
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
        "broods.Agent", {"Work": grpc.unary_unary_rpc_method_handler(work)}),))
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    server.wait_for_termination()


def grpc_client(addr: str, mode: str) -> float:
    import grpc
    fn = grpc.insecure_channel(addr).unary_unary(METHOD)
    if mode == "timeout":
        try:
            fn(b"root", timeout=GIVE_UP_S)
        except grpc.RpcError:
            pass
        return time.time()
    fut = fn.future(b"root")
    time.sleep(GIVE_UP_S)
    tc = time.time()
    fut.cancel()
    return tc


# =============================================================== driver
def _wait_port(port: int, timeout: float = 30.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"port {port} never opened")


def _spawn(kind: str, hop: int, port: int, down: str | None, env: dict,
           forward: bool = False) -> subprocess.Popen:
    args = [sys.executable, "-m", "broods.experiments.protocols", "serve",
            kind, str(hop), str(port)]
    if down:
        args += ["--down", down]
    if forward:
        args.append("--forward")
    # stderr to a file, never an unread PIPE: a chatty server fills the pipe
    # buffer and blocks.
    errlog = open(ROOT / "results" / "protocols" / f"server_{kind}_hop{hop}.log", "a")
    return subprocess.Popen(args, cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=errlog)


def trial(obs: Observer, proto: str, mode: str, n: int, outdir: Path) -> dict:
    kind = "grpc" if proto.startswith("grpc") else proto
    forward = proto == "grpc-forward"
    ev_path = outdir / f"events_{proto}_{mode}_{n}.jsonl"
    for old in outdir.glob(ev_path.name + "*"):
        old.unlink()
    env = {**os.environ, "BROODS_PROTO_EVENTS": str(ev_path),
           "PYTHONPATH": str(ROOT)}
    p1, p2 = 8151, 8152
    if kind == "grpc":
        u1, u2 = f"127.0.0.1:{p1}", f"127.0.0.1:{p2}"
    elif kind == "mcp":
        u1, u2 = f"http://127.0.0.1:{p1}/mcp", f"http://127.0.0.1:{p2}/mcp"
    else:
        u1, u2 = f"http://127.0.0.1:{p1}", f"http://127.0.0.1:{p2}"
    procs = [_spawn(kind, 2, p2, None, env, forward),
             _spawn(kind, 1, p1, u2, env, forward)]
    try:
        _wait_port(p2), _wait_port(p1)
        time.sleep(1.0)
        obs.reset()
        os.environ["BROODS_PROTO_EVENTS"] = str(ev_path)
        t_start = time.time()
        if kind == "grpc":
            t_cancel = grpc_client(u1, mode)
        elif kind == "mcp":
            t_cancel = asyncio.run(mcp_client(u1, mode))
        else:
            t_cancel = asyncio.run(a2a_client(u1, mode))
        time.sleep(OBSERVE_S)
        orphans, total = obs.after(t_cancel), obs.total()
    finally:
        for p in procs:
            p.kill()
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
    events = []
    for f in sorted(outdir.glob(ev_path.name + ".*")):
        events += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
    events.sort(key=lambda e: e["t"])
    per_hop = {}
    for h in (1, 2):
        evs = [e for e in events if e.get("hop") == h]
        rec = next((e for e in evs if e["event"] == "received"), None)
        can = next((e for e in evs if e["event"] in ("cancelled",)), None)
        per_hop[h] = {
            "reached": rec is not None,
            "deadline_seen": bool(rec and rec.get("deadline_seen")),
            "time_remaining": rec.get("time_remaining") if rec else None,
            "cancel_delivered": can is not None,
            "cancel_latency_s": round(can["t"] - t_cancel, 3) if can else None,
            "cancel_called": any(e["event"] == "cancel_called" for e in evs),
            "orphan_calls": orphans.get(str(h), 0),
            "calls_total": total.get(str(h), 0),
        }
    return {"proto": proto, "mode": mode, "trial": n,
            "gave_up_after_s": round(t_cancel - t_start, 2),
            "hops": per_hop,
            "client_events": [e for e in events if e.get("hop") == 0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve")
    s.add_argument("kind")
    s.add_argument("hop", type=int)
    s.add_argument("port", type=int)
    s.add_argument("--down")
    s.add_argument("--forward", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--protos", nargs="+",
                   default=["mcp", "a2a", "grpc-naive", "grpc-forward"])
    r.add_argument("--modes", nargs="+", default=["timeout", "cancel"])
    r.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "serve":
        if a.kind == "mcp":
            mcp_server(a.hop, a.port, a.down)
        elif a.kind == "a2a":
            a2a_server(a.hop, a.port, a.down)
        else:
            grpc_server(a.hop, a.port, a.down, a.forward)
        return
    if a.cmd is None:
        a = r.parse_args([])
    outdir = ROOT / "results" / "protocols"
    outdir.mkdir(parents=True, exist_ok=True)
    obs = Observer(OBS_PORT)
    rows = []
    for proto in a.protos:
        for mode in a.modes:
            for n in range(a.trials):
                row = trial(obs, proto, mode, n, outdir)
                rows.append(row)
                h1, h2 = row["hops"][1], row["hops"][2]
                print(f"{proto:<13} {mode:<8} #{n}  "
                      f"hop1 orphans={h1['orphan_calls']:>2} cancel={h1['cancel_delivered']!s:<5} "
                      f"dl={h1['deadline_seen']!s:<5} | "
                      f"hop2 orphans={h2['orphan_calls']:>2} cancel={h2['cancel_delivered']!s:<5} "
                      f"dl={h2['deadline_seen']!s:<5} reached={h2['reached']}",
                      flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = outdir / f"protocols_{stamp}.json"
    out.write_text(json.dumps({"meta": {"n_calls": N_CALLS, "delay_s": DELAY_S,
                                        "give_up_s": GIVE_UP_S,
                                        "observe_s": OBSERVE_S},
                               "rows": rows}, indent=2, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
