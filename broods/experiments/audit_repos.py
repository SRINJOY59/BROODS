"""Static audit of public repositories: what bound do real delegating agents
declare, and what does the growth law say it admits?

    python -m broods.experiments.audit_repos discover   # GitHub repo search
    python -m broods.experiments.audit_repos fetch      # sparse clones, *.py only
    python -m broods.experiments.audit_repos analyse    # AST scan + growth law

The matrix measures how a declared bound composes under delegation in a tree
WE build. This asks what trees people write. For every delegation site found
by AST pattern, it records the bound the author declared (or the framework
default in force when they declared none) and the one-level worst case the
measured growth law implies for that construct:

  construct                           worst case, one level (outer Lo, inner Li)
  openai_agents  Agent.as_tool()      Lo + Lo*Li    Li = as_tool max_turns, else 10
  openai_agents  handoffs=[...]       Lo            (composes; Table: native)
  crewai         allow_delegation /   (Lo+1)(Li+1)  measured (L+1)^2
                 Process.hierarchical
  smolagents     managed_agents=[..]  (Lo+1)(Li+1)  measured (L+1)^2
  pydantic_ai    agent.run inside a   Lo + Lo*Li    unless usage= is threaded,
                 tool, no usage=                    in which case Lo (composes)
  autogen        AgentTool / TeamTool Lo + Lo*Li    a = b = L
  langgraph      create_supervisor    Lo            (sub-agents run as nodes)

These are upper bounds under the law -- every step delegates -- not observed
costs. That is the question an operator sizing a run has to answer, and the
declared number does not answer it.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "audit"
# Scratch OUTSIDE the repository: the repository lives in a OneDrive folder,
# and clones placed there were being synced to the cloud. Temp is not synced.
import tempfile as _tf
CLONES = Path(_tf.gettempdir()) / "broods" / "audit_clones"
MIN_FREE_GB = 1.5

DEFAULTS = {"openai_agents": 10, "crewai": 25, "smolagents": 20,
            "pydantic_ai": 50, "autogen": 1, "langgraph": 10007}
AS_TOOL_DEFAULT = 10                    # agents.run.DEFAULT_MAX_TURNS

QUERIES = {
    "crewai": "crewai in:readme,description language:Python",
    "openai_agents": '"openai-agents" in:readme,description language:Python',
    "smolagents": "smolagents in:readme,description language:Python",
    "pydantic_ai": '"pydantic-ai" in:readme,description language:Python',
    "autogen": "autogen_agentchat in:readme,description language:Python",
    "langgraph": '"langgraph" multi-agent in:readme,description language:Python',
}
# The frameworks' own source trees define these APIs rather than use them.
EXCLUDE_OWNERS = {"crewaiinc", "openai", "huggingface", "pydantic",
                  "microsoft", "langchain-ai", "run-llama"}
IMPORT_ROOTS = {"crewai": "crewai", "agents": "openai_agents",
                "smolagents": "smolagents", "pydantic_ai": "pydantic_ai",
                "autogen_agentchat": "autogen", "langgraph": "langgraph",
                "langgraph_supervisor": "langgraph"}


# ------------------------------------------------------------ discover
def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "broods-audit"})
    for attempt in range(5):
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):           # search: 10 requests/minute
                time.sleep(20 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"rate limited: {url}")


def discover(pages: int, sort: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}
    for fw, q in QUERIES.items():
        for page in range(1, pages + 1):
            url = ("https://api.github.com/search/repositories?"
                   + urllib.parse.urlencode({"q": q, "sort": sort, "order": "desc",
                                             "per_page": 100, "page": page}))
            items = _get(url).get("items", [])
            for it in items:
                if it["owner"]["login"].lower() in EXCLUDE_OWNERS or it.get("fork"):
                    continue
                rec = found.setdefault(it["full_name"], {
                    "full_name": it["full_name"], "clone_url": it["clone_url"],
                    "stars": it["stargazers_count"], "size_kb": it["size"],
                    "pushed_at": it["pushed_at"], "queries": []})
                rec["queries"].append(fw)
            (OUT / "repos.json").write_text(json.dumps(list(found.values()), indent=1))
            print(f"{fw:<14} page {page}: {len(items)} items, {len(found)} unique",
                  flush=True)
            time.sleep(7)                      # stay under 10/min unauthenticated
            if len(items) < 100:
                break
    (OUT / "repos.json").write_text(json.dumps(list(found.values()), indent=1))
    print(f"-> {len(found)} repos")


# --------------------------------------------------------------- fetch
def _clone(rec: dict) -> str:
    dest = CLONES / rec["full_name"].replace("/", "__")
    if (dest / ".done").exists():
        return "cached"
    if rec["size_kb"] > 100_000:          # 100 MB: disk-bounded run
        return "too_large"
    if shutil.disk_usage(CLONES.parent).free / 1e9 < MIN_FREE_GB:
        return "skipped_low_disk"
    shutil.rmtree(dest, ignore_errors=True)
    try:
        run = lambda *a: subprocess.run(a, cwd=dest if dest.exists() else None,
                                        capture_output=True, text=True, timeout=180)
        r = run("git", "clone", "--depth", "1", "--filter=blob:none",
                "--no-checkout", "--quiet", rec["clone_url"], str(dest))
        if r.returncode:
            return "clone_failed"
        run("git", "sparse-checkout", "set", "--no-cone", "*.py")
        r = run("git", "checkout", "--quiet")
        if r.returncode:
            return "checkout_failed"
        (dest / ".done").write_text("ok")
        return "ok"
    except subprocess.TimeoutExpired:
        return "timeout"


_WRITE = __import__("threading").Lock()


def _scan_repo(d: Path, rec: dict) -> dict:
    sites, n_py, fws_seen = [], 0, set()
    for py in d.rglob("*.py"):
        if any(p in (".venv", "venv", "site-packages", "node_modules", ".git")
               for p in py.parts):
            continue
        n_py += 1
        try:
            src = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        got = scan_file(src)
        for s in got:
            s["file"] = str(py.relative_to(d))
        sites += got
        if any(k in src for k in ("crewai", "agents import", "smolagents",
                                  "pydantic_ai", "autogen_agentchat", "langgraph")):
            try:
                fws_seen |= _frameworks_imported(ast.parse(src))
            except (SyntaxError, ValueError):
                pass
    return {"repo": rec["full_name"], "stars": rec["stars"],
            "pushed_at": rec["pushed_at"], "queries": rec["queries"],
            "py_files": n_py, "frameworks": sorted(fws_seen), "sites": sites}


def _fetch_scan(rec: dict) -> str:
    """Clone, scan, delete. The machine this ran on had 6 GB free; keeping
    ~1,000 clones was not an option, and only the scan result is needed."""
    status = _clone(rec)
    dest = CLONES / rec["full_name"].replace("/", "__")
    if status in ("ok", "cached"):
        row = _scan_repo(dest, rec)
        with _WRITE, (OUT / "sites.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    shutil.rmtree(dest, ignore_errors=True)
    return status


def _stream_scan(rec: dict) -> str:
    """Scan a repository without touching the disk: stream its tarball from
    GitHub and parse *.py members in memory. Replaces clone-scan-delete on a
    machine whose disk could not hold even a few clones at a time."""
    import tarfile
    if rec["size_kb"] > 100_000:
        return "too_large"
    url = f"https://github.com/{rec['full_name']}/archive/HEAD.tar.gz"
    sites, n_py, fws_seen = [], 0, set()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "broods-audit"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            tf = tarfile.open(fileobj=resp, mode="r|gz")
            for m in tf:
                if not (m.isfile() and m.name.endswith(".py")) or m.size > 2_000_000:
                    continue
                parts = m.name.split("/")
                if any(p in (".venv", "venv", "site-packages", "node_modules")
                       for p in parts):
                    continue
                n_py += 1
                src = tf.extractfile(m).read().decode("utf-8", "replace")
                got = scan_file(src)
                rel = "/".join(parts[1:])
                for s in got:
                    s["file"] = rel
                sites += got
                if any(k in src for k in ("crewai", "agents import", "smolagents",
                                          "pydantic_ai", "autogen_agentchat", "langgraph")):
                    try:
                        fws_seen |= _frameworks_imported(ast.parse(src))
                    except (SyntaxError, ValueError):
                        pass
    except Exception as e:  # noqa: BLE001
        return f"stream_failed:{type(e).__name__}"
    row = {"repo": rec["full_name"], "stars": rec["stars"],
           "pushed_at": rec["pushed_at"], "queries": rec["queries"],
           "py_files": n_py, "frameworks": sorted(fws_seen), "sites": sites}
    with _WRITE, (OUT / "sites.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return "ok"


def stream(workers: int) -> None:
    repos = json.loads((OUT / "repos.json").read_text())
    done = set()
    if (OUT / "sites.jsonl").exists():
        done = {json.loads(l)["repo"] for l in
                (OUT / "sites.jsonl").read_text(encoding="utf-8").splitlines() if l}
    repos = [r for r in repos if r["full_name"] not in done]
    status: dict[str, int] = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, s in enumerate(ex.map(_stream_scan, repos), 1):
            key = s.split(":")[0]
            status[key] = status.get(key, 0) + 1
            if i % 50 == 0:
                print(f"{i}/{len(repos)} {status}", flush=True)
    print(status, flush=True)


def fetch(workers: int) -> None:
    repos = json.loads((OUT / "repos.json").read_text())
    done = set()
    if (OUT / "sites.jsonl").exists():
        done = {json.loads(l)["repo"] for l in
                (OUT / "sites.jsonl").read_text(encoding="utf-8").splitlines() if l}
    repos = [r for r in repos if r["full_name"] not in done]
    CLONES.mkdir(parents=True, exist_ok=True)
    status: dict[str, int] = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, s in enumerate(ex.map(_fetch_scan, repos), 1):
            status[s] = status.get(s, 0) + 1
            if i % 25 == 0:
                print(f"{i}/{len(repos)} {status}", flush=True)
    print(status, flush=True)


# ------------------------------------------------------------- analyse
def _const(node) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value
    return None


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _fname(func) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _frameworks_imported(tree: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        mod = None
        if isinstance(n, ast.ImportFrom) and n.module:
            mod = n.module.split(".")[0]
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in IMPORT_ROOTS:
                    out.add(IMPORT_ROOTS[a.name.split(".")[0]])
        if mod in IMPORT_ROOTS:
            out.add(IMPORT_ROOTS[mod])
    return out


def scan_file(src: str) -> list[dict]:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return []
    fws = _frameworks_imported(tree)
    if not fws:
        return []
    sites: list[dict] = []
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    # Outer bounds declared anywhere in the file, per framework.
    def declared(fw: str) -> int | None:
        vals = []
        for c in calls:
            f = _fname(c.func)
            if fw == "openai_agents" and f in ("run", "run_sync", "run_streamed"):
                v = _const(_kw(c, "max_turns"))
            elif fw == "crewai" and f == "Agent":
                v = _const(_kw(c, "max_iter"))
            elif fw == "smolagents" and f.endswith("Agent"):
                v = _const(_kw(c, "max_steps"))
            elif fw == "pydantic_ai" and f == "UsageLimits":
                v = _const(_kw(c, "request_limit"))
            elif fw == "autogen" and f == "AssistantAgent":
                v = _const(_kw(c, "max_tool_iterations"))
            elif fw == "langgraph" and isinstance(c.func, ast.Attribute) \
                    and f in ("invoke", "ainvoke", "stream", "astream"):
                v = None
                for a in c.args[1:] + [k.value for k in c.keywords if k.arg == "config"]:
                    if isinstance(a, ast.Dict):
                        for k, val in zip(a.keys, a.values):
                            if isinstance(k, ast.Constant) and k.value == "recursion_limit":
                                v = _const(val)
            else:
                v = None
            if v is not None:
                vals.append(v)
        return max(vals) if vals else None

    def site(fw, construct, inner_declared, composes, **extra):
        outer = declared(fw)
        lo = outer if outer is not None else DEFAULTS[fw]
        li = inner_declared if inner_declared is not None else (
            AS_TOOL_DEFAULT if construct == "as_tool" else DEFAULTS[fw])
        if composes:
            worst = lo
        elif construct in ("hierarchical", "allow_delegation", "managed_agents"):
            worst = (lo + 1) * (li + 1)
        else:
            worst = lo + lo * li
        sites.append({"framework": fw, "construct": construct,
                      "outer_declared": outer, "outer_effective": lo,
                      "inner_declared": inner_declared, "inner_effective": li,
                      "composes": composes, "worst_one_level": worst,
                      "multiplier": round(worst / lo, 2), **extra})

    for c in calls:
        f = _fname(c.func)
        if "openai_agents" in fws:
            if f == "as_tool" and isinstance(c.func, ast.Attribute):
                site("openai_agents", "as_tool", _const(_kw(c, "max_turns")), False,
                     inner_bound_passed=_kw(c, "max_turns") is not None)
            if f == "Agent":
                h = _kw(c, "handoffs")
                if isinstance(h, (ast.List, ast.Tuple)) and h.elts:
                    site("openai_agents", "handoffs", None, True)
        if "crewai" in fws:
            if f == "Agent":
                ad = _kw(c, "allow_delegation")
                if isinstance(ad, ast.Constant) and ad.value is True:
                    site("crewai", "allow_delegation", _const(_kw(c, "max_iter")), False)
            if f == "Crew":
                p = _kw(c, "process")
                if isinstance(p, ast.Attribute) and p.attr == "hierarchical":
                    site("crewai", "hierarchical", None, False)
        if "smolagents" in fws and f.endswith("Agent"):
            m = _kw(c, "managed_agents")
            if isinstance(m, (ast.List, ast.Tuple)) and m.elts:
                site("smolagents", "managed_agents", _const(_kw(c, "max_steps")), False)
        if "autogen" in fws and f in ("AgentTool", "TeamTool"):
            site("autogen", f, None, False)
        if "langgraph" in fws and f in ("create_supervisor", "create_swarm"):
            site("langgraph", f, None, True)

    if "pydantic_ai" in fws:
        # A tool is a function taking a RunContext; a delegation site is an
        # agent.run*/run_sync call inside one. Threading `usage=` is the
        # documented idiom that makes the child spend the parent's budget.
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            ann = " ".join(ast.unparse(a.annotation) for a in fn.args.args
                           if a.annotation is not None)
            if "RunContext" not in ann:
                continue
            for c in ast.walk(fn):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) \
                        and c.func.attr in ("run", "run_sync", "run_stream"):
                    threaded = _kw(c, "usage") is not None
                    site("pydantic_ai", "run_in_tool", None, threaded,
                         usage_threaded=threaded)
    return sites


def analyse() -> dict:
    per_repo = [json.loads(l) for l in
                (OUT / "sites.jsonl").read_text(encoding="utf-8").splitlines() if l]
    return summarise(per_repo)


def _pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def summarise(per_repo: list[dict]) -> dict:
    import statistics as st
    scanned = len(per_repo)
    using = [r for r in per_repo if r["frameworks"]]
    deleg = [r for r in per_repo if r["sites"]]
    all_sites = [s for r in per_repo for s in r["sites"]]
    out: dict = {"repos_scanned": scanned, "repos_using_a_framework": len(using),
                 "repos_with_delegation": len(deleg), "sites": len(all_sites),
                 "by_framework": {}}
    for fw in DEFAULTS:
        ss = [s for s in all_sites if s["framework"] == fw]
        rs = [r for r in deleg if any(s["framework"] == fw for s in r["sites"])]
        if not ss:
            continue
        nonc = [s for s in ss if not s["composes"]]
        mults = [s["multiplier"] for s in nonc]
        worst = [s["worst_one_level"] for s in nonc]
        by_c: dict[str, int] = {}
        for s in ss:
            by_c[s["construct"]] = by_c.get(s["construct"], 0) + 1
        out["by_framework"][fw] = {
            "repos": len(rs), "sites": len(ss), "constructs": by_c,
            "composing_sites_pct": _pct(len(ss) - len(nonc), len(ss)),
            "outer_bound_declared_pct": _pct(sum(s["outer_declared"] is not None
                                                 for s in ss), len(ss)),
            "multiplier_median": st.median(mults) if mults else None,
            "worst_one_level_median": st.median(worst) if worst else None,
            "worst_one_level_max": max(worst) if worst else None,
        }
    oa = [s for s in all_sites if s["construct"] == "as_tool"]
    if oa:
        out["openai_as_tool_inner_bound_passed_pct"] = _pct(
            sum(s.get("inner_bound_passed", False) for s in oa), len(oa))
    pa = [s for s in all_sites if s["construct"] == "run_in_tool"]
    if pa:
        out["pydantic_usage_threaded_pct"] = _pct(
            sum(s.get("usage_threaded", False) for s in pa), len(pa))
    rep_mult = []
    for r in deleg:
        nonc = [s["multiplier"] for s in r["sites"] if not s["composes"]]
        if nonc:
            rep_mult.append(max(nonc))
    out["repos_noncomposing"] = len(rep_mult)
    out["repos_noncomposing_pct_of_delegating"] = _pct(len(rep_mult), len(deleg))
    if rep_mult:
        out["repo_multiplier_median"] = st.median(rep_mult)
        out["repos_multiplier_ge_10_pct"] = _pct(sum(m >= 10 for m in rep_mult),
                                                 len(rep_mult))
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--pages", type=int, default=2)
    d.add_argument("--sort", default="stars")
    f = sub.add_parser("fetch")
    f.add_argument("--workers", type=int, default=6)
    sg = sub.add_parser("stream")
    sg.add_argument("--workers", type=int, default=6)
    sub.add_parser("analyse")
    a = ap.parse_args()
    if a.cmd == "discover":
        discover(a.pages, a.sort)
    elif a.cmd == "fetch":
        fetch(a.workers)
    elif a.cmd == "stream":
        stream(a.workers)
    else:
        print(json.dumps(analyse(), indent=2))


if __name__ == "__main__":
    main()
