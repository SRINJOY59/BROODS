# BROODS

**B**ounds **R**esist **O**nly **O**ne **D**elegation **S**tep.

A conformance suite that asks whether agent frameworks uphold the containment
guarantees an operating system gives a process, and whether those guarantees
survive **delegation**.

They hold exactly for a single agent, and almost never compose. A framework
that correctly bounds one agent to five model calls will run 155 of them once
that agent delegates two levels deep, with all 31 agents individually obeying
the limit and stopping for the reason their documentation gives.

## What is measured

Eleven probes, each mapped to a guarantee an OS provides a process.

| probe | OS analogue | question |
|---|---|---|
| `spawn` | `pids.max` | does a step bound bound the whole tree? |
| `deadline` | `RLIMIT_CPU` | does a wall-clock bound return control? |
| `token_budget` | cgroup quota | can you bound the quantity that costs money? |
| `context_bound` | — | can you bound the size of a single request? |
| `concurrency` | thread pool | is the number of requests in flight bounded? |
| `rate_limit` | token bucket | is the achieved request rate bounded? |
| `cancellation` | process groups | does cancel reach spawned descendants? |
| `fault_isolation` | fault domains | is one sibling's failure contained? |
| `state_isolation` | address spaces | can an agent read another's state? |
| `accounting` | `rusage` | is reported usage correct under delegation? |
| `reclamation` | `wait()` | do resources return after a normal run? |

Each probe x system pair is a **cell**, and every cell resolves to a level
computed from measurements, never assigned by hand:

```
L0   the guarantee cannot be requested at all - no API to ask for it
L1   declared, and nothing enforces it
L2   enforced, but only after an overshoot
L3   enforced exactly for one agent; does not survive delegation
L4   enforced, and composes across the tree
ERR  the probe did not execute - NOT a level (see "failing toward a pass")
N/A  the property was never exercised (e.g. no threads were ever created)
```

## Systems under test

Seven frameworks — `langgraph`, `crewai`, `autogen`, `pydantic_ai`,
`openai_agents`, `smolagents`, `llamaindex` — and `baseline`, a 40-line
handwritten ReAct loop with no dependencies outside the standard library.

The baseline is the control, and it is the point. If frameworks were merely
immature, obviously-correct handwritten code would do better. It does not: it
reproduces their growth curve digit for digit, which locates the shortfall in
where the counter lives and not in the quality of any library.

## Results

88 cells: 32 at L0, 24 L1, 6 L2, 15 L3, 6 L4, plus 2 N/A and 3 ERR.

| | spawn | deadline | token | context | concur | rate | cancel | fault | state | acct | reclaim |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | L3 | L1 | L0 | L0 | L0 | L0 | L1 | L1 | L1 | L1 | N/A |
| `langgraph` | L3 | L1 | L0 | L0 | L3 | L3 | L1 | L1 | L1 | **L4** | **L4** |
| `crewai` | L3 | L1 | L2 | L0 | L0 | L2 | L0 | L3 | L1 | L1 | L2 |
| `autogen` | L3 | L1 | L3 | L0 | L0 | L0 | L1 | L3 | L1 | L1 | **L4** |
| `pydantic_ai` | L3 | L0 | L2 | L2 | L2 | L0 | L1 | L1 | L1 | L1 | **L4** |
| `openai_agents` | L3 | L0 | L0 | L0 | L0 | L0 | L1 | L3 | L1 | L1 | **L4** |
| `smolagents` | L3 | L0 | L0 | L0 | L0 | L0 | L0 | L3 | L1 | L0 | N/A |
| `llamaindex` | L0 | L3 | L0 | L0 | L0 | ERR | L0 | ERR | ERR | L0 | **L4** |

Only 32 of the 88 cells can say anything about composition at all — the rest
either run at a single depth or cannot separate systems. Of those 32, one
reaches L4.

`spawn` at a declared limit of five, LLM calls (agents) by depth:

| | d0 | d1 | d2 | d3 |
|---|---|---|---|---|
| `baseline` | 5 (1) | 30 (6) | 155 (31) | 780 (156) |
| `langgraph` | 3 (1) | 9 (3) | 21 (7) | 45 (15) |
| `crewai` | 6 (1) | 36 (6) | 186 (31) | ERR |
| `autogen` | 5 (1) | 30 (6) | 155 (31) | 780 (156) |
| `pydantic_ai` | 5 (1) | 30 (6) | 155 (31) | 780 (156) |
| `openai_agents` | 5 (1) | 30 (6) | 155 (31) | 780 (156) |
| `smolagents` | 6 (1) | 36 (6) | 186 (31) | 936 (156) |

Counting units differ: only four of the seven issue exactly `L` calls for a
declared `L`. `crewai` and `smolagents` issue `L+1`, because the terminal
answer is not an iteration by their counters, and `langgraph` issues
`ceil(L/2)`, because `recursion_limit` counts graph steps and a model call
costs two of them.

## Layout

```
probes/        one directory per probe; each defines run() and verdict()
frameworks/    one adapter per system, each executed in its own venv
adapters/      shared adapter base and the in-process variants
native/        the framework-native multi-agent constructs, measured separately
proxy.py       OpenAI-compatible proxy: fault injector and independent observer
runner.py      one probe x systems x depths
run_all.py     the whole matrix, into one timestamped bundle with a manifest
audit_caps.py  transitive introspection sweep: what bound can a framework express?
analyze.py     reads a bundle and emits the matrix
```

## Running it

The package imports itself as `broods`, so clone it into a directory of that
name and work from the parent:

```bash
git clone https://github.com/SRINJOY59/BROODS.git broods
```

The eight environments are separate because the frameworks pin incompatible
dependencies — `openai` alone spans 2.54.0 to 3.5.0 across them. Each adapter
runs as a subprocess under its own interpreter, resolved from `.venvs/<name>`
relative to the parent directory.

```bash
python -m broods.run_all
```

```bash
python -m broods.run_all --probes spawn accounting --trials 3
```

```bash
python -m broods.runner --probe spawn --depths 0 1 2
```

```bash
python -m broods.runner --probe rate_limit --frameworks crewai langgraph
```

Live-provider runs read `OPENROUTER_API_KEY` from a `.env` in the parent
directory. Only four probes admit a live model; the rest need the injector to
create the condition they measure, and run offline.

## How it works

```
runner ──> proxy (:8099)          OpenAI-compatible; observer + fault injector
       └─> subprocess per venv ──> adapter ──> framework ──> back to the proxy
```

The proxy is both halves of the method. As a **fault injector** it serves
scripted replies so an agent provably never stops on its own, which makes any
stop attributable to the framework and not to a model changing its mind. As an
**observer** it counts every request that physically arrived — socket truth a
framework cannot misreport, which is what lets `accounting` catch a 33%
under-report and `cancellation` catch orphaned children.

Probes are written against a uniform `FrameworkAdapter` and never import a
framework, so one probe runs unmodified against every system, and the
delegation tree is built in framework-agnostic code so the topology under test
is fixed by the harness.

Adding a framework is one file implementing `version`, `supports`,
`make_agent` and `invoke`. Adding a probe is one class with `run` and
`verdict`.

## Failing toward a pass

Every bug found while building this produced a **false pass**, never a false
failure:

- a heap scan that returned nothing became "perfect isolation"
- child agents that never executed became "the budget composes"
- a probe that never reached the proxy became "perfectly bounded"
- threads that were never allocated became "all threads reclaimed"
- a usage callback overwritten by a nested child gave a 66.7% attribution
  error where the true figure was 33.3%
- an elapsed-time denominator floored at 1e-9 and then rounded to three
  places, restoring the zero, would have read one request as a perfectly
  respected rate

The asymmetry is structural, not bad luck: a probe that does not run, a scan
that finds nothing, and a resource never created are all indistinguishable
from compliance. The suite is therefore built to distrust its own clean
results. Zero-call runs and non-executing children return `ERR` instead of a
level, and `reclamation` returns `N/A` instead of crediting a framework for
reaping threads it never created.

Anyone extending this should assume the same: a new probe reporting everything
green is far more likely to be broken than to be good news.

## Establishing absence

An `L0` is a claim that a framework exposes no way to ask for a guarantee, so
it is made by a separate instrument and not by the adapters. `audit_caps.py`
runs inside each framework's own environment, enumerates every constructor
parameter, model field and run-method parameter of the classes that
framework's adapter builds on, expands any whose declared type is itself one
of that framework's classes, and matches the transitive surface against a
deliberately broad keyword set per capability.

That sweep has caught five false `L0`s. Two were found only once it followed a
parameter into its own type: `langgraph` and `llamaindex` both express a
request-rate bound through a limiter that hangs off the model object instead
of the agent, and neither is visible from the agent constructors alone.

## Known limitations

- **`llamaindex` reports ERR on three probes.** Its `FunctionAgent` accepts a
  tool call and never executes the tool under this harness. Those are adapter
  gaps, **not findings**. Its `L0` results elsewhere are genuine.
- **The workload is adversarial by construction.** The suite measures what a
  substrate permits when delegation does not stop, not how often real
  deployments delegate that way.
- **The default matrix is one trial per cell.** Use `--trials 3` for anything
  quoted. Call counts are invariant across runs; wall-clock quantities are not.
- **`cancellation` uses harness-owned concurrency**, modelling the common
  pattern of firing a sub-agent from inside a tool. It does not exercise
  framework-native parallelism, where a framework may track children it
  started itself, so orphan counts bound containment from above.
- **Absence is only as good as the names.** The introspection sweep recognises
  a mechanism when its parameter is named like one; a bound spelled
  unrecognisably would still read `L0`.
