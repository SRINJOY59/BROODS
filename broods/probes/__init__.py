"""Probe registry. Importing this module is safe from any venv -- every
probe module is stdlib-only, same as this package.

Each probe asks about one OS-level guarantee:

    spawn            bounded sub-agent spawning        (pids.max)
    deadline         wall-clock bound                  (RLIMIT_CPU)
    token_budget     aggregate spend bound             (quota)
    cancellation     cancel reaches descendants        (process groups)
    fault_isolation  a sibling's failure is contained  (fault domains)
    state_isolation  agents cannot read each other     (address spaces)
    accounting       usage is reported accurately      (rusage)
    reclamation      resources return after completion (wait / reaping)
    context_bound    a single request's context is capped   (no OS analogue)
    concurrency      simultaneous calls are capped          (thread pool)
    rate_limit       request RATE is capped                 (token bucket)
"""
from broods.probes.spawn.probe import SpawnProbe
from broods.probes.deadline.probe import DeadlineProbe
from broods.probes.token_budget.probe import TokenBudgetProbe
from broods.probes.cancellation.probe import CancellationProbe
from broods.probes.fault_isolation.probe import FaultIsolationProbe
from broods.probes.state_isolation.probe import StateIsolationProbe
from broods.probes.accounting.probe import AccountingProbe
from broods.probes.reclamation.probe import ReclamationProbe
from broods.probes.context_bound.probe import ContextBoundProbe
from broods.probes.concurrency.probe import ConcurrencyProbe
from broods.probes.rate_limit.probe import RateLimitProbe
from broods.probes.offspring.probe import OffspringProbe
from broods.probes.scope_tree.probe import ScopeTreeProbe

REGISTRY = {
    "spawn": SpawnProbe,
    "deadline": DeadlineProbe,
    "token_budget": TokenBudgetProbe,
    "cancellation": CancellationProbe,
    "fault_isolation": FaultIsolationProbe,
    "state_isolation": StateIsolationProbe,
    "accounting": AccountingProbe,
    "reclamation": ReclamationProbe,
    "context_bound": ContextBoundProbe,
    "concurrency": ConcurrencyProbe,
    "rate_limit": RateLimitProbe,
    # live-model distributional probe; not part of the 88-cell matrix
    "offspring": OffspringProbe,
    # TaskScope evaluation workloads; not part of the matrix
    "scope_tree": ScopeTreeProbe,
}
