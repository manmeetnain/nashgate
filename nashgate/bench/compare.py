"""Runs nashgate's trained policy and every static baseline against the
same routing game and reports them side by side."""

from typing import Dict, List, Optional

from nashgate.bench.baselines import (
    CostBasedRouter,
    LatencyBasedRouter,
    RoundRobinRouter,
    TrainedRouterAdapter,
    WeightedRouter,
)
from nashgate.bench.runner import BenchResult, run_episode
from nashgate.env.backend_state import BackendConfig
from nashgate.env.caller_state import CallerConfig
from nashgate.env.routing_env import MultiAgentRoutingEnv
from nashgate.policy import NashEquilibriumRouter


def compare_routers(
    backend_configs: List[BackendConfig],
    caller_configs: List[CallerConfig],
    policy: NashEquilibriumRouter,
    n_steps: int = 5000,
    window_steps: int = 50,
    seed: int = 0,
) -> Dict[str, BenchResult]:
    n_callers = len(caller_configs)
    n_backends = len(backend_configs)

    routers = {
        "nashgate": TrainedRouterAdapter(policy),
        "round_robin": RoundRobinRouter(n_callers, n_backends),
        "weighted": WeightedRouter.from_backend_configs(backend_configs, seed=seed),
        "latency_based": LatencyBasedRouter(n_backends),
        "cost_based": CostBasedRouter(n_backends),
    }

    results: Dict[str, BenchResult] = {}
    for name, router in routers.items():
        # Fresh env per router, same seed, so every router faces the same
        # request-size/error-jitter draw schedule as far as its own action
        # sequence allows — differences in outcome trace back to routing
        # decisions, not to which router got luckier traffic.
        env = MultiAgentRoutingEnv(
            backend_configs, caller_configs, episode_len=n_steps, window_steps=window_steps, seed=seed
        )
        results[name] = run_episode(env, router, n_steps=n_steps)
    return results


def format_table(results: Dict[str, BenchResult], title: Optional[str] = None) -> str:
    headers = ["router", "avg reward", "success", "violations", "fairness"]
    best_reward = max(r.avg_reward for r in results.values())

    rows = []
    for name, r in results.items():
        label = f"{name} ★" if r.avg_reward == best_reward else name
        rows.append([
            label,
            f"{r.avg_reward:+.3f}",
            f"{r.success_rate * 100:.1f}%",
            f"{r.violation_rate * 100:.1f}%",
            f"{r.fairness_jain:.3f}",
        ])

    col_w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(cell))

    def hline(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in col_w) + right

    def fmt_row(cells):
        return "│ " + " │ ".join(c.ljust(col_w[i]) for i, c in enumerate(cells)) + " │"

    width = sum(col_w) + 3 * len(col_w) + 1
    lines = []
    if title:
        lines.append("╭" + "─" * (width - 2) + "╮")
        lines.append("│ " + title.ljust(width - 4) + " │")
        lines.append(hline("├", "┬", "┤"))
    else:
        lines.append(hline("╭", "┬", "╮"))
    lines.append(fmt_row(headers))
    lines.append(hline("├", "┼", "┤"))
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(hline("╰", "┴", "╯"))
    return "\n".join(lines)
