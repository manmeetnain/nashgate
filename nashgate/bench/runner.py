"""Runs one router against the routing game and scores it."""

from dataclasses import dataclass, field

from nashgate.env.routing_env import MultiAgentRoutingEnv


@dataclass
class BenchResult:
    avg_reward: float
    success_rate: float
    violation_rate: float
    fairness_jain: float
    n_requests: int
    backend_counts: list[int] = field(default_factory=list)


def jain_fairness_index(counts: list[int]) -> float:
    """1.0 = every backend got an equal share of traffic; 1/n = all
    traffic landed on one backend. Standard load-balancing fairness
    metric (Jain et al., 1984) — used here instead of raw variance
    because it's already normalized to [1/n, 1] regardless of n."""
    total = sum(counts)
    if total == 0:
        return 0.0
    n = len(counts)
    sum_sq = sum(c * c for c in counts)
    return (total ** 2) / (n * sum_sq) if sum_sq > 0 else 0.0


def run_episode(env: MultiAgentRoutingEnv, router, n_steps: int) -> BenchResult:
    obs = env.reset()
    total_reward = 0.0
    n_requests = 0
    n_success = 0
    n_violations = 0
    backend_counts = [0] * env.n_backends

    for _ in range(n_steps):
        actions = router.route(obs)
        obs, rewards, done, info = env.step(actions)

        for reward in rewards.values():
            total_reward += reward
            n_requests += 1
        n_success += sum(1 for ok in info["success"].values() if ok)
        n_violations += len(info["rate_limited"]) + len(info["errored"])
        for backend_id in actions.values():
            backend_counts[backend_id] += 1

        if done:
            obs = env.reset()

    return BenchResult(
        avg_reward=total_reward / max(1, n_requests),
        success_rate=n_success / max(1, n_requests),
        violation_rate=n_violations / max(1, n_requests),
        fairness_jain=jain_fairness_index(backend_counts),
        n_requests=n_requests,
        backend_counts=backend_counts,
    )
