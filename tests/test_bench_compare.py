from nashgate.bench.compare import compare_routers, format_table
from nashgate.bench.runner import BenchResult
from nashgate.bench.train import train_policy


def test_compare_routers_returns_all_five_routers(backend_configs, caller_configs):
    policy = train_policy(backend_configs, caller_configs, steps=50, seed=0)
    results = compare_routers(backend_configs, caller_configs, policy, n_steps=50, seed=0)
    assert set(results.keys()) == {"nashgate", "round_robin", "weighted", "latency_based", "cost_based"}
    for r in results.values():
        assert r.n_requests > 0


def test_format_table_includes_every_router_name_and_title():
    results = {
        "nashgate": BenchResult(
            avg_reward=0.9, success_rate=0.95, violation_rate=0.05, fairness_jain=0.9,
            n_requests=100, backend_counts=[50, 50],
        ),
        "round_robin": BenchResult(
            avg_reward=0.5, success_rate=0.8, violation_rate=0.2, fairness_jain=1.0,
            n_requests=100, backend_counts=[50, 50],
        ),
    }
    table = format_table(results, title="test run")
    assert "nashgate" in table
    assert "round_robin" in table
    assert "test run" in table


def test_format_table_marks_the_best_avg_reward():
    results = {
        "a": BenchResult(
            avg_reward=0.9, success_rate=1.0, violation_rate=0.0, fairness_jain=1.0,
            n_requests=1, backend_counts=[1],
        ),
        "b": BenchResult(
            avg_reward=0.1, success_rate=1.0, violation_rate=0.0, fairness_jain=1.0,
            n_requests=1, backend_counts=[1],
        ),
    }
    table = format_table(results)
    lines = table.splitlines()
    a_line = next(line for line in lines if line.strip().startswith("│ a "))
    b_line = next(line for line in lines if line.strip().startswith("│ b "))
    assert "★" in a_line
    assert "★" not in b_line
