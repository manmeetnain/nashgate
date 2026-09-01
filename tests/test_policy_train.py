from nashgate.policy.router_policy import NashEquilibriumRouter
from nashgate.policy.train import train_policy


def test_train_policy_returns_a_router_sized_to_the_configs(backend_configs, caller_configs):
    policy = train_policy(backend_configs, caller_configs, steps=20, batch_size=4, seed=0)
    assert isinstance(policy, NashEquilibriumRouter)
    assert set(policy.agents.keys()) == set(range(len(caller_configs)))
    assert policy.agents[0].n_actions == len(backend_configs)


def test_train_policy_calls_on_progress_the_expected_number_of_times(backend_configs, caller_configs):
    calls = []
    train_policy(
        backend_configs, caller_configs, steps=20, batch_size=4, seed=0,
        on_progress=lambda step, total, stats: calls.append((step, total, stats)),
        progress_every=5,
    )
    assert [c[0] for c in calls] == [5, 10, 15, 20]
    assert all(c[1] == 20 for c in calls)
    for _, _, stats in calls:
        assert set(stats.keys()) == {"mean_reward", "mean_alpha"}


def test_train_policy_without_on_progress_does_not_error(backend_configs, caller_configs):
    train_policy(backend_configs, caller_configs, steps=10, batch_size=4, seed=0)
