import numpy as np

from nashgate.policy.router_policy import NashEquilibriumRouter


def make_router(n_players=3, obs_dim=6, n_backends=4, batch_size=4):
    return NashEquilibriumRouter(n_players=n_players, obs_dim=obs_dim, n_backends=n_backends, batch_size=batch_size)


def make_obs(n_players, obs_dim):
    return {i: np.random.randn(obs_dim).astype(np.float32) for i in range(n_players)}


def test_creates_one_independent_agent_per_player():
    router = make_router(n_players=5)
    assert set(router.agents.keys()) == set(range(5))
    assert len({id(a) for a in router.agents.values()}) == 5


def test_route_returns_one_action_per_player_in_range():
    router = make_router(n_players=3, n_backends=4)
    obs = make_obs(3, 6)
    actions = router.route(obs, explore=True)
    assert set(actions.keys()) == {0, 1, 2}
    assert all(0 <= a < 4 for a in actions.values())


def test_store_and_update_runs_every_agent_independently():
    router = make_router(n_players=3, batch_size=4)
    obs = make_obs(3, 6)
    actions = {i: 0 for i in range(3)}
    rewards = {i: 1.0 for i in range(3)}
    next_obs = make_obs(3, 6)
    for _ in range(10):
        router.store(obs, actions, rewards, next_obs, done=False)
    results = router.update()
    assert set(results.keys()) == {0, 1, 2}
    assert all(r is not None for r in results.values())


def test_save_and_load_round_trip(tmp_path):
    router = make_router(n_players=2, batch_size=4)
    obs = make_obs(2, 6)
    actions = {0: 0, 1: 1}
    rewards = {0: 1.0, 1: -1.0}
    for _ in range(10):
        router.store(obs, actions, rewards, obs, done=False)
    router.update()

    router.save(str(tmp_path))

    fresh = make_router(n_players=2, batch_size=4)
    fresh.load(str(tmp_path))
    for pid in router.agents:
        assert fresh.agents[pid].alpha == router.agents[pid].alpha
