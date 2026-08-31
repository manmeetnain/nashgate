import numpy as np

from nashgate.bench.baselines import (
    CostBasedRouter,
    LatencyBasedRouter,
    RoundRobinRouter,
    TrainedRouterAdapter,
    WeightedRouter,
    shared_features,
)
from nashgate.env.backend_state import BackendConfig
from nashgate.env.features import LOCAL_OBS_DIM, SHARED_OBS_PER_BACKEND


def make_obs_with_shared(shared_matrix: np.ndarray) -> np.ndarray:
    local = np.zeros(LOCAL_OBS_DIM, dtype=np.float32)
    return np.concatenate([local, shared_matrix.flatten().astype(np.float32)])


def test_shared_features_reshapes_correctly():
    shared = np.arange(2 * SHARED_OBS_PER_BACKEND, dtype=np.float32)
    obs = make_obs_with_shared(shared.reshape(2, SHARED_OBS_PER_BACKEND))
    reshaped = shared_features(obs, n_backends=2)
    assert reshaped.shape == (2, SHARED_OBS_PER_BACKEND)
    assert np.array_equal(reshaped[0], shared[:SHARED_OBS_PER_BACKEND])


def test_round_robin_cycles_through_backends_in_order():
    router = RoundRobinRouter(n_callers=1, n_backends=3)
    obs_dict = {0: np.zeros(LOCAL_OBS_DIM + 3 * SHARED_OBS_PER_BACKEND, dtype=np.float32)}
    picks = [router.route(obs_dict)[0] for _ in range(6)]
    assert picks == [0, 1, 2, 0, 1, 2]


def test_round_robin_tracks_each_caller_independently():
    router = RoundRobinRouter(n_callers=2, n_backends=2)
    obs_dict = {0: np.zeros(1), 1: np.zeros(1)}
    picks = [router.route(obs_dict) for _ in range(3)]
    assert [p[0] for p in picks] == [0, 1, 0]
    assert [p[1] for p in picks] == [0, 1, 0]


def test_weighted_router_from_backend_configs_weighs_by_rate_limit():
    configs = [
        BackendConfig(base_latency_ms=1, latency_jitter_ms=1, cost_per_1k_tokens=0.01, rate_limit_per_window=10),
        BackendConfig(base_latency_ms=1, latency_jitter_ms=1, cost_per_1k_tokens=0.01, rate_limit_per_window=90),
    ]
    router = WeightedRouter.from_backend_configs(configs)
    assert router.weights == [0.1, 0.9]


def test_weighted_router_never_picks_a_zero_weight_backend():
    router = WeightedRouter(weights=[1.0, 0.0], seed=1)
    obs_dict = {0: np.zeros(1)}
    picks = {router.route(obs_dict)[0] for _ in range(50)}
    assert picks == {0}


def test_latency_based_router_picks_lowest_latency_backend():
    shared = np.array([
        [0.0, 0.9, 0.0, 1.0, 0.0],  # backend 0: high latency
        [0.0, 0.1, 0.0, 1.0, 0.0],  # backend 1: low latency
        [0.0, 0.5, 0.0, 1.0, 0.0],  # backend 2: mid latency
    ])
    obs = make_obs_with_shared(shared)
    router = LatencyBasedRouter(n_backends=3)
    actions = router.route({0: obs})
    assert actions[0] == 1


def test_cost_based_router_picks_cheapest_backend():
    shared = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.8],  # backend 0: expensive
        [0.0, 0.0, 0.0, 1.0, 0.1],  # backend 1: cheap
    ])
    obs = make_obs_with_shared(shared)
    router = CostBasedRouter(n_backends=2)
    actions = router.route({0: obs})
    assert actions[0] == 1


class _StubPolicy:
    def __init__(self, fixed_action=2):
        self.fixed_action = fixed_action
        self.last_explore = None

    def route(self, obs_dict, explore):
        self.last_explore = explore
        return {cid: self.fixed_action for cid in obs_dict}


def test_trained_router_adapter_forces_greedy_evaluation():
    stub = _StubPolicy(fixed_action=2)
    adapter = TrainedRouterAdapter(stub)
    actions = adapter.route({0: np.zeros(1), 1: np.zeros(1)})
    assert stub.last_explore is False
    assert actions == {0: 2, 1: 2}
