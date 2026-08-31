from nashgate.env.backend_state import BackendConfig
from nashgate.env.caller_state import CallerConfig
from nashgate.env.routing_env import MultiAgentRoutingEnv


def make_env(episode_len=20, window_steps=10, seed=0, backend_configs=None, caller_configs=None):
    backends = backend_configs or [
        BackendConfig(base_latency_ms=300, latency_jitter_ms=50, cost_per_1k_tokens=0.01, rate_limit_per_window=40),
        BackendConfig(base_latency_ms=600, latency_jitter_ms=80, cost_per_1k_tokens=0.002, rate_limit_per_window=100),
    ]
    callers = caller_configs or [CallerConfig() for _ in range(3)]
    return MultiAgentRoutingEnv(backends, callers, episode_len=episode_len, window_steps=window_steps, seed=seed)


def test_reset_returns_one_obs_per_caller_with_correct_shape():
    env = make_env()
    obs = env.reset()
    assert set(obs.keys()) == set(range(env.n_callers))
    for o in obs.values():
        assert o.shape == (env.obs_dim,)


def test_step_returns_reward_and_success_for_every_acting_caller():
    env = make_env()
    obs = env.reset()
    actions = {i: 0 for i in range(env.n_callers)}
    next_obs, rewards, done, info = env.step(actions)
    assert set(rewards.keys()) == set(range(env.n_callers))
    assert set(info["success"].keys()) == set(range(env.n_callers))
    assert isinstance(done, bool)


def test_episode_terminates_after_episode_len_steps():
    env = make_env(episode_len=3)
    env.reset()
    actions = {i: 0 for i in range(env.n_callers)}
    dones = []
    for _ in range(3):
        _, _, done, _ = env.step(actions)
        dones.append(done)
    assert dones == [False, False, True]


def test_rate_limited_backend_forces_a_violation():
    tiny_backend = [BackendConfig(base_latency_ms=100, latency_jitter_ms=0, cost_per_1k_tokens=0.01, rate_limit_per_window=1)]
    env = make_env(episode_len=10, window_steps=1000, backend_configs=tiny_backend, caller_configs=[CallerConfig()])
    env.reset()
    # first request fills the only rate-limit slot
    _, rewards_1, _, info_1 = env.step({0: 0})
    assert info_1["rate_limited"] == []
    # second request in the same window is rate-limited
    _, rewards_2, _, info_2 = env.step({0: 0})
    assert info_2["rate_limited"] == [0]
    assert info_2["success"][0] is False
    assert rewards_2[0] < rewards_1[0]


def test_window_reset_clears_rate_limit_and_restores_budget():
    tiny_backend = [BackendConfig(base_latency_ms=100, latency_jitter_ms=0, cost_per_1k_tokens=0.01, rate_limit_per_window=1)]
    env = make_env(episode_len=10, window_steps=2, backend_configs=tiny_backend, caller_configs=[CallerConfig()])
    env.reset()
    env.step({0: 0})           # step 1: fills the window
    _, _, _, info = env.step({0: 0})   # step 2: rate-limited, and triggers window reset (2 % 2 == 0)
    assert info["rate_limited"] == [0]
    assert not env.backends[0].is_rate_limited()


def test_congestion_makes_a_busy_backend_score_worse_next_step():
    # one backend, two callers hammering it simultaneously each step
    busy_backend = [BackendConfig(
        base_latency_ms=200, latency_jitter_ms=0, cost_per_1k_tokens=0.001,
        rate_limit_per_window=1000, congestion_latency_ms_per_inflight=500,
    )]
    env = make_env(episode_len=10, window_steps=1000, backend_configs=busy_backend, caller_configs=[CallerConfig(), CallerConfig()])
    env.reset()
    _, rewards_step1, _, _ = env.step({0: 0, 1: 0})
    _, rewards_step2, _, _ = env.step({0: 0, 1: 0})
    # in_flight accumulates across simultaneous requests, so step 2 sees more congestion
    assert rewards_step2[0] < rewards_step1[0]


def test_backend_chosen_recorded_in_info():
    env = make_env()
    env.reset()
    actions = {i: i % env.n_backends for i in range(env.n_callers)}
    _, _, _, info = env.step(actions)
    assert info["backend_chosen"] == actions
