import numpy as np

from nashgate.env.backend_state import BackendConfig, BackendState
from nashgate.env.caller_state import CallerConfig, CallerRuntime
from nashgate.env.features import (
    LOCAL_OBS_DIM,
    SHARED_OBS_PER_BACKEND,
    build_local_obs,
    build_shared_obs,
    obs_dim,
)


def test_obs_dim_formula():
    assert obs_dim(3) == LOCAL_OBS_DIM + SHARED_OBS_PER_BACKEND * 3
    assert obs_dim(0) == LOCAL_OBS_DIM


def make_backends():
    return [
        BackendState(config=BackendConfig(base_latency_ms=500, latency_jitter_ms=50, cost_per_1k_tokens=0.01, rate_limit_per_window=10)),
        BackendState(config=BackendConfig(base_latency_ms=1000, latency_jitter_ms=50, cost_per_1k_tokens=0.03, rate_limit_per_window=10)),
    ]


def test_build_shared_obs_shape_and_range():
    backends = make_backends()
    shared = build_shared_obs(backends, sla_ref_ms=1000.0)
    assert shared.shape == (SHARED_OBS_PER_BACKEND * 2,)
    assert shared.dtype == np.float32
    assert np.all(shared >= 0.0)


def test_build_shared_obs_latency_feature_reflects_ema():
    backends = make_backends()
    shared = build_shared_obs(backends, sla_ref_ms=1000.0)
    reshaped = shared.reshape(2, SHARED_OBS_PER_BACKEND)
    # column 1 is latency_ema / sla_ref; backend 1's base latency is 2x backend 0's
    assert reshaped[1, 1] > reshaped[0, 1]


def test_build_shared_obs_clips_extreme_latency():
    backends = [BackendState(config=BackendConfig(
        base_latency_ms=100_000, latency_jitter_ms=0, cost_per_1k_tokens=0.01, rate_limit_per_window=10,
    ))]
    shared = build_shared_obs(backends, sla_ref_ms=1000.0)
    # LATENCY_CLIP_MULTIPLE = 3.0
    assert shared[1] == 3.0


def test_build_local_obs_shape_and_composition():
    caller = CallerRuntime(config=CallerConfig(cost_budget_per_window=1.0))
    shared = np.zeros(SHARED_OBS_PER_BACKEND * 2, dtype=np.float32)
    obs = build_local_obs(caller, shared)
    assert obs.shape == (LOCAL_OBS_DIM + SHARED_OBS_PER_BACKEND * 2,)
    # fresh caller: full budget, no inflight, no reward yet, full success ema
    assert obs[0] == 1.0
    assert obs[1] == 0.0
    assert obs[4] == 1.0


def test_build_local_obs_clips_last_reward():
    caller = CallerRuntime(config=CallerConfig())
    caller.last_reward = -50.0
    shared = np.zeros(SHARED_OBS_PER_BACKEND * 2, dtype=np.float32)
    obs = build_local_obs(caller, shared)
    assert obs[2] == -1.0
