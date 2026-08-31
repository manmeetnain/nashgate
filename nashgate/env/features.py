"""Observation features — the exact shape the policy is trained on.

Used by both the training environment (against simulated backend
state) and the live router (against real backend state), so there's
no train/serve skew in what the policy sees. See routing_env.py for
the full field-by-field spec.
"""


import numpy as np

from nashgate.env.backend_state import BackendState
from nashgate.env.caller_state import CallerRuntime

LOCAL_OBS_DIM = 5
SHARED_OBS_PER_BACKEND = 5

OWN_CONCURRENCY_CAP = 8.0
QUEUE_DEPTH_NORM = 20.0
MAX_REQUEST_TOKENS = 4000.0
MAX_COST_PER_1K_REF = 0.06
LATENCY_CLIP_MULTIPLE = 3.0


def obs_dim(n_backends: int) -> int:
    return LOCAL_OBS_DIM + SHARED_OBS_PER_BACKEND * n_backends


def build_shared_obs(backends: list[BackendState], sla_ref_ms: float) -> np.ndarray:
    features = []
    for backend in backends:
        features.extend([
            min(backend.in_flight / QUEUE_DEPTH_NORM, 1.0),
            min(backend.latency_ema_ms / sla_ref_ms, LATENCY_CLIP_MULTIPLE),
            backend.error_rate_ema,
            backend.rate_limit_headroom(),
            min(backend.config.cost_per_1k_tokens / MAX_COST_PER_1K_REF, 1.0),
        ])
    return np.array(features, dtype=np.float32)


def build_local_obs(caller: CallerRuntime, shared: np.ndarray) -> np.ndarray:
    local = np.array([
        caller.budget_remaining / max(caller.config.cost_budget_per_window, 1e-6),
        min(caller.inflight / OWN_CONCURRENCY_CAP, 1.0),
        float(np.clip(caller.last_reward, -1.0, 1.0)),
        min(caller.pending_tokens / MAX_REQUEST_TOKENS, 1.0),
        caller.success_ema,
    ], dtype=np.float32)
    return np.concatenate([local, shared])
