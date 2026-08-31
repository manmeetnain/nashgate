from nashgate.env.backend_state import BackendConfig, BackendState


def make_backend(rate_limit=10):
    return BackendState(config=BackendConfig(
        base_latency_ms=500.0, latency_jitter_ms=50.0,
        cost_per_1k_tokens=0.01, rate_limit_per_window=rate_limit,
    ))


def test_latency_ema_seeded_from_base_latency():
    backend = make_backend()
    assert backend.latency_ema_ms == 500.0


def test_rate_limit_headroom_full_when_unused():
    backend = make_backend(rate_limit=10)
    assert backend.rate_limit_headroom() == 1.0
    assert not backend.is_rate_limited()


def test_rate_limit_headroom_decreases_and_trips():
    backend = make_backend(rate_limit=4)
    for _ in range(4):
        backend.requests_this_window += 1
    assert backend.rate_limit_headroom() == 0.0
    assert backend.is_rate_limited()


def test_reset_window_clears_request_count_not_inflight():
    backend = make_backend(rate_limit=4)
    backend.requests_this_window = 4
    backend.in_flight = 3
    backend.reset_window()
    assert backend.requests_this_window == 0
    assert backend.in_flight == 3
    assert not backend.is_rate_limited()


def test_decay_in_flight_reduces_but_never_negative():
    backend = make_backend()
    backend.in_flight = 10
    backend.decay_in_flight(fraction=0.3)
    assert backend.in_flight == 7
    backend.in_flight = 0
    backend.decay_in_flight(fraction=1.0)
    assert backend.in_flight == 0


def test_update_ema_moves_toward_observed_value():
    backend = make_backend()
    before = backend.latency_ema_ms
    backend.update_ema(observed_latency_ms=2000.0, errored=False, alpha=0.5)
    assert before < backend.latency_ema_ms < 2000.0


def test_update_ema_tracks_error_rate():
    backend = make_backend()
    for _ in range(5):
        backend.update_ema(observed_latency_ms=500.0, errored=True, alpha=0.5)
    assert backend.error_rate_ema > 0.9
