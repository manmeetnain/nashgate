import time

import pytest

from nashgate.router.live_router import LiveRouter


def test_select_backend_returns_valid_choice(backend_configs, caller_configs):
    router = LiveRouter(backend_configs, caller_configs, explore=True)
    routed = router.select_backend(caller_id=0, request_tokens=500)
    assert 0 <= routed.backend_id < len(backend_configs)
    assert routed.caller_id == 0
    assert routed.obs.shape == (5 + 5 * len(backend_configs),)


def test_select_backend_marks_backend_in_flight_immediately(backend_configs, caller_configs):
    router = LiveRouter(backend_configs, caller_configs, explore=False)
    routed = router.select_backend(caller_id=0, request_tokens=500)
    assert router.backends[routed.backend_id].in_flight == 1
    assert router.callers[0].inflight == 1


def test_report_result_clears_inflight_and_returns_reward(backend_configs, caller_configs):
    router = LiveRouter(backend_configs, caller_configs, explore=False, online_learning=False)
    routed = router.select_backend(caller_id=0, request_tokens=500)
    reward = router.report_result(routed, latency_ms=250.0, cost=0.01, success=True)
    assert isinstance(reward, float)
    assert router.backends[routed.backend_id].in_flight == 0
    assert router.callers[0].inflight == 0


def test_report_result_updates_caller_budget_and_success_ema(backend_configs, caller_configs):
    router = LiveRouter(backend_configs, caller_configs, explore=False, online_learning=False)
    caller = router.callers[0]
    budget_before = caller.budget_remaining
    routed = router.select_backend(caller_id=0, request_tokens=500)
    router.report_result(routed, latency_ms=250.0, cost=0.05, success=True)
    assert caller.budget_remaining == budget_before - 0.05


def test_online_learning_feeds_the_agents_replay_buffer(backend_configs, caller_configs):
    router = LiveRouter(backend_configs, caller_configs, explore=True, online_learning=True)
    agent = router.policy.agents[0]
    assert len(agent.buffer) == 0
    routed = router.select_backend(caller_id=0, request_tokens=500)
    router.report_result(routed, latency_ms=250.0, cost=0.01, success=True)
    assert len(agent.buffer) == 1


def test_online_learning_disabled_does_not_touch_buffer(backend_configs, caller_configs):
    router = LiveRouter(backend_configs, caller_configs, explore=True, online_learning=False)
    agent = router.policy.agents[0]
    routed = router.select_backend(caller_id=0, request_tokens=500)
    router.report_result(routed, latency_ms=250.0, cost=0.01, success=True)
    assert len(agent.buffer) == 0


def test_window_reset_clears_backend_and_caller_windows(backend_configs, caller_configs):
    router = LiveRouter(backend_configs, caller_configs, explore=False, window_seconds=0.05)
    router.select_backend(caller_id=0, request_tokens=500)
    router.backends[0].requests_this_window = 999
    time.sleep(0.06)
    router._maybe_reset_window()
    assert router.backends[0].requests_this_window == 0


def test_from_checkpoint_loads_a_saved_policy(tmp_path, backend_configs, caller_configs):
    trained = LiveRouter(backend_configs, caller_configs, explore=True)
    trained.policy.save(str(tmp_path))

    served = LiveRouter.from_checkpoint(str(tmp_path), backend_configs, caller_configs, explore=False)
    for pid in trained.policy.agents:
        assert served.policy.agents[pid].alpha == pytest.approx(trained.policy.agents[pid].alpha)
