from nashgate.env.caller_state import CallerConfig, CallerRuntime


def test_budget_remaining_starts_full():
    caller = CallerRuntime(config=CallerConfig(cost_budget_per_window=2.5))
    assert caller.budget_remaining == 2.5


def test_reset_window_restores_full_budget():
    caller = CallerRuntime(config=CallerConfig(cost_budget_per_window=2.5))
    caller.budget_remaining = 0.1
    caller.reset_window()
    assert caller.budget_remaining == 2.5


def test_reset_window_does_not_touch_other_state():
    caller = CallerRuntime(config=CallerConfig())
    caller.inflight = 3
    caller.success_ema = 0.5
    caller.reset_window()
    assert caller.inflight == 3
    assert caller.success_ema == 0.5
