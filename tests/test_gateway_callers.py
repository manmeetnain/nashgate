import pytest

from nashgate.env.caller_state import CallerConfig
from nashgate.gateway.callers import CallerRegistry, NamedCaller, callers_from_dicts


def test_resolve_returns_stable_index_by_registration_order():
    registry = CallerRegistry([
        NamedCaller(name="alpha", config=CallerConfig()),
        NamedCaller(name="beta", config=CallerConfig()),
    ])
    assert registry.resolve("alpha") == 0
    assert registry.resolve("beta") == 1
    assert len(registry) == 2


def test_resolve_unknown_caller_raises_with_known_names_listed():
    registry = CallerRegistry([NamedCaller(name="alpha", config=CallerConfig())])
    with pytest.raises(KeyError, match="alpha"):
        registry.resolve("ghost")


def test_callers_from_dicts_applies_defaults_and_overrides():
    raw = [
        {"name": "a"},
        {"name": "b", "sla_latency_ms": 500.0, "cost_budget_per_window": 3.0},
    ]
    callers = callers_from_dicts(raw)
    assert callers[0].config.sla_latency_ms == 2000.0
    assert callers[1].config.sla_latency_ms == 500.0
    assert callers[1].config.cost_budget_per_window == 3.0
