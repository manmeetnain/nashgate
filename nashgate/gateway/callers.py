"""Caller identity: maps a request to one of the fixed players the
policy was trained with. The roster is fixed at startup — each caller
has its own trained agent, so an unrecognized caller is rejected
rather than silently mapped onto someone else's policy."""

from dataclasses import dataclass

from nashgate.env import CallerConfig

CALLER_HEADER = "x-nashgate-caller"


@dataclass
class NamedCaller:
    name: str
    config: CallerConfig


class CallerRegistry:
    def __init__(self, callers: list[NamedCaller]):
        self._by_name: dict[str, int] = {c.name: i for i, c in enumerate(callers)}
        self.configs: list[CallerConfig] = [c.config for c in callers]

    def resolve(self, name: str) -> int:
        if name not in self._by_name:
            raise KeyError(
                f"unknown caller '{name}' — configured callers: {sorted(self._by_name)}"
            )
        return self._by_name[name]

    def __len__(self) -> int:
        return len(self.configs)


def callers_from_dicts(raw: list) -> list[NamedCaller]:
    out = []
    for entry in raw:
        out.append(NamedCaller(
            name=entry["name"],
            config=CallerConfig(
                sla_latency_ms=entry.get("sla_latency_ms", 2000.0),
                cost_budget_per_window=entry.get("cost_budget_per_window", 1.0),
                min_request_tokens=entry.get("min_request_tokens", 200),
                max_request_tokens=entry.get("max_request_tokens", 2000),
            ),
        ))
    return out
