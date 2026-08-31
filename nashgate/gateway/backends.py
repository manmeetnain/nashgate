"""A backend the gateway can actually forward requests to.

Wraps two things every backend needs: connection details (where to
send the HTTP request, which API key, which model name to substitute)
and a nashgate.env.BackendConfig — the cost/rate-limit numbers the
router's observation features are built from. Same object, same
index, both concerns.
"""

import os
from dataclasses import dataclass

from nashgate.env import BackendConfig


@dataclass
class GatewayBackend:
    name: str
    base_url: str
    api_key_env: str
    model: str
    routing_config: BackendConfig

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"backend '{self.name}': environment variable {self.api_key_env} is not set"
            )
        return key


def backends_from_dicts(raw: list) -> list:
    """Build GatewayBackend list from plain dicts (e.g. loaded from YAML)."""
    out = []
    for entry in raw:
        out.append(GatewayBackend(
            name=entry["name"],
            base_url=entry["base_url"],
            api_key_env=entry["api_key_env"],
            model=entry["model"],
            routing_config=BackendConfig(
                base_latency_ms=entry.get("base_latency_ms", 500.0),
                latency_jitter_ms=entry.get("latency_jitter_ms", 100.0),
                cost_per_1k_tokens=entry["cost_per_1k_tokens"],
                rate_limit_per_window=entry["rate_limit_per_window"],
                congestion_latency_ms_per_inflight=entry.get("congestion_latency_ms_per_inflight", 40.0),
            ),
        ))
    return out
