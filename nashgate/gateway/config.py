"""Loads the gateway's backend/caller roster from a YAML file. See
docs/example.config.yaml for the format."""

import yaml

from nashgate.gateway.app import create_app
from nashgate.gateway.backends import backends_from_dicts
from nashgate.gateway.callers import callers_from_dicts


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def app_from_config(path: str):
    cfg = load_config(path)
    backends = backends_from_dicts(cfg["backends"])
    callers = callers_from_dicts(cfg["callers"])
    return create_app(
        backends=backends,
        callers=callers,
        policy_checkpoint=cfg.get("policy_checkpoint"),
        explore=cfg.get("explore", False),
        online_learning=cfg.get("online_learning", True),
    )
