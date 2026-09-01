# Changelog

All notable changes to this project are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-01

First tagged release — the full pipeline (policy, game, router,
gateway, benchmark) working end to end, with tests and CI to back it.

### Added

- **Policy** — Nash equilibrium-seeking multi-agent SAC: discrete
  actor with an epsilon-floored softmax (prevents entropy collapse),
  twin critics, auto-tuned and clamped entropy temperature, and a
  `NashEquilibriumRouter` coordinating one independent agent per
  player (`nashgate/policy/`).
- **The routing game** — `MultiAgentRoutingEnv`: observation/action/reward
  for backend selection under simulated multi-caller contention, with
  the observation-building and reward math factored into shared
  modules (`nashgate/env/features.py`, `nashgate/env/reward.py`) so
  training and live serving score requests identically.
- **Router** — `LiveRouter`, wiring real requests to a trained policy:
  `select_backend()` / `report_result()`, optional online learning,
  `from_checkpoint()` to load a saved policy.
- **Gateway** — an OpenAI-compatible proxy (`POST /v1/chat/completions`)
  built on FastAPI: caller resolution against a fixed roster, request
  forwarding via `httpx`, response annotation with the routing
  decision, and a YAML config format (`nashgate/gateway/`,
  `docs/example.config.yaml`).
- **Benchmark harness** — `nashgate bench`: compares the trained policy
  against static baselines (`round_robin`, `weighted`, `latency_based`,
  `cost_based`) on identical simulated traffic, reporting reward,
  success rate, violation rate, and Jain's fairness index
  (`nashgate/bench/`).
- **CLI** — `nashgate route` / `policy` / `bench`, with a boxed
  terminal banner on bare invocation.
- **Tests** — 93 tests across every layer, including the gateway's
  HTTP surface via FastAPI's test client with `httpx.MockTransport`
  (no real network calls).
- **CI** — GitHub Actions workflows running lint + the full test suite
  (`tests.yml`) and a Docker build (`docker.yml`) on every push and PR
  to `main`.
- MIT [LICENSE](LICENSE), [CONTRIBUTING.md](CONTRIBUTING.md),
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [SECURITY.md](SECURITY.md),
  [CHANGELOG.md](CHANGELOG.md), issue templates, a PR template.
- **Docker** — `Dockerfile` (CPU-only PyTorch, non-root user,
  `/healthz`-backed `HEALTHCHECK`) and `docker-compose.yml`; both
  build- and run-verified, not just written.
- **`ruff`** for linting, and a **`Makefile`** wiring together every
  dev command (`dev`, `test`, `lint`, `format`, `check`, `route`,
  `bench`, `docker-build`, `compose-up`, `clean`) so none of this needs
  to be re-discovered per contributor.

### Fixed

- Missing `numpy` dependency in `pyproject.toml` — used throughout
  `env/` and `policy/`, but only worked in local testing because the
  environment happened to already have it installed; caught by the
  first real CI run.
- `gateway/app.py` used FastAPI's deprecated `@app.on_event("shutdown")`;
  replaced with a proper `lifespan` context manager.
- Modernized legacy `typing.Dict`/`List`/`Optional` usage to built-in
  generics (`dict`/`list`/`X | None`) across the codebase, and removed
  a couple of unused imports/variables — surfaced by adding `ruff`.

[Unreleased]: https://github.com/manmeetnain/nashgate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.1.0
