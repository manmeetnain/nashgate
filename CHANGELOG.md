# Changelog

All notable changes to this project are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-09-01

Closes two of the three items that were open after 0.1.0 — real CLI
training and gateway streaming. The third (live-LLM integration
testing) needs real API keys/spend, so it stays open; see
[SECURITY.md](SECURITY.md) and [CONTRIBUTING.md](CONTRIBUTING.md#whats-open).
Not tagged 1.0.0: that's a stable-API signal this project hasn't
earned yet, on the same grounds.

### Added

- **`nashgate policy train` / `policy inspect`** — a real CLI for
  training a policy against a config's routing game (with progress
  reporting) and saving a checkpoint, plus inspecting a saved
  checkpoint's per-player step/update counts and entropy temperature.
  Was a stub before; the training loop itself moved to
  `nashgate/policy/train.py` as the canonical implementation, shared
  with `nashgate bench`'s own pretrain path (`nashgate/bench/train.py`
  now just re-exports it).
- **Streaming support in the gateway** — `"stream": true` requests get
  a real SSE passthrough (`StreamingResponse`) instead of only working
  non-streaming. The connection opens and its status is checked before
  any response commits, so a backend error still surfaces as a proper
  HTTP status rather than a 200 stream with an error buried in the
  body. Verified against mocked backends in tests, and separately
  against two real running servers over real HTTP (a mock
  OpenAI-compatible backend + the actual gateway process) — live
  incremental SSE passthrough, and a real 429 correctly propagating.

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

[Unreleased]: https://github.com/manmeetnain/nashgate/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.2.0
[0.1.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.1.0
