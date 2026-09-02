# Changelog

All notable changes to this project are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.9.0] - 2026-09-02

### Added

- **Verified with real latency-differentiated routing** — the
  project's last remaining simulation-only claim, closed. Set up a
  second real provider (OpenAI, alongside Anthropic) — genuinely
  different infrastructure, not two routes to one backend. First
  attempt (fresh policy, online learning on, 20 real sequential calls)
  showed real latencies genuinely differ (Anthropic ~868ms mean,
  OpenAI ~2003ms mean) but barely moved the routing split — traced to
  `NashSACAgent.update()` requiring 256 buffered transitions before any
  gradient update fires, which 20 real calls never reached. Correctly
  redesigned: trained a policy in simulation using those real measured
  latencies as the simulated basis (free, converged cleanly, reward
  0.918 → 0.938), then verified the trained checkpoint against the
  same real two-provider traffic — 20/20 requests routed to the real
  faster provider, zero to the slower one. See
  README#verified-with-real-latency-differentiated-routing.

### Changed

- **Rate limits are now enforced in live serving, not just advisory.**
  Previously `LiveRouter.select_backend()` exposed rate-limit headroom
  as an observation feature but never blocked a policy from actually
  routing to an exhausted backend — a gap found and confirmed by code
  inspection during the v0.7.0 concurrent-traffic work. Now: if the
  policy's chosen backend is out of budget for the window,
  `select_backend()` reroutes to whichever available backend has the
  most headroom instead of overloading it; if every backend is
  exhausted, it raises `AllBackendsRateLimitedError`, which the
  gateway converts to a proper `429` instead of silently dropping or
  crashing. The training env (`MultiAgentRoutingEnv`) is deliberately
  unchanged — it still lets an agent pick a rate-limited backend and
  penalizes it in the reward, since that's how the policy learns to
  avoid doing it; live serving adds a hard safety net on top of that
  learned preference, it doesn't replace it. See
  README#the-router. 8 new tests (`tests/test_live_router.py`,
  `tests/test_gateway_app.py`); also verified end-to-end outside the
  test suite by draining two rate-limited backends through the real
  `select_backend()`/`report_result()` flow.

## [0.7.0] - 2026-09-02

### Added

- Link to the [Equilibrium Console](https://claude.ai/code/artifact/6b80bc14-797c-477c-8581-73495629e49a)
  in README#multi-seed-validation — a live simulation of both the
  light-load and severe-contention scenarios, animated, showing the
  static router collapsing under contention while nashgate holds
  steady.
- **Verified under real concurrent traffic** — a real
  `severe_contention_100k` checkpoint (multi-seed validated, not a
  one-off) hit with 16 genuinely concurrent real requests
  (`asyncio.gather`) across 4 callers against a real Anthropic
  endpoint. 16/16 succeeded, truly concurrent (1247ms wall-clock, not
  13+ seconds sequential), and the routing spread sensibly across all
  3 configured backends — contrasted against an earlier attempt with a
  policy trained on a badly-scoped test scenario, which collapsed to a
  10/2 split under the same kind of real concurrent load. Narrows the
  project's one remaining simulation-only claim down to
  latency-differentiated routing specifically (this test's 3 "backends"
  were all the same real provider, differentiated only by rate limit
  and cost, not by genuinely different latency) — see
  README#verified-under-real-concurrent-traffic.

## [0.6.0] - 2026-09-01

### Added

- **Multi-seed validation of the severe-contention scenario** — 10
  independent policies trained (seeds 0–9, 100k steps each) against
  `docs/severe_contention.config.yaml`, completing the rigor pass
  started in v0.5.0 (which only covered the light-load config). Under
  real scarcity, nashgate wins consistently: 98.9% success / 1.1%
  violations across all 10 seeds, while `latency_based` collapses to a
  79.8% violation rate on *every* seed (std 0.0024 — not a fluke). The
  flattering result and the honest tie from v0.5.0 have now both been
  put through the same statistical test. See
  README#multi-seed-validation.

## [0.5.0] - 2026-09-01

### Added

- **Multi-seed training validation** — 10 independent policies trained
  (seeds 0–9, 100k steps each) against `docs/example.config.yaml`,
  evaluated against all four static baselines seed-matched. Two real
  findings: training is stable (reward std ~0.002 across seeds), and
  nashgate ties with `round_robin` under this config's light load —
  confirming, with real statistical rigor, the light-load caveat the
  benchmark section already documented from a single run. See
  README#multi-seed-validation. `checkpoints/` added to `.gitignore` —
  30 files / 42MB of model binaries aren't worth carrying in git
  history when they're reproducible from `nashgate policy train`.

## [0.4.0] - 2026-09-01

Automates what v0.3.0 verified by hand.

### Added

- **`.github/workflows/real-api-check.yml`** + **`tests/test_real_api.py`**
  — the gateway (streaming and non-streaming) is now checked against a
  real, live Anthropic backend on every push to main, via a funded
  `NASHGATE_ANTHROPIC_TEST_KEY` repo secret. Without that secret set,
  the same tests skip cleanly — locally too, so this never costs
  anything by accident. First automated run made two real calls and
  passed. `workflow_dispatch` included for on-demand runs.

## [0.3.0] - 2026-09-01

### Verified

- The gateway, streaming and non-streaming, manually verified against
  a real live backend for the first time — Anthropic's OpenAI-compatible
  endpoint, model `claude-haiku-4-5-20251001`, no mocking anywhere in
  the path. Real usage/latency/cost/reward on the non-streaming path;
  real incremental SSE passthrough on the streaming path, including
  confirming the `stream_options: {include_usage: true}` usage-extraction
  fallback behaves exactly as documented (falls back to the token
  estimate when absent, extracts real usage when present). One manual
  run, not automated CI coverage — see README#verified-against-a-real-api.

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

[Unreleased]: https://github.com/manmeetnain/nashgate/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.9.0
[0.8.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.8.0
[0.7.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.7.0
[0.6.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.6.0
[0.5.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.5.0
[0.4.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.4.0
[0.3.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.3.0
[0.2.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.2.0
[0.1.0]: https://github.com/manmeetnain/nashgate/releases/tag/v0.1.0
