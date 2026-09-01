# nashgate

[![tests](https://github.com/manmeetnain/nashgate/actions/workflows/tests.yml/badge.svg)](https://github.com/manmeetnain/nashgate/actions/workflows/tests.yml)
[![docker](https://github.com/manmeetnain/nashgate/actions/workflows/docker.yml/badge.svg)](https://github.com/manmeetnain/nashgate/actions/workflows/docker.yml)
[![real API check](https://github.com/manmeetnain/nashgate/actions/workflows/real-api-check.yml/badge.svg)](https://github.com/manmeetnain/nashgate/actions/workflows/real-api-check.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

```
╭─────────────────────────────────────────────────────────────────╮
│ ✻ nashgate                                                       │
│                                                                   │
│   An LLM / agent gateway that routes traffic by finding a Nash   │
│   equilibrium, not a fixed weight table.                         │
│                                                                   │
│   nashgate route    start the gateway (OpenAI-compatible /v1)    │
│   nashgate policy   inspect / train the equilibrium router       │
│   nashgate bench    replay traffic vs. static routing            │
╰─────────────────────────────────────────────────────────────────╯
```

## The idea

Every LLM/agent gateway that exists today — LiteLLM, Bifrost, Portkey,
APISIX — routes requests with hand-tuned rules: weighted round-robin,
latency-based, cost-based, complexity classifiers. All of these assume
routing is a single-requester optimization problem.

It isn't, once real traffic hits. The moment N agents, subagents, or
tenants share the same rate limits, GPU budget, and cost ceiling,
they're playing a game against each other for that capacity — and a
static rule table doesn't converge to a fair or stable allocation
under contention. One noisy workflow starves the rest; nobody's
routing decision accounts for what everyone else is doing.

**nashgate is a gateway whose routing policy is trained, not
hand-written.** Each requester (agent, tenant, workflow) is a player
in a multi-agent RL system with its own reward — latency, cost,
success — and the router is trained until independent best-responses
settle into a Nash equilibrium: an allocation where no single player
can improve their own outcome by unilaterally routing elsewhere. The
result is a drop-in, OpenAI-compatible gateway that stays fair and
stable under multi-agent load instead of just under benchmark load.

## The game

One episode is a window of routing decisions. Each step, every caller
(agent, tenant, or workflow sharing this gateway) picks one backend
for its next request — all callers act simultaneously, so a caller's
own choice and everyone else's choice that step jointly decide how
congested each backend gets. That feedback loop is what makes "always
route to the fastest backend" a losing static strategy, and gives
independent learners something to find an equilibrium over.

- **Observation** (per caller) — local state (own remaining cost
  budget, own in-flight requests, last reward, next request's size,
  own recent success rate) plus shared, per-backend state everyone
  sees identically (queue depth, latency EMA, error rate, rate-limit
  headroom, cost per 1k tokens) — nobody observes what a specific
  other caller is about to do, only the aggregate effect on backends.
- **Action** — pick one backend for the next request (discrete).
- **Reward** — `+1` on success within SLA, `-1.5` on rate-limit/error/SLA
  miss, minus continuous latency and cost pressure. Dumping every
  caller onto the cheapest backend drives its queue up and its next-step
  latency up with it, so a fixed "always pick the best one" policy
  self-defeats — exactly the dynamic an equilibrium-seeking policy has
  to learn to spread load against.

Implemented in `nashgate/env/routing_env.py`, with the observation and
reward math factored into `nashgate/env/features.py` and
`nashgate/env/reward.py` — the live router imports those same
functions, so a request scores identically whether it happened in
training or in production.

## The router

`nashgate/router/live_router.py` is what a gateway calls per request:

```python
routed = router.select_backend(caller_id, request_tokens)
# ... actually call routed.backend_id, time it, catch errors ...
router.report_result(routed, latency_ms, cost, success)
```

`select_backend()` builds an observation from the router's own live
backend/caller state and asks that caller's trained agent which
backend to use. `report_result()` scores the real outcome with the
same reward used in training, updates live state so the *next*
request already reflects this one's effect on load, and — if
`online_learning` is on — feeds the transition back into that agent's
replay buffer and runs an update, so the policy keeps adapting to
real traffic instead of freezing at whatever it learned offline.
Load a trained policy with `LiveRouter.from_checkpoint(path, ...)`, or
let it construct a fresh (untrained) one to serve on. Verified
end-to-end: train against `MultiAgentRoutingEnv` → `policy.save()` →
`LiveRouter.from_checkpoint()` → `select_backend()` / `report_result()`
on synthetic live traffic.

## The gateway

`nashgate/gateway/` is the OpenAI-compatible proxy — the thing an
agent's HTTP client actually points at. `POST /v1/chat/completions`
with an `X-Nashgate-Caller` header:

```
caller request
  → resolve caller from X-Nashgate-Caller (fixed roster, unknown = 400)
  → router.select_backend()      the trained policy picks a backend
  → forward_chat_completion()    the real HTTP call to that backend
  → router.report_result()       score it, feed it back to the policy
  → response, annotated with which backend served it and why
```

Start it with `nashgate route --config path/to/config.yaml` — see
[docs/example.config.yaml](docs/example.config.yaml) for the format:
a roster of backends (base URL, API key env var, model, cost/rate-limit)
and a roster of callers (SLA, cost budget), plus an optional trained
`policy_checkpoint` to load. The response body carries a `nashgate`
field (`backend`, `latency_ms`, `cost`, `reward`) and an
`X-Nashgate-Backend` response header, so you can see the routing
decision without extra tooling.

Token counts are estimated with a ~4-chars/token heuristic
(`nashgate/gateway/tokens.py`) when a request needs pricing before a
backend has responded, and corrected from real `usage` once it comes
back — good enough to feed the router's observation, not a tokenizer
replacement.

**Streaming** (`"stream": true` in the request body) is supported: the
connection to the backend opens and its status is checked *before* any
response is returned to the caller, so a backend error still comes
back as a proper HTTP status — not a 200 stream with an error buried
in the body. Once the backend accepts the request, chunks pass through
live via `StreamingResponse`; the routing outcome (latency, cost,
success) is only reported to the policy once the stream finishes or
drops, since none of that is known until then. Token usage is read
from a trailing `usage` chunk when the backend includes one (the
`stream_options: {include_usage: true}` convention), falling back to
the same request-size estimate otherwise.

Smoke-tested end-to-end with a mocked backend response: caller
resolution, unknown-caller rejection, successful routing with reward
reporting, and the backend-failure error path (both streaming and not)
all verified through FastAPI's test client — and separately verified
against two real running servers (a mock OpenAI-compatible backend and
the actual gateway process, both over real HTTP): live SSE passthrough
with incrementally-arriving chunks, and a real 429 from the backend
correctly surfacing as a 429 through the gateway instead of a broken
200 stream.

**Docker:**

```bash
docker build -t nashgate .
docker run -p 8000:8000 \
  -v $(pwd)/docs/example.config.yaml:/config/config.yaml:ro \
  -e NASHGATE_BACKEND_FAST_KEY=sk-... \
  nashgate
```

Runs as a non-root user, ships a `/healthz`-backed `HEALTHCHECK`, and
installs the CPU-only PyTorch build — the policy is a tiny MLP with no
GPU work in the gateway, so there's no reason to drag in CUDA
libraries. Built and run-verified locally (~1.4GB image).

Or with `docker compose`:

```bash
cp .env.example .env   # fill in your backend API keys
docker compose up --build
```

[docker-compose.yml](docker-compose.yml) mounts
`docs/example.config.yaml`, reads secrets from `.env` (gitignored —
never baked into the image), and publishes port 8000. Point it at your
own config by editing the volume mount. Also build-and-run verified —
`GET /healthz` returned `200` and the container reported `healthy`.

## Verified against a real API

Everything above was tested against simulated traffic or mocked
backends — real, but not the same claim as "works against an actual
model provider." As of 2026-09-01, it does: a real `nashgate route`
process was pointed at Anthropic's OpenAI-compatible endpoint
(`https://api.anthropic.com/v1`, model `claude-haiku-4-5-20251001`)
and hit with real requests, no mocking anywhere in the path.

- **Non-streaming** — real response, real `usage` (18 prompt / 12
  completion tokens), real latency (787.5ms), correctly costed and
  scored (`"nashgate": {"cost": 3e-05, "reward": 0.9685}`).
- **Streaming** — real incremental SSE chunks passed through live
  (`"Blue"` then `" is the color of the sky and ocean."` arriving as
  separate `data:` events, not buffered). Without `stream_options:
  {"include_usage": true}` in the request, no chunk carried a `usage`
  field, and the gateway correctly fell back to the token-count
  estimate — exactly the documented fallback behavior. Re-run *with*
  that flag set: the final chunk included real usage
  (13 prompt / 5 completion tokens) and it was extracted correctly.

That first pass was a manual run. It's automated now:
[`.github/workflows/real-api-check.yml`](.github/workflows/real-api-check.yml)
runs `tests/test_real_api.py` — the same streaming and non-streaming
checks, against the same real backend — on every push to main, using a
funded `NASHGATE_ANTHROPIC_TEST_KEY` repo secret. First automated run
made two real calls and passed for real, not skipped — see
[CONTRIBUTING.md#whats-open](CONTRIBUTING.md#whats-open) for how it's
wired and what it costs when the secret isn't set (nothing — it skips).

## Multi-seed validation

Trained 10 independent policies each (seeds 0–9, 100k steps) against
both real configs, then evaluated every one against the same static
baselines over 5000 steps, seed-matched.

**Light load** — `docs/example.config.yaml`:

| Router | Reward (mean ± std) | Success | Fairness (mean ± std) |
|---|---|---|---|
| **nashgate** | 0.8925 ± 0.0021 | 98.9% | 0.9982 ± 0.0013 |
| round_robin | 0.8929 ± 0.0021 | 98.9% | 1.0000 ± 0.0000 |
| weighted | 0.8867 ± 0.0021 | 98.9% | 0.7805 ± 0.0045 |
| latency_based | 0.8694 ± 0.0022 | 98.9% | 0.7930 ± 0.1337 |
| cost_based | 0.8696 ± 0.0021 | 98.9% | 0.3333 ± 0.0000 |

**Severe contention** — `docs/severe_contention.config.yaml` (4 callers, 40–200 req/window):

| Router | Reward (mean ± std) | Success | Violations | Fairness (mean ± std) |
|---|---|---|---|---|
| **nashgate** | 0.7916 ± 0.0017 | 98.9% | 1.1% | 0.8793 ± 0.0014 |
| weighted | 0.7639 ± 0.0018 | 99.0% | 1.0% | 0.7484 ± 0.0030 |
| cost_based | 0.6659 ± 0.0015 | 99.0% | 1.0% | 0.3333 ± 0.0000 |
| round_robin | 0.4771 ± 0.0015 | 85.8% | 14.2% | 1.0000 ± 0.0000 |
| latency_based | &minus;1.1612 ± 0.0024 | 20.2% | 79.8% | 0.3364 ± 0.0006 |

Three findings:

- **Training is stable in both regimes.** Reward std stays in the
  0.0015–0.0024 range across 10 independent random seeds in *both*
  scenarios — the policy converges to essentially the same equilibrium
  every time, not a lucky single run.
- **Under light load, nashgate ties `round_robin`** — statistically
  indistinguishable, with round_robin marginally ahead. Not a
  regression: `example.config.yaml`'s rate limits are generous relative
  to 3 callers, so there's no real contention to route around, and an
  equilibrium-seeking policy has no edge when there's no game being
  played.
- **Under severe contention, nashgate wins, consistently.** `latency_based`
  collapses to a 79.8% violation rate on *every one* of the 10 seeds
  (std 0.0024 — not a fluke), while nashgate holds 98.9% success and
  1.1% violations across all 10. This is the same collapse dynamic [The
  benchmark](#the-benchmark) first showed from a single run, now backed
  by the same statistical rigor as the light-load result above — the
  flattering number and the unflattering one were both put through the
  same test, not just the one that looked good.

Together these two tables are the actual claim of the project: not
"nashgate always wins," but "nashgate wins exactly when there's a game
to win, and doesn't pretend otherwise when there isn't." Checkpoints
(60 files, ~85MB across both runs) are gitignored — reproducible from
`nashgate policy train --config <config> --seed N`, not worth carrying
in git history.

**See it happen, not just the tables** — [Equilibrium Console](https://claude.ai/code/artifact/6b80bc14-797c-477c-8581-73495629e49a)
is a live simulation of both scenarios above: the same routing-game
mechanics, animated, with the static router visibly collapsing under
contention while nashgate holds steady.

## Training a policy

```bash
nashgate policy train --config path/to/config.yaml --out ./checkpoints/latest --steps 100000
nashgate policy inspect --config path/to/config.yaml --checkpoint ./checkpoints/latest
```

`train` runs [`nashgate/policy/train.py`](nashgate/policy/train.py)
against the routing game defined by your config's backends/callers,
printing progress (mean reward, mean entropy temperature) every ~10%
of the run, and saves a checkpoint `LiveRouter.from_checkpoint()` and
`nashgate bench --checkpoint` can both load. `inspect` prints each
player's step/update count and current alpha from a saved checkpoint.

This is the same training loop `nashgate bench` uses internally when
no `--checkpoint` is given — now it's a first-class command instead of
only reachable as a bench side effect. It's still the straightforward
loop, not a tuned research pipeline: no LR schedule, no eval-during-training,
no multi-seed runs. Good enough to get a real, non-random policy to
point the gateway at; not a substitute for actually tuning one.

## The benchmark

`nashgate bench --config path/to/config.yaml` trains (or loads, with
`--checkpoint`) a policy and runs it against the same routing game as
every static baseline a real gateway ships today — `round_robin`,
`weighted` (fixed, capacity-proportional — LiteLLM's default),
`latency_based` (greedy lowest-EMA — what most gateways call "smart"
routing), and `cost_based` (greedy cheapest) — over the identical
traffic, and reports avg reward, success rate, violation rate, and
load fairness (Jain's index: 1.0 = perfectly even split, 1/n = all
traffic on one backend).

The result is contention-dependent, which is the whole thesis, not a
caveat on it:

- **Under real scarcity** (rate limits tight relative to demand — 4
  callers, backends capped at 40–200 req/window) `latency_based`
  collapses: every caller greedily picks the same "fastest right now"
  backend simultaneously, drives its latency through the roof, and
  averages a **negative** reward with a 79% violation rate. `nashgate`
  and `weighted` both stay well ahead; `cost_based` piles everyone
  onto the cheapest backend (fairness 0.33) and pays for it in
  violations.
- **Under light load** (generous rate limits, nothing actually
  contended) every router — including `round_robin` — lands within a
  few percent of `nashgate`, because there's no congestion to route
  around. The equilibrium-seeking policy only has an edge when there's
  a real game being played; it's not claiming to beat a coin flip when
  there's nothing to compete over.

`--train-steps` (default 20k) is a lightweight demo-scale pretrain,
not a tuned research run — enough to reach a stable, non-random
policy for comparison, not a benchmark-grade result on its own.

## Development

```bash
make dev     # pip install -e ".[dev]"
make check   # lint + test — what CI runs
```

`make help` lists everything: `test`, `lint`, `format`/`format-check`
(ruff), `route`/`bench` against the example config, `docker-build`/
`docker-run`, `compose-up`/`compose-down`, `clean`. See the
[Makefile](Makefile).

**Testing** — 93 tests covering every layer: backend/caller state
transitions and rate-limiting, the epsilon-floor and entropy-clamping
behavior in the actor/critic networks, the SAC agent's update
lifecycle and save/load round-trip, the routing game's contention and
reward mechanics, the live router's observation-building and
online-learning wiring, the gateway's caller resolution and error
propagation (via FastAPI's test client with a mocked backend response
— `httpx.MockTransport`, no real network calls), and every baseline
router's routing logic plus the fairness math in `bench/`.

**Linting** — `ruff check .`; `ruff format` is available via
`make format` but isn't CI-enforced, since the default formatter would
explode a lot of intentionally compact multi-value lines (dataclass
calls, test setup) into one-arg-per-line — a stylistic tradeoff, not
a correctness one.

Both run in CI on every push and PR to `main` —
[`.github/workflows/tests.yml`](.github/workflows/tests.yml).

## Status

- [x] Nash-SAC policy network (actor, twin critics, auto-tuned entropy, multi-player coordinator)
- [x] The routing game — obs/action/reward, simulated backend contention (`nashgate/env/`)
- [x] `router/` — wires live requests to the trained policy, with online learning
- [x] OpenAI-compatible proxy layer (`gateway/`)
- [x] Benchmark harness vs. static routers (`nashgate/bench/`)

## Layout

```
nashgate/
├── nashgate/
│   ├── gateway/   the OpenAI-compatible proxy — what agents talk to
│   │   ├── app.py        FastAPI app: /v1/chat/completions, /healthz
│   │   ├── config.py     loads a YAML roster into a running app
│   │   ├── backends.py   GatewayBackend — connection info + routing config
│   │   ├── callers.py    CallerRegistry — fixed caller roster
│   │   ├── proxy.py      the actual HTTP forward to a chosen backend
│   │   └── tokens.py     cheap token-count estimate for pricing/obs
│   ├── router/    live_router.py — request -> backend, wraps policy/ + env/features
│   ├── bench/     nashgate vs. static routing, on the same traffic
│   │   ├── baselines.py  round_robin / weighted / latency_based / cost_based
│   │   ├── runner.py     runs one router, scores reward/success/fairness
│   │   ├── train.py      lightweight pretrain for `bench` without a checkpoint
│   │   └── compare.py    runs every router, formats the comparison table
│   ├── env/       the routing game — obs/action/reward, shared by training + serving
│   │   ├── backend_state.py  per-backend queue / latency / rate-limit state
│   │   ├── caller_state.py   per-caller budget / inflight / success-rate state
│   │   ├── features.py       obs-building — same function used live and in training
│   │   ├── reward.py         reward shaping — same function used live and in training
│   │   └── routing_env.py    MultiAgentRoutingEnv — the simulated game, for training
│   ├── policy/    Nash-SAC equilibrium-seeking controller
│   │   ├── networks.py       actor / critic architectures
│   │   ├── replay_buffer.py
│   │   ├── agent.py          single-player SAC agent
│   │   ├── router_policy.py  N independent players → one routing policy
│   │   └── train.py          the training loop — shared by `policy train` and `bench`
│   └── cli/       nashgate route / policy train,inspect / bench
├── tests/
└── docs/
    └── example.config.yaml
```

## Community

- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup, layer boundaries, PR checklist
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — Contributor Covenant v2.1
- [SECURITY.md](SECURITY.md) — how to report a vulnerability, and the design boundaries worth knowing before you deploy this
- [CHANGELOG.md](CHANGELOG.md) — what's shipped, in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format
- [LICENSE](LICENSE) — MIT
