# nashgate

[![tests](https://github.com/manmeetnain/nashgate/actions/workflows/tests.yml/badge.svg)](https://github.com/manmeetnain/nashgate/actions/workflows/tests.yml)
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

Smoke-tested end-to-end with a mocked backend response: caller
resolution, unknown-caller rejection, successful routing with reward
reporting, and the backend-failure error path all verified through
FastAPI's test client.

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

## Testing

```bash
pip install -e ".[dev]"
pytest
```

93 tests covering every layer: backend/caller state transitions and
rate-limiting, the epsilon-floor and entropy-clamping behavior in the
actor/critic networks, the SAC agent's update lifecycle and
save/load round-trip, the routing game's contention and reward
mechanics, the live router's observation-building and online-learning
wiring, the gateway's caller resolution and error propagation (via
FastAPI's test client with a mocked backend response — `httpx.MockTransport`,
no real network calls), and every baseline router's routing logic
plus the fairness math in `bench/`.

Runs in CI on every push and PR to `main` — [`.github/workflows/tests.yml`](.github/workflows/tests.yml).

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
│   │   └── router_policy.py  N independent players → one routing policy
│   └── cli/       nashgate route / policy / bench
├── tests/
└── docs/
    └── example.config.yaml
```
