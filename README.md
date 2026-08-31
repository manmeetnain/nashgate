# nashgate

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

Implemented in `nashgate/env/routing_env.py`, and verified end-to-end
against the policy: `env.step()` → `router.route()` → `router.update()`
runs a full training loop with reward accumulating and alpha
stabilizing per player.

## Status

- [x] Nash-SAC policy network (actor, twin critics, auto-tuned entropy, multi-player coordinator)
- [x] The routing game — obs/action/reward, simulated backend contention (`nashgate/env/`)
- [ ] OpenAI-compatible proxy layer (`gateway/`)
- [ ] `router/` — wires live requests to the trained policy
- [ ] Benchmark harness vs. static routers (weighted / latency-based)

## Layout

```
nashgate/
├── nashgate/
│   ├── gateway/   OpenAI-compatible proxy — the thing agents talk to
│   ├── router/    request → backend selection, wraps policy/
│   ├── env/       the routing game — simulated backend contention for training
│   │   ├── backend_state.py  per-backend queue / latency / rate-limit state
│   │   └── routing_env.py    MultiAgentRoutingEnv — obs/action/reward
│   ├── policy/    Nash-SAC equilibrium-seeking controller
│   │   ├── networks.py       actor / critic architectures
│   │   ├── replay_buffer.py
│   │   ├── agent.py          single-player SAC agent
│   │   └── router_policy.py  N independent players → one routing policy
│   └── cli/       nashgate route / policy / bench
├── tests/
└── docs/
```
