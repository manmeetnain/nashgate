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

## Status

`nashgate/policy/` — the equilibrium-seeking multi-agent SAC
controller — is implemented and smoke-tested standalone. Everything
else is scaffold:

- [x] Nash-SAC policy network (actor, twin critics, auto-tuned entropy, multi-player coordinator)
- [ ] Define the routing "game" — observation/action/reward for a real backend-selection env
- [ ] OpenAI-compatible proxy layer (`gateway/`)
- [ ] `router/` — wires live requests to the trained policy
- [ ] Benchmark harness vs. static routers (weighted / latency-based)

## Layout

```
nashgate/
├── nashgate/
│   ├── gateway/   OpenAI-compatible proxy — the thing agents talk to
│   ├── router/    request → backend selection, wraps policy/
│   ├── policy/    Nash-SAC equilibrium-seeking controller
│   │   ├── networks.py       actor / critic architectures
│   │   ├── replay_buffer.py
│   │   ├── agent.py          single-player SAC agent
│   │   └── router_policy.py  N independent players → one routing policy
│   └── cli/       nashgate route / policy / bench
├── tests/
└── docs/
```
