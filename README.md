# nashgate

```
$ nashgate route --help

  nashgate — an LLM/agent gateway that routes under contention
  by finding a Nash equilibrium, not a fixed weight table.

  USAGE
    nashgate route     start the gateway (OpenAI-compatible /v1 API)
    nashgate policy     inspect / train the equilibrium-seeking router
    nashgate bench      replay traffic, compare vs. static routing

  Every other gateway (LiteLLM, Bifrost, Portkey, APISIX) routes with
  hand-tuned rules: weighted round-robin, latency-based, cost-based.
  Those rules assume one requester. Once N agents/tenants share the
  same rate limits, GPU budget, and cost ceiling, they're playing a
  game against each other — and static rules don't converge to a
  fair or stable allocation under load. nashgate's router is trained
  to find one.
```

## What this is

An open-source, OpenAI-compatible gateway for routing LLM/agent traffic
across backends. Instead of static heuristics, the routing policy is a
Nash-equilibrium-seeking multi-agent RL controller: each requester
(agent, tenant, workflow) is a player with its own reward (latency,
cost, success), and the router allocates requests toward an
equilibrium — stable, and no single player can improve outcomes by
unilaterally hammering one backend.

The core algorithm is a direct port of Nash-SAC, developed and
validated for edge-container scheduling under contention. Same shape
of problem — multiple self-interested agents competing for scarce,
rate-limited resources — different domain.

## Why this over LiteLLM / Bifrost / Portkey

They're mature, fast, and solve routing for the single-requester case
well. None of them model contention between concurrent agents as a
game. Under real multi-agent load (many subagents, many tenants, one
shared rate limit) a fixed weight table doesn't hold — someone starves
while someone else churns evictable capacity. That's the gap.

## Status

Idea capture + scaffold only — no routing logic implemented yet.
Next: define the game (players, actions, reward), port the Nash-SAC
policy network, wire up an OpenAI-compatible proxy layer to test it
against synthetic multi-agent traffic vs. LiteLLM's static routers.

## Layout

```
nashgate/
├── nashgate/
│   ├── gateway/    OpenAI-compatible proxy (the thing agents talk to)
│   ├── router/      request → backend selection, wraps policy/
│   ├── policy/       Nash-SAC equilibrium-seeking controller
│   └── cli/           nashgate route / policy / bench
├── tests/
└── docs/
```
