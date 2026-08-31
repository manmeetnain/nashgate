# Contributing to nashgate

## Setup

```bash
git clone https://github.com/manmeetnain/nashgate.git
cd nashgate
pip install -e ".[dev]"
pytest
```

93+ tests, no external services required — the gateway tests mock the
HTTP layer with `httpx.MockTransport`, nothing hits a real backend.

## Layout

See the [README](README.md#layout) for the full module map. Short
version: `env/` is the routing game (obs/action/reward, shared by
training and serving), `policy/` is the Nash-SAC algorithm, `router/`
wires live requests to a trained policy, `gateway/` is the
OpenAI-compatible HTTP proxy, `bench/` compares against static
routing. Keep changes inside the layer they belong to — e.g. reward
shaping changes belong in `env/reward.py`, not duplicated into
`router/live_router.py`, since the whole point of that split is that
training and serving score requests identically.

## Before opening a PR

- `pytest` passes locally — CI runs the same suite on every push and PR.
- New behavior gets a test. A bug fix gets a test that fails without
  the fix.
- If you touch `env/features.py` or `env/reward.py`, check whether the
  change needs `nashgate/bench`'s baselines re-evaluated — they read
  the same observation, so a shape change affects them too.

## Code style

- No comments unless they explain something a reader couldn't get
  from the code itself — a non-obvious constraint, the reason a value
  is clamped, a failure mode a check exists to prevent. Match the
  existing modules for the bar to clear.
- No speculative abstraction. A file gets split up when it's doing two
  jobs, not in anticipation of a third.
- New dependencies need a real reason — check `pyproject.toml`'s
  `dependencies` before adding an import; CI will fail on anything
  missing there even if it happens to be installed locally (this bit
  us once already — see the git history around the `numpy` fix).

## What's open

The README's [Status](README.md#status) section is current as of the
last update to this file. Beyond that: real integration testing
against a live LLM backend (everything today is tested against
simulated or mocked traffic), streaming response support in the
gateway, and a proper (non-demo-scale) training script are all
open — check open issues before starting on something larger.

## License

By contributing, you agree your contribution is licensed under this
repo's [MIT license](LICENSE).
