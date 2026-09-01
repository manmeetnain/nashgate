# Contributing to nashgate

## Setup

```bash
git clone https://github.com/manmeetnain/nashgate.git
cd nashgate
make dev     # pip install -e ".[dev]"
make check   # lint + test — what CI runs
```

`make help` lists every command (`test`, `lint`, `format`, `route`,
`bench`, `docker-build`, `compose-up`, `clean`, ...) — see the
[Makefile](Makefile).

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

- `make check` passes locally — CI runs the same lint + test suite on
  every push and PR.
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
last update to this file. The gateway (streaming and non-streaming) is
checked against a real, live Anthropic backend automatically —
`.github/workflows/real-api-check.yml` runs `tests/test_real_api.py`
on every push to main, using a `NASHGATE_ANTHROPIC_TEST_KEY` repo
secret (a funded key dedicated to CI, ideally with its own spend cap).
Without that secret set, the same tests just skip — locally too, so
they never cost anything unless you export the same variable name with
a real key when running `pytest` yourself. See
[README#verified-against-a-real-api](README.md#verified-against-a-real-api)
for what this actually caught the first time it ran.

Moving `nashgate policy train` past demo-scale (LR schedule,
eval-during-training, multi-seed) is still open — check open issues
before starting on something larger.

## License

By contributing, you agree your contribution is licensed under this
repo's [MIT license](LICENSE).
