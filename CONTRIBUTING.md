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
last update to this file. Closed since the project's early releases:
the gateway (streaming and non-streaming) is checked against a real,
live Anthropic backend automatically on every push to main
(`.github/workflows/real-api-check.yml` + `tests/test_real_api.py`,
gated on a `NASHGATE_ANTHROPIC_TEST_KEY` repo secret — skips cleanly,
locally too, when that secret isn't set); and both real configs
(`docs/example.config.yaml`, `docs/severe_contention.config.yaml`)
have 10-seed multi-seed validation, not just single anecdotal runs —
see [README#multi-seed-validation](README.md#multi-seed-validation).
A real, multi-seed-validated `severe_contention_100k` checkpoint has
also been run against genuinely concurrent real traffic (16 requests,
`asyncio.gather`, not sequential) — see
[README#verified-under-real-concurrent-traffic](README.md#verified-under-real-concurrent-traffic).
And routing that reacts to genuinely different real backend latencies
(not just simulation, and not just one real provider differentiated by
config) has been verified against two real providers (Anthropic +
OpenAI) — see
[README#verified-with-real-latency-differentiated-routing](README.md#verified-with-real-latency-differentiated-routing).
Rate limits are enforced in live serving too, not just advisory — see
[README#the-router](README.md#the-router).

Still open:

- **Tuning past demo-scale.** `nashgate policy train`'s multi-seed
  results show the training loop converges reliably and
  reproducibly — that's a different claim from "these are
  well-tuned policies." No LR schedule, no eval-during-training, no
  hyperparameter search has been done.

Check open issues before starting on something larger.

## License

By contributing, you agree your contribution is licensed under this
repo's [MIT license](LICENSE).
