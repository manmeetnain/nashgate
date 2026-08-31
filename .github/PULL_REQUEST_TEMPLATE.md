## What this does

<!-- One or two sentences: what changed and why. -->

## Which layer(s)

<!-- env/ (the routing game) · policy/ (Nash-SAC) · router/ (live wiring)
     gateway/ (the HTTP proxy) · bench/ (baselines/comparison) · cli/ · other -->

## Checklist

- [ ] `pytest` passes locally
- [ ] New behavior has a test; a bug fix has a test that fails without the fix
- [ ] If this touches `env/features.py` or `env/reward.py` — checked whether
      `bench/`'s baselines need re-evaluating, since they read the same
      observation (see [CONTRIBUTING.md](../CONTRIBUTING.md#layout))
- [ ] Any new dependency is declared in `pyproject.toml`, not just installed locally
- [ ] README/CHANGELOG updated if this changes documented behavior

## Related issues

<!-- Closes #123, or "n/a" -->
