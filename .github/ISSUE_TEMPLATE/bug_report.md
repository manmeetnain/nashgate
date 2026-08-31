---
name: Bug report
about: Something in nashgate isn't behaving the way it should
title: ""
labels: bug
assignees: ""
---

**Which layer is this in?**
`env/` (the routing game) · `policy/` (Nash-SAC) · `router/` (live wiring) · `gateway/` (the HTTP proxy) · `bench/` (baselines/comparison) · `cli/` · other

**What happened**
A clear description of the bug.

**What you expected**
What should have happened instead.

**To reproduce**
Steps, or a minimal snippet — e.g.:

```python
from nashgate.env import MultiAgentRoutingEnv, BackendConfig, CallerConfig
# ...
```

If it's a gateway issue, include the relevant part of your config
YAML (redact API keys/URLs if needed) and the request/response.

**Environment**
- nashgate version / commit: 
- Python version: 
- OS: 

**Additional context**
Logs, stack traces, anything else useful.
