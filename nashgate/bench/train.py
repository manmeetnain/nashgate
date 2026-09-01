"""Re-exported from nashgate.policy.train, which is the canonical
implementation — shared with `nashgate policy train`. Kept here so
`from nashgate.bench.train import train_policy` (and `nashgate.bench`'s
own __init__) keep working without callers needing to know it moved."""

from nashgate.policy.train import train_policy

__all__ = ["train_policy"]
