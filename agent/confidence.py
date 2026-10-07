"""Confidence signals for the grammar node (closed loop 1; see eval/PROTOCOL.md).

Two signals, both answering "how likely is it that the error we are about to show is real?":
  verbalized   the model reports a confidence next to its verdict (one call)
  consistency  sample the same prompt k times at temperature > 0; the score is the share of
               samples that also report an error (k calls)
Both are OFF by default (GRAMMAR_CONF_MODE=off) until the held-out eval justifies a threshold.
"""
from typing import Optional

CONF_SUFFIX = (
    " Also return `confidence` (a number from 0 to 1): how likely it is that your verdict "
    "(has_error true or false) is correct. Be calibrated: 0.9 means you expect to be right "
    "9 times out of 10.")


def consistency_score(flags: list[Optional[bool]]) -> Optional[float]:
    """Share of successful samples that flagged an error. None if every sample failed."""
    ok = [f for f in flags if f is not None]
    return sum(ok) / len(ok) if ok else None
