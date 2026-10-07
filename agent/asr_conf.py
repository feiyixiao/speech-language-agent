"""ASR confidence features from Whisper segments (closed loop 2; see eval/PROTOCOL.md).

Whisper's verbose output gives, per segment: avg_logprob, no_speech_prob, compression_ratio.
We turn them into scores where HIGHER = MORE CONFIDENT that the transcript is right:
  avg_prob      duration-weighted exp(avg_logprob)  (geometric-mean token probability)
  min_prob      the weakest segment's exp(avg_logprob)
  no_speech     1 - max no_speech_prob
  compression   1 / max compression_ratio            (repetition loops have a high ratio)
"""
import math
from typing import Optional

FEATURES = ("avg_prob", "min_prob", "no_speech", "compression")


def _get(seg, key):
    return seg.get(key) if isinstance(seg, dict) else getattr(seg, key, None)


def asr_features(segments) -> dict:
    segs = [s for s in (segments or []) if _get(s, "avg_logprob") is not None]
    if not segs:
        return {}
    weights = [max((_get(s, "end") or 0) - (_get(s, "start") or 0), 1e-3) for s in segs]
    probs = [math.exp(min(0.0, _get(s, "avg_logprob"))) for s in segs]
    ns = [_get(s, "no_speech_prob") or 0.0 for s in segs]
    cr = [_get(s, "compression_ratio") or 1.0 for s in segs]
    return {"avg_prob": sum(w * p for w, p in zip(weights, probs)) / sum(weights),
            "min_prob": min(probs),
            "no_speech": 1.0 - max(ns),
            "compression": 1.0 / max(max(cr), 1.0),
            "n_segments": len(segs)}


def gate(features: dict, feature: str, threshold: Optional[float]) -> tuple[Optional[float], bool]:
    """(score, uncertain). Never uncertain when the gate is off or the feature is missing."""
    score = features.get(feature) if features and feature in FEATURES else None
    if score is None or threshold is None:
        return score, False
    return score, score < threshold
