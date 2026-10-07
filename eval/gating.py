"""Threshold gating: show a flag only if its confidence score >= tau.

An item is a dict: gold (bool: the thing really is an error / transcript really is wrong),
flag (bool: the system flagged it), score (float in 0..1, higher = more confident, or None).
Rule: suppressed = score is not None and score < tau.  shown = flag and not suppressed.
tau = 0 suppresses nothing. Items with score None are never suppressed.
"""
from eval.stats import f_beta


def evaluate(items, tau):
    tp = fp = fn = tn = 0
    sup_all = sup_flagged = n_flagged = 0
    for it in items:
        sup = tau is not None and it["score"] is not None and it["score"] < tau
        sup_all += sup
        n_flagged += it["flag"]
        sup_flagged += it["flag"] and sup
        shown = it["flag"] and not sup
        if it["gold"]:
            tp += shown
            fn += not shown
        else:
            fp += shown
            tn += not shown
    n = len(items)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tau": tau, "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "false_alarm": fp / (fp + tn) if fp + tn else 0.0,
            "suppressed_flag_rate": sup_flagged / n_flagged if n_flagged else 0.0,
            "repeat_rate": sup_all / n if n else 0.0}


def candidate_taus(items):
    """0 (no gate) plus the midpoint between each pair of neighbouring distinct scores, so a threshold
    never sits exactly on an observed score (suppressed = score < tau)."""
    s = sorted({it["score"] for it in items if it["score"] is not None})
    return [0.0] + [(a + b) / 2 for a, b in zip(s, s[1:])]


def curve(items):
    return [evaluate(items, t) for t in candidate_taus(items)]


def choose_tau(items, beta: float = 0.5, max_rate: float | None = None, rate_key: str = "repeat_rate"):
    """Pick tau on DEV items: maximise F_beta of detection, optionally with rate_key <= max_rate.
    Ties go to the lower tau (more recall). Returns the evaluate() row of the chosen tau."""
    best = None
    for t in candidate_taus(items):
        row = evaluate(items, t)
        if max_rate is not None and row[rate_key] > max_rate:
            continue
        f = f_beta(row["precision"], row["recall"], beta)
        if best is None or f > best[0] + 1e-12:
            best = (f, row)
    return best[1] if best else evaluate(items, 0.0)
