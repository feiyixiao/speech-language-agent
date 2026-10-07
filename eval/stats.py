"""Evaluation statistics. Pure functions, no network; tested in tests/test_eval_stats.py."""
import random
from math import comb


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def f_beta(p: float, r: float, beta: float = 0.5) -> float:
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar. b = only A correct, c = only B correct (paired items)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def ece(conf, correct, n_bins: int = 3) -> float:
    """Expected calibration error, equal-width bins: sum_b (n_b/N) |acc_b - mean_conf_b|."""
    n = len(conf)
    if n == 0:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(conf, correct):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, float(y)))
    return sum(len(b) / n * abs(sum(y for _, y in b) / len(b) - sum(c for c, _ in b) / len(b))
               for b in bins if b)


def auroc(scores, labels) -> float:
    """P(score of a random positive > score of a random negative); ties count 1/2."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def bootstrap_ci(n_items: int, stat, n: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI of stat(indices) over item resamples. stat may return None/NaN (skipped)."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        v = stat([rng.randrange(n_items) for _ in range(n_items)])
        if v is not None and v == v:
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return vals[int(alpha / 2 * len(vals))], vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
