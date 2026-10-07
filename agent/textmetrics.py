"""Text normalisation and word error rate, shared by the API code and the evals."""
import re
import unicodedata


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().replace("’", "'")
    return re.sub(r"[^\w' ]", "", re.sub(r"\s+", " ", s)).strip()


def words(s: str) -> list[str]:
    return norm_text(s).split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate of `hyp` against `ref` (Levenshtein over normalised words)."""
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)
