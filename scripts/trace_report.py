"""Summarise traces/*.jsonl: volume, error / degraded / fallback rates, latency, cost.

    python scripts/trace_report.py            # all days
    python scripts/trace_report.py 2026-09-30 # one day
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.config import settings  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))]) if xs else "-"


def main(day=None):
    files = sorted(settings.trace_dir.glob(f"{day or '*'}.jsonl"))
    traces = [json.loads(l) for f in files for l in open(f)]
    traces = [t for t in traces if t["name"].startswith("feedback")]
    if not traces:
        print("no traces"); return
    n = len(traces)
    lat = [t["latency_ms"] for t in traces]
    print(f"requests: {n}   errors: {sum(t['status'] != 'ok' for t in traces) / n:.1%}   "
          f"degraded: {sum(bool(t['attrs'].get('degraded')) for t in traces) / n:.1%}   "
          f"fallback used: {sum(t['fallback_used'] for t in traces) / n:.1%}")
    print(f"latency p50/p95/max: {pct(lat, .5)} / {pct(lat, .95)} / {round(max(lat))} ms   "
          f"cost/request: ${sum(t['cost_usd'] for t in traces) / n:.6f}")
    print("intents:", dict(Counter(t["attrs"].get("intent") for t in traces)))

    per_node = defaultdict(list)
    fails = Counter()
    for t in traces:
        for s in t["spans"]:
            per_node[(s["node"], s["model"])].append(s)
            if s["status"] != "ok":
                fails[(s["node"], s["model"], s.get("error", s["status"])[:60])] += 1
    print("\nnode / model                               calls  fail%  p50ms  p95ms")
    for (node, model), ss in sorted(per_node.items()):
        l = [s["latency_ms"] for s in ss]
        print(f"{node:12} {model:30} {len(ss):5}  {sum(s['status'] != 'ok' for s in ss) / len(ss):5.1%}"
              f"  {pct(l, .5):>5}  {pct(l, .95):>5}")
    if fails:
        print("\ntop failures:")
        for (node, model, err), c in fails.most_common(5):
            print(f"  {c:4}× {node} {model}: {err}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
