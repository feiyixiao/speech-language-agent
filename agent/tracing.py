"""Minimal request tracing: one JSONL line per request, one span per LLM/ASR call.

No external account needed. Each request gets a trace_id; every call inside it
appends a span (node, model, latency, tokens, cost, error, whether a fallback
was used). `scripts/trace_report.py` turns the JSONL into p50/p95 latency,
error rate, fallback rate and cost per request.
"""
import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from agent.config import settings

_current: contextvars.ContextVar[dict | None] = contextvars.ContextVar("trace", default=None)


@contextmanager
def trace(name: str, **attrs):
    """Open a request-level trace. Nested calls record spans into it."""
    t = {"trace_id": uuid.uuid4().hex[:12], "name": name,
         "ts": datetime.now(timezone.utc).isoformat(), "attrs": attrs, "spans": []}
    token = _current.set(t)
    start = time.perf_counter()
    try:
        yield t
        t["status"] = "ok"
    except Exception as e:
        t["status"] = "error"
        t["error"] = f"{type(e).__name__}: {e}"[:300]
        raise
    finally:
        t["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        t["cost_usd"] = round(sum(s.get("cost_usd", 0) for s in t["spans"]), 6)
        t["fallback_used"] = any(s.get("fallback") for s in t["spans"])
        _current.reset(token)
        _write(t)


def record_span(**span):
    t = _current.get()
    if t is not None:
        t["spans"].append(span)


def current_trace_id() -> str | None:
    t = _current.get()
    return t["trace_id"] if t else None


def _write(t: dict):
    try:
        settings.trace_dir.mkdir(parents=True, exist_ok=True)
        day = t["ts"][:10]
        with open(settings.trace_dir / f"{day}.jsonl", "a") as f:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    except OSError:
        pass  # tracing must never take the request down
