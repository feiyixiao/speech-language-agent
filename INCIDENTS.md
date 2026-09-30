# Incident log

Short postmortems: what broke, how it was found, what was measured to confirm the fix.

---

## #1 — Default LLM retired by the provider → every request failed (2026-09-30)

**Impact.** Every grammar / vocabulary / router / RAG call returned
`404 model_not_found`. The Streamlit app showed a stack trace on every click.
100% of feedback requests failed. Unknown since when — there was no monitoring.

**How it was found.** Not by an alert (there were none). I ran a manual smoke
call before starting the Phase 1 work, and `ChatGroq(model="llama-3.1-8b-instant")`
returned 404. `groq.models.list()` confirmed that neither `llama-3.1-8b-instant`
nor `llama-3.3-70b-versatile` is served any more.

**Root cause.** The model name was hard-coded in three places (`nodes.py` ×2,
`rag.py`), with no fallback, no health check and no test that touched the
real provider.

**Fix.**
1. Model names moved to config (`LLM_MODELS`) as an ordered fallback chain
   (`agent/config.py`).
2. `agent/llm.py`: 404/400 → skip to next model immediately; 429/5xx/timeout →
   retry with backoff, then fall back; invalid JSON → next model. Every attempt
   is a span in the trace, with `fallback: true` when not the primary.
3. `GET /readyz` pings every configured model; `scripts/check_models.py`
   fails CI if any configured model is missing from `models.list()`.
4. Regression test: `tests/test_agent.py::test_retired_model_falls_back_without_retrying`.

**Verification (measured).**
- Before: 100% failure on the old code path.
- After: the offline eval ran 129 calls with 0 failures (`eval/RESULTS.md`).
- Fault injection (`scripts/fault_injection.py`, 10 requests,
  `LLM_MODELS=llama-3.1-8b-instant,openai/gpt-oss-20b`): 0 errors, and the
  traces show `fallback_used` on 100% of requests. A 404 hop costs ~20–50 ms,
  so a retired model is cheap to skip. The same run exposed #3.

**What I'd still add.** An alert (Cloud Monitoring log-based metric on
`fallback_used`, or on `/readyz` != ok) — a fallback silently doubling
latency is the next incident.

---

## #2 — Router sent learner *question sentences* to the knowledge branch (2026-09-30)

**Impact.** "Kannst du mich helfen?" (a German practice sentence with a case
error — should be *mir*) was routed as a *question about German*. The RAG
branch answered with a rewrite and marked it `grounded: true`; the grammar
error was never flagged.

**How it was found.** Manual test through the streaming endpoint, then
confirmed on the router eval set (see `eval/RESULTS.md`, router errors listed
in `eval/results/*.json`).

**Fix & measurement.** Hypothesis: the router doesn't know the target
language, so it can't tell "a question *in* German" from "a question *about*
German". I made the router prompt language-aware and added contrastive
examples. I checked that none of the examples are eval items (my first draft
had three near-duplicates, which I replaced). Router accuracy on the same 50
items went from 0.80 to 0.94, with 7 items fixed and 0 broken (McNemar p = 0.016).
Question recall stayed at 1.0. Remaining misroutes: "Can you help me with my
homework?" (arguably a real request to the tutor, so the label is debatable),
"Wie spät ist es?", and "What do you think about the new teacher?".

**Not fixed.** Once routed correctly, the grammar node still misses the case
error in *Kannst du mich helfen?* (mich → mir). German dative errors are the
weakest category in the grammar eval.

**Also learned.** The model's self-reported `grounded` flag is not reliable
evidence of grounding — it said `true` for an answer the context did not
support. Next step: judge groundedness separately instead of trusting the
generator.

---

## #3 — Fallback chain collapsed onto one rate-limited model → 30% degraded (2026-09-30)

**Impact.** In the fault-injection run for #1, 3 of 10 requests came back
with `degraded: ["grammar"]`. The user got vocabulary feedback but no
grammar feedback.

**How it was found.** `scripts/trace_report.py` on the fault-injection
traces. `top failures` showed 9× `RateLimitError` on the only healthy model.

**Root cause.** With the primary retired, all traffic went to one model. The
free tier allows 8k tokens/min *per model*. My 429 retries waited 0.5 s and
1 s, which is far shorter than the provider's `retry-after`, so every retry
burned an attempt and the node gave up.

**Fix.** On 429 the client now waits for the provider's `retry-after`, capped
at `LLM_MAX_WAIT_S`. Unit test: `test_retry_after_header_is_honoured`.
Operationally, the default chain uses models with *separate* rate-limit buckets.

**Verification.** I re-ran the identical fault-injection script: degraded went
from **30% to 0%** (10/10 full responses), with 0 errors.

**Related finding in my own tooling.** The first v2 eval reported p50 latency
of 1978 ms against v1's 346 ms, which looked like a 6× regression. It was the
same rate-limit waiting, counted as model latency. The traces showed the real
figures, grammar p50 380 ms and p95 1138 ms. The eval now times only the
successful call.

---

## #4 — First request after start took ~9 s (2026-09-30)

**Found.** The p95 in the healthy trace report was 9.2 s, but p50 was 0.9 s.
The one slow request was the first knowledge question, which loaded the
embedding model and Chroma index lazily.

**Fix.** A FastAPI lifespan hook warms the vector store before serving. The
Docker build also bakes the embedding model into the image, so Cloud Run cold
starts skip the download.

**Verification.** After a fresh restart, the first knowledge question took
**1.15 s** (was ~9 s) and the second took 0.61 s.
