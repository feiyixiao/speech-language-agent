# Speech Language-Learning Agent

A LangGraph agent for language learners. You say or type a sentence in English or
German (Japanese and Mandarin are partly supported) and get back grammar corrections,
vocabulary suggestions and word-level pronunciation scores. You can also ask a question
about the language and get an answer from a small grammar knowledge base.

It started as a local Streamlit prototype. This version is the "make it work beyond my
laptop" pass: a typed, tested, traced FastAPI service with an offline eval harness, and a
log of what broke and how each fix was measured ([INCIDENTS.md](INCIDENTS.md)).

```
                  ┌──practice──▶ grammar ──────┐
text / audio ─ASR─▶ router      vocabulary ─────┼──▶ response (partial if a node fails)
                  │             pronunciation ──┘      (fan-out runs concurrently)
                  └──question──▶ RAG (Chroma) ─────▶ answer
```

| Piece | Choice | Why |
|---|---|---|
| Orchestration | LangGraph `StateGraph`, async nodes, conditional fan-out | The router picks a branch. Practice nodes run in parallel, and each owns its own state keys. |
| LLM outputs | JSON-schema-constrained + Pydantic validation (`agent/schemas.py`) | The prototype parsed free text. Its "was there an error?" check was a substring match that misfired (see eval). |
| Providers | Ordered model chain with per-error policy (`agent/llm.py`) | 404 → skip, 429 → wait for `retry-after`, 5xx/timeout → backoff, bad JSON → next model. |
| ASR | Groq-hosted Whisper large-v3-turbo, local whisper as fallback | Keeps torch/whisper out of the container. |
| Pronunciation | Azure Pronunciation Assessment (word-level) | Used only when audio is present. The node degrades if keys are missing. |
| API | FastAPI: `/v1/feedback`, `/v1/feedback/audio`, `/v1/feedback/stream` (SSE, one event per node), `/healthz`, `/readyz` | A partial failure still returns 200 with `degraded: [...]`. 503 only if every provider is down. |
| Observability | Per-request JSONL traces (`agent/tracing.py`) + `scripts/trace_report.py` | Spans record node, model, latency, tokens, cost and fallback. The report gives p50/p95, error/degraded/fallback rate and cost per request. |
| Deploy | Dockerfile (Cloud Run), GitHub Actions: tests → live model check → manual deploy | The embedding model is baked into the image, so there's no download on cold start. |

## Results

Offline eval: 79 grammar sentences (EN/DE, 43 with errors) and 50 routing cases.
v1 is the prototype's prompts and parsing, verbatim. All variants ran on the same model
(`gpt-oss-20b`) with fallback disabled. Full table and caveats: [eval/RESULTS.md](eval/RESULTS.md).

| | v1 prototype | v2 structured | v3 current |
|---|---|---|---|
| Router accuracy | 0.86 | 0.80 | **0.94** (v2→v3: 7 fixed / 0 broken, McNemar p = 0.016) |
| Grammar detection F1 | 0.889 | **0.920** | same prompt as v2 |
| … recall / false-alarm rate | 0.84 / 0.056 | **0.93** / 0.111 ⚠️ | |
| German F1 | 0.783 | **0.846** | |
| Correction quality (exact / calibrated LLM judge) | n/a | 0.90 / 0.925 | |

What the numbers say, including the parts that don't flatter the change:
- Structured output made routing *worse* at first. The fix was a language-aware router
  prompt, tested as an explicit hypothesis. v3 vs v1 is not significant (p = 0.29).
- Grammar recall went up, but false alarms doubled. One false alarm "corrected" a correct
  German sentence into a wrong one. That trade-off is the next thing to fix.
- The LLM judge was checked before its numbers were used. It accepts 100% of exact-match
  corrections and rejects 100% of negative controls.

Reliability (fault injection, `scripts/fault_injection.py`, primary model set to a retired one):

| | errors | degraded | fallback used |
|---|---|---|---|
| before retry-after fix | 0% | 30% | 100% |
| after | 0% | **0%** | 100% |

Latency per LLM call (healthy traces): router p50 270 ms, grammar p50 ≈ 390 ms / p95 ≈ 1.1 s.
Cost is about $0.00015 per request on list prices (approximate).

## Run it

```bash
cp .env.example .env               # GROQ_API_KEY (+ AZURE_SPEECH_KEY / REGION for pronunciation)
pip install -r requirements-dev.txt
uvicorn api.main:app --reload      # API on :8000, docs at /docs
streamlit run ui/app.py            # demo UI
pytest -q                          # 36 tests, no network
python -m eval.run_eval --variants v1 v3   # needs GROQ_API_KEY; ~15 min on the free tier
python scripts/trace_report.py     # summarise traces/
```

```bash
curl -N -X POST localhost:8000/v1/feedback/stream -H 'content-type: application/json' \
     -d '{"text": "Gestern ich habe Fußball gespielt.", "target_language": "German"}'
```

Deploy (Cloud Run, after `gcloud auth login` and creating the two secrets):

```bash
gcloud run deploy speech-agent --source . --region europe-west3 --memory 2Gi \
  --max-instances 2 --allow-unauthenticated \
  --set-secrets GROQ_API_KEY=groq-api-key:latest,AZURE_SPEECH_KEY=azure-speech-key:latest \
  --set-env-vars AZURE_SPEECH_REGION=westeurope
```

## Demo

`python demo/run_demo.py` starts the server and runs a scripted walkthrough: readiness check, a German word-order error,
a question answered through RAG, a correct sentence, streaming, a trace report, and one known failure shown on purpose.
Walkthrough and talking points: [demo/DEMO.md](demo/DEMO.md). A real run is in [demo/sample_output.txt](demo/sample_output.txt).

## Closed loops: confidence-gated feedback (loop 1 run once: null result; loop 2 not run)

Two measure -> gate -> held-out test loops, both OFF by default. Protocol written before any run: [eval/PROTOCOL.md](eval/PROTOCOL.md).

- **Grammar.** False alarms doubled when recall went up, so a flagged error can now be gated on a confidence score
  (`GRAMMAR_CONF_MODE=verbalized|consistency`, `GRAMMAR_CONF_THRESHOLD`). Suppressed errors stay in `suppressed_errors`.
  `python -m eval.conf_eval collect | analyze | analyze --final` on the dev/test split in `eval/data/grammar_split.json`.
- **Speech.** Whisper may "autocorrect" a learner's mistake before the grammar node sees it, and Azure is scored against the
  Whisper transcript. `/v1/feedback/audio` now returns `transcript_confidence` / `transcript_uncertain`
  (`ASR_CONF_FEATURE`, `ASR_CONF_THRESHOLD`), the pronunciation node reports `reference_mismatch_wer`, and
  `python -m eval.speech_eval` measures both on recordings of your own voice (`eval/audio/README.md`).

**Loop 1 result (2026-10-07, `gpt-oss-20b`, run once on the held-out split, protocol unchanged): null.**
Neither signal gave a threshold worth using: tau chosen on dev was 0 (suppress nothing) for both, because the problem the gate
targets did not show up in this run. The baseline false-alarm rate was 0.04 on dev (1 of 23 correct sentences) and 0.08 on test
(1 of 13), against 0.111 (4 of 36) in the earlier v2 run. With 1 false alarm per split there is nothing to separate, so test AUROC is
undefined (nan) and the gated and ungated rows are identical (McNemar p = 1.0). 5 sentences were dropped on dev/test because the baseline
call failed (rate limits), so n is 21 / 24 for the two signals. Read this as: "no evidence the gate helps, and no evidence it hurts;
the false-alarm problem is too rare in this seed set to test it." Next: more correct German sentences, a second annotator, and a
repeated baseline run to see how much of 0.111 vs 0.04 is sampling noise. Details in `eval/RESULTS.md`.

**Loop 2 (ASR confidence)** has not been run: it needs recordings of your own voice (`eval/audio/README.md`).

## Known limitations / next

- German case errors (dative) are the weakest category (*Kannst du mich helfen?* is still missed).
- The false-alarm rate on correct sentences. Next experiment: ask for a confidence score and
  only surface errors above a threshold tuned on the eval set.
- The generator's own `grounded` flag is unreliable. Groundedness should be judged separately.
- Eval data is a one-annotator seed set. Next: a second annotator plus a W&I+LOCNESS sample.
- Traces are local JSONL. On Cloud Run they should go to Cloud Logging with an alert on
  `fallback_used` and `/readyz`.
- Session memory (error types) lives in the client. There's no server-side persistence yet.
