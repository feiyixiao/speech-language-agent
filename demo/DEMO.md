# Demo walkthrough (about 6 minutes)

Shows what the agent does, what it does when things go wrong, and what it cannot do yet.

## Run it

```bash
cp .env.example .env              # add GROQ_API_KEY
pip install -r requirements-dev.txt
python demo/run_demo.py           # starts the server, runs 7 steps, stops it
python demo/run_demo.py --pause   # same, but waits for Enter between steps (live walkthrough)
```

First start loads the embedding model (10 to 20 s). The free Groq tier is rate limited, so a step can be slow;
the client waits for the provider's `retry-after` instead of failing.
`demo/sample_output.txt` is a real run, for reading without an API key.

Other ways to show it: `uvicorn api.main:app` then open `http://localhost:8000/docs` (interactive API docs),
or `streamlit run ui/app.py` (web UI, typed text or an audio file).

## The seven steps

| # | Step | What to point at |
|---|---|---|
| 1 | `/readyz` | Every configured model answers. Added after incident #1, where a provider retired the default model. |
| 2 | German word-order error | The router picks the practice path, grammar returns a typed result with an error category, not free text. |
| 3 | A question about the language | The router picks the RAG path; the answer cites the grammar sections it retrieved. |
| 4 | A correct English sentence | No invented error. False alarms are one of the things the evaluation measures. |
| 5 | Streaming | One event per finished node. Grammar and vocabulary run in parallel, so their order can change between runs. |
| 6 | Trace report | Latency p50 and p95, fallback rate, degraded rate and cost per request, for this run only. |
| 7 | "Kannst du mich helfen?" | A known miss (it should be "mir"). Shown on purpose: German dative errors are the weakest category. |

## Then show the evidence behind the demo

1. `INCIDENTS.md`: four incidents, each with how it was found, the fix and how the fix was measured.
2. `eval/RESULTS.md`: router 0.86 / 0.80 / 0.94, grammar recall against false alarms, judge calibration, and the confidence-gating result, which is null.
3. `agent/graph.py` (about 60 lines) and `agent/llm.py` (the fallback policy).

## What this demo does not show

- Pronunciation scoring: it needs audio and Azure Speech keys, and it has not been evaluated.
- Anything deployed: the Dockerfile and CI exist, there is no running deployment and there are no users.
- The vocabulary node is not covered by the evaluation sets; its suggestions are of mixed quality.

## If something goes wrong

| Symptom | Cause |
|---|---|
| Server does not start | Missing dependency or key. Read `demo/server.log`. |
| A step takes 10+ seconds | Free-tier rate limit; wait. |
| `degraded` is not "none" | A node failed and the rest still answered. That is the intended behaviour. |
| Different wording than `sample_output.txt` | LLM output varies between runs. |
