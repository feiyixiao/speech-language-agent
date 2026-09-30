# Eval results

One row per run. Details in `eval/results/*.json`.

| time | variant | model | router acc | question F1 | grammar P / R / F1 | false alarm | DE F1 | correction exact / judge | grammar p50/p95 ms | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-30T22:20:44 | v1 | openai/gpt-oss-20b | 0.86 | 0.868 | 0.947 / 0.837 / 0.889 | 0.056 | 0.783 | — | 345.9 / 689.9 | 0 |
| 2026-09-30T22:25:49 | v2 | openai/gpt-oss-20b | 0.8 | 0.833 | 0.909 / 0.93 / 0.92 | 0.111 | 0.846 | 0.9 / 0.925 | 1978.1 / 3095.0 ⚠️ | 0 |
| 2026-09-30T22:29:39 | v3 | openai/gpt-oss-20b | 0.94 | 0.943 | — / — / — | — | — | — | — / — | 0 |

**Notes**

- ⚠️ The v2 latency is wrong. That run counted time spent waiting on 429 rate limits (138 in the run)
  as latency. Re-computed from the successful-call spans in the traces, grammar latency is
  **p50 380 ms / p95 1138 ms**, in line with v1. The eval now records only the successful call.
- The v2 → v3 router gain is significant on paired items: 7 fixed, 0 broken, exact McNemar **p = 0.016**.
  v3 vs v1 is *not* significant (6 fixed, 2 broken, p = 0.29). So the honest claim is: "structured
  output hurt routing until the prompt was made language-aware, and it now matches or beats the prototype."
- Grammar (v2 = v3 prompt): recall 0.84 → 0.93 and German F1 0.78 → 0.85, **but the false-alarm rate on correct
  sentences doubled, 0.056 → 0.111 (2 → 4 of 36)**. One false alarm is harmful: it "corrected" a correct
  *weil … lernen muss* into wrong word order. This trade-off is the next thing to work on.
- Correction judge (gpt-oss-120b) was calibrated before use. It accepted 100% of exact-match corrections and
  rejected 100% of negative controls (the uncorrected sentence offered as the correction).
- Seed data: 79 + 50 items written and labelled by one person (me). This is enough to catch
  regressions, too small for fine-grained claims. Next: a second annotator plus a W&I+LOCNESS sample.
