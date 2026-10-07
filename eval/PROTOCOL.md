# Protocol for the two closed loops (written BEFORE any real run)

Both loops follow the same shape: **measure a problem -> add a confidence signal -> gate on a threshold chosen
on dev only -> evaluate once on a held-out test split -> record the result, including a negative one.**
Anything decided after looking at test numbers is not allowed to be called "held-out".

Split: `eval/data/grammar_split.json` (79 sentences -> 50 dev / 29 test, stratified by language x has_error,
fixed seed). The same split is used for the speech recordings. Test is touched once per loop
(`FINAL_*.json` lock files in `eval/results/`; `--force` exists but breaks the held-out claim).

## Loop 1 — grammar: show an error only when we are confident it is real

Problem (already measured): recall rose 0.84 -> 0.93 but false alarms on correct sentences doubled
(0.056 -> 0.111; `eval/RESULTS.md`).

| signal | how | cost |
|---|---|---|
| `verbalized` | the model returns `confidence` (0..1) with its verdict (`GrammarFeedbackConf`, prompt + `CONF_SUFFIX`) | 1 call |
| `consistency` | same prompt as the baseline, sampled k=5 times at T=0.7; score = share of samples that also report an error | 1 + k calls |

Gate: a flagged error is shown iff `score >= tau`; otherwise it is suppressed (kept in `suppressed_errors`,
not shown, not added to the error history). The gate never adds flags.
tau rule (fixed in advance): **maximise F0.5 of error detection on dev** (precision weighted over recall because a
false correction misleads the learner); ties -> lower tau. Candidate taus are midpoints between observed scores.

Run: `python -m eval.conf_eval collect` (~550 calls) -> `analyze` (dev) -> `analyze --final` (test, once).

Reported on test: precision / recall / false-alarm rate for baseline, ungated and gated; 95% bootstrap CIs;
exact McNemar on paired per-item correctness (gated vs baseline, gated vs own ungated); AUROC and ECE (3 bins)
of the score over flagged items. **n_test = 29 (about 16 real errors, 13 correct sentences)**: intervals will be wide;
the claim that can be made is a direction, not an effect size.

Confound to state: for `verbalized`, the confidence prompt can also change the verdicts, so the ungated row of that
signal is reported next to the baseline. `consistency` gates the baseline verdicts themselves.

## Loop 2 — speech: do we trust the transcript before grading the grammar on it?

Two problems that cannot be seen from text-only evals:
1. **ASR autocorrection.** Whisper has a language-model prior; a learner who says "She go to school" may be transcribed
   "She goes to school.", and the grammar node then never sees the error. Measured by recording yourself reading the eval
   sentences verbatim, including mistakes (`eval/audio/READING_SHEET.md`).
2. **Circular pronunciation reference.** Azure is scored against the Whisper transcript (`pronunciation_node`). A word
   Whisper "corrected" is scored against the corrected word. `reference_mismatch_wer` = disagreement between Azure's
   own recognition and the Whisper transcript.

Signals (higher = more confident): Whisper segment features `avg_prob`, `min_prob`, `no_speech`, `compression`
(`agent/asr_conf.py`) and, optionally, `azure_agree = 1 - WER(whisper, azure)`.
Target for signal quality: `transcript wrong` = WER(text read, ASR text) > 0 (normalised).
Gate: if `score < tau` the transcript is `transcript_uncertain`: grammar errors are suppressed and the client should
ask the learner to repeat. tau rule: **maximise F0.5 of end-to-end detection on dev subject to asking to repeat
<= 20% of the items**. The feature is the one with the best dev AUROC unless `--feature` is given.

Conditions: `clean` (quiet room) and `noisy` (`scripts/add_noise.py --snr 5`, or a real noisy recording).
If the clean condition has no transcript errors the gate cannot be evaluated and the analysis says so; that is a
result (clean read speech is easy), and the noisy condition is the one that tests the gate.

Run: record -> `python -m eval.speech_eval collect --condition clean [--azure]` -> `analyze` (dev + descriptive) ->
`analyze --feature <f> --final`.

## What these loops can and cannot show
- They show whether a confidence signal separates real errors from false alarms / right from wrong transcripts **on this
  data**: one speaker (you), one voice, read speech, 79 short sentences, one annotator, one provider.
- They do not show calibration in general, robustness to other speakers or accents, or that learners prefer the gated feedback.
- Gate defaults are OFF in production (`GRAMMAR_CONF_MODE=off`, no thresholds) until a result justifies turning one on.

## Write-up template (for INCIDENTS.md / RESULTS.md after the real runs)
Problem (number) -> hypothesis -> signal -> dev threshold -> test result with CI and n -> what it does not show ->
decision (turn on / leave off / next experiment). Report a null or negative result the same way as a positive one.
