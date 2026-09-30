"""Offline eval: router intent + grammar error detection + correction quality.

    python -m eval.run_eval                       # v2 (current) on default model
    python -m eval.run_eval --variants v1 v2      # before/after comparison
    python -m eval.run_eval --model openai/gpt-oss-120b

v1 = the July prototype's prompts and parsing, verbatim (free-text router reply,
"grammatically correct" substring check). v2 = structured outputs, first router
prompt. v3 = current agent/nodes.py (language-aware router). Grammar prompt is
the same in v2 and v3.
Both run on the SAME model with fallback disabled, so the comparison isolates
the prompt/contract change. Results go to eval/results/*.json and a row is
appended to eval/RESULTS.md.
"""
import argparse
import asyncio
import json
import re
import statistics
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from agent import nodes
from agent.llm import LLMUnavailable, _cost, _model_kwargs, get_client, structured_call
from agent.schemas import GrammarFeedback, RouterDecision
from agent.tracing import trace

HERE = Path(__file__).parent
SEM = asyncio.Semaphore(2)  # free tier: 8k tokens/min per model -> keep concurrency low
RL_WAIT_S = 15

# v2 router prompt, frozen for reproducibility (v3 = the current agent/nodes.py prompt,
# adopted after it won on this eval: router acc 0.80 -> 0.94).
V2_ROUTER = (
    "You route inputs from a language learner. Decide the intent:\n"
    "- practice: the learner wrote/said a sentence in the target language that should be "
    "checked (even if it is a question sentence like 'Where you are going?' — that is practice).\n"
    "- question: the learner asks ABOUT the language: grammar rules, word meaning, usage, "
    "pronunciation (e.g. 'When do I use the dative?', 'How do you pronounce th?').\n"
    "Return JSON."
)

# ---------------------------------------------------------------- v1 (prototype) prompts
V1_ROUTER = (
    "You are classifying a language learner's sentence. "
    "Decide what kind of feedback would be MOST useful:\n"
    "- grammar: the sentence has a clear grammatical error\n"
    "- vocabulary: the sentence is correct but word choices are unnatural or repetitive\n"
    "- pronunciation: the user is asking about how to pronounce something\n"
    "- knowledge_query: the user is asking a grammar or language question (e.g. 'when do I use...', 'what is...', 'how do I...')\n"
    "If the sentence is correct and natural, reply 'vocabulary'.\n"
    "Reply with only one word: grammar, vocabulary, pronunciation, or knowledge_query.")
V1_GRAMMAR = (
    "You are a language teacher. The user is learning {lang}. "
    "First, check if the sentence has any grammar errors. "
    "If the sentence is correct, say 'Your sentence is grammatically correct!' and give one brief compliment. "
    "If there are errors, identify each one, explain briefly, and provide the corrected version. "
    "Do NOT invent errors that do not exist.\n\n"
    "Past errors this session: none yet\n"
    "If the current error matches a past one, start your response with "
    "'You made this mistake before: [error]. Let's fix it again!'")


async def raw_call(model, system, user, stats):
    for attempt in range(8):
        try:
            t = time.perf_counter()
            r = await get_client().chat.completions.create(
                model=model, temperature=0, **_model_kwargs(model),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            stats.append({"latency_ms": (time.perf_counter() - t) * 1000,
                          "cost": _cost(model, r.usage)})
            return r.choices[0].message.content or ""
        except Exception as e:
            if "rate" not in str(e).lower() and "429" not in str(e):
                raise
            await asyncio.sleep(RL_WAIT_S)
    raise RuntimeError("rate limited")


async def v2_call(model, node, system, user, schema, stats):
    for attempt in range(8):
        t = time.perf_counter()
        with trace(f"eval_{node}") as tr:
            try:
                out = await structured_call(node, system, user, schema, models=[model])
            except LLMUnavailable as e:
                if "RateLimit" in str(e) and attempt < 7:
                    await asyncio.sleep(RL_WAIT_S)
                    continue
                raise
        # latency = the successful model call only; 429 waits are a quota artefact, not the model
        # (first v2 run reported p50 1978 ms because it counted them — see RESULTS.md note)
        ok = [s for s in tr["spans"] if s["status"] == "ok"]
        stats.append({"latency_ms": ok[-1]["latency_ms"] if ok else (time.perf_counter() - t) * 1000,
                      "cost": sum(s.get("cost_usd", 0) for s in tr["spans"])})
        return out
    raise LLMUnavailable("rate limited")


# ---------------------------------------------------------------- tasks
V1_MAP = {"grammar": "practice", "vocabulary": "practice",
          "pronunciation": "question", "knowledge_query": "question"}


async def router_item(variant, model, ex, stats):
    async with SEM:
        try:
            if variant == "v1":
                reply = (await raw_call(model, V1_ROUTER, ex["text"], stats)).strip().lower()
                label = reply if reply in V1_MAP else "grammar"  # prototype's fallback
                return {**ex, "pred": V1_MAP[label], "raw": reply[:80]}
            system = V2_ROUTER if variant == "v2" else nodes.ROUTER_SYSTEM.format(lang=ex.get("lang", "English"))
            d = await v2_call(model, "router", system, ex["text"], RouterDecision, stats)
            return {**ex, "pred": d.intent, "raw": d.reason}
        except Exception as e:
            return {**ex, "pred": None, "error": str(e)[:200]}


async def grammar_item(variant, model, ex, stats):
    async with SEM:
        try:
            if variant == "v1":
                reply = await raw_call(model, V1_GRAMMAR.format(lang=ex["lang"]), ex["text"], stats)
                # exactly how the prototype decided whether to log an error
                return {**ex, "pred": "grammatically correct" not in reply.lower(), "raw": reply[:300]}
            g = await v2_call(model, "grammar", nodes.GRAMMAR_SYSTEM.format(lang=ex["lang"]),
                              ex["text"], GrammarFeedback, stats)
            return {**ex, "pred": bool(g.errors), "pred_correction": g.corrected_sentence,
                    "pred_types": sorted({e.error_type for e in g.errors})}
        except Exception as e:
            return {**ex, "pred": None, "error": str(e)[:200]}


# ---------------------------------------------------------------- LLM-as-judge for corrections
JUDGE_SYSTEM = (
    "You grade grammar corrections. Given a learner sentence, a reference correction and a "
    "candidate correction, decide whether the candidate is fully grammatical AND fixes every "
    "error the reference fixes without changing the meaning. Different but equally valid wording "
    "counts as correct. Return JSON.")


class Verdict(BaseModel):
    acceptable: bool
    reason: str


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower().replace("’", "'")
    return re.sub(r"[^\w' ]", "", re.sub(r"\s+", " ", s)).strip()


async def judge(model, src, ref, cand, stats):
    async with SEM:
        try:
            v = await v2_call(model, "judge", JUDGE_SYSTEM,
                              f"Learner: {src}\nReference: {ref}\nCandidate: {cand}", Verdict, stats)
            return v.acceptable
        except Exception:
            return None  # counted as judge failure, excluded from rates


# ---------------------------------------------------------------- metrics
def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None


def prf(rows, pos=True):
    tp = sum(r["pred"] == pos and r["gold"] == pos for r in rows)
    fp = sum(r["pred"] == pos and r["gold"] != pos for r in rows)
    fn = sum(r["pred"] != pos and r["gold"] == pos for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0


def perf(stats):
    lat = [s["latency_ms"] for s in stats]
    return {"p50_ms": pct(lat, .5), "p95_ms": pct(lat, .95),
            "cost_per_1k_calls_usd": round(1000 * statistics.mean(s["cost"] for s in stats), 4) if stats else None}


async def run(variant, model, judge_model, only=None):
    router = [json.loads(l) for l in open(HERE / "data/router_eval.jsonl")]
    grammar = [json.loads(l) for l in open(HERE / "data/grammar_eval.jsonl")]
    rs, gs, js = [], [], []

    if only == "router":
        grammar = []
    r_out = await asyncio.gather(*(router_item(variant, model, ex, rs) for ex in router))
    g_out = await asyncio.gather(*(grammar_item(variant, model, ex, gs) for ex in grammar))
    if only == "router":
        r_ok = [dict(r, gold=r["intent"]) for r in r_out if r["pred"] is not None]
        return {"variant": variant, "model": model, "ts": datetime.now().isoformat(timespec="seconds"),
                "router": {"n": len(router), "failed": len(router) - len(r_ok),
                           "accuracy": round(sum(r["pred"] == r["gold"] for r in r_ok) / len(r_ok), 3),
                           "question_prf": prf(r_ok, "question"), **perf(rs),
                           "errors": [r for r in r_out if r["pred"] != r["intent"]]}}

    r_ok = [dict(r, gold=r["intent"]) for r in r_out if r["pred"] is not None]
    g_ok = [dict(r, gold=r["has_error"]) for r in g_out if r["pred"] is not None]
    correct_sents = [r for r in g_ok if not r["gold"]]
    res = {
        "variant": variant, "model": model, "ts": datetime.now().isoformat(timespec="seconds"),
        "router": {"n": len(router), "failed": len(router) - len(r_ok),
                   "accuracy": round(sum(r["pred"] == r["gold"] for r in r_ok) / len(r_ok), 3),
                   "question_prf": prf(r_ok, "question"), **perf(rs),
                   "errors": [r for r in r_out if r["pred"] != r["intent"]]},
        "grammar": {"n": len(grammar), "failed": len(grammar) - len(g_ok),
                    "detect_prf": prf(g_ok, True),
                    "false_alarm_rate": round(sum(r["pred"] for r in correct_sents) / len(correct_sents), 3),
                    "by_lang": {l: prf([r for r in g_ok if r["lang"] == l], True)
                                for l in ("English", "German")}, **perf(gs),
                    "errors": [r for r in g_out if r["pred"] != r["has_error"]]},
    }

    if variant != "v1":
        # correction quality on sentences that really have errors and were flagged
        cand = [r for r in g_ok if r["gold"] and r["pred"]]
        exact = [norm(r["pred_correction"]) == norm(r["gold_correction"]) for r in cand]
        verdicts = await asyncio.gather(*(judge(judge_model, r["text"], r["gold_correction"],
                                                r["pred_correction"], js) for r in cand))
        # judge calibration: (a) exact matches must be accepted, (b) negative controls —
        # the uncorrected learner sentence offered as the "correction" — must be rejected
        neg = await asyncio.gather(*(judge(judge_model, r["text"], r["gold_correction"], r["text"], js)
                                     for r in cand))
        acc_on_exact = [v for v, e in zip(verdicts, exact) if e and v is not None]
        judged = [v for v in verdicts if v is not None]
        neg_ok = [v for v in neg if v is not None]
        res["correction"] = {
            "n": len(cand), "exact_match": round(sum(exact) / len(cand), 3),
            "judge_acceptable": round(sum(judged) / len(judged), 3) if judged else None,
            "judge_failed": len(verdicts) - len(judged) + len(neg) - len(neg_ok),
            "judge_model": judge_model,
            "judge_calibration": {
                "accepts_exact_matches": round(sum(acc_on_exact) / len(acc_on_exact), 3) if acc_on_exact else None,
                "rejects_uncorrected_input": round(1 - sum(neg_ok) / len(neg_ok), 3) if neg_ok else None},
            "rejected": [{"text": r["text"], "gold": r["gold_correction"], "pred": r["pred_correction"]}
                         for r, v in zip(cand, verdicts) if v is False],
        }
    return res


def append_md(res):
    md = HERE / "RESULTS.md"
    if not md.exists():
        md.write_text("# Eval results\n\nOne row per run. Details in `eval/results/*.json`.\n\n"
                      "| time | variant | model | router acc | question F1 | grammar P / R / F1 | "
                      "false alarm | DE F1 | correction exact / judge | grammar p50/p95 ms | failed |\n"
                      "|---|---|---|---|---|---|---|---|---|---|---|\n")
    r, g, c = res["router"], res.get("grammar"), res.get("correction")
    if g is None:  # router-only run
        g = {"detect_prf": ("—",) * 3, "false_alarm_rate": "—", "by_lang": {"German": ("—",) * 3},
             "p50_ms": "—", "p95_ms": "—", "failed": 0}
    row = (f"| {res['ts']} | {res['variant']} | {res['model']} | {r['accuracy']} | "
           f"{r['question_prf'][2]} | {' / '.join(map(str, g['detect_prf']))} | {g['false_alarm_rate']} | "
           f"{g['by_lang']['German'][2]} | "
           f"{f'{c['exact_match']} / {c['judge_acceptable']}' if c else '—'} | "
           f"{g['p50_ms']} / {g['p95_ms']} | {r['failed'] + g['failed']} |\n")
    with open(md, "a") as f:
        f.write(row)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["v3"])
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b")
    ap.add_argument("--only", choices=["router"], help="run a single task to save quota")
    a = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    for v in a.variants:
        res = await run(v, a.model, a.judge_model, a.only)
        name = f"{res['ts'].replace(':', '')}_{v}_{a.model.replace('/', '_')}.json"
        (HERE / "results" / name).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        append_md(res)
        summary = {k: {kk: vv for kk, vv in res[k].items() if kk not in ("errors", "rejected")}
                   for k in ("router", "grammar", "correction") if k in res}
        print(v, json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
