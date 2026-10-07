"""Closed loop 1 (grammar): show an error only when we are confident it is real.

    python -m eval.conf_eval collect --model openai/gpt-oss-20b --k 5   # ~550 calls, needs GROQ_API_KEY
    python -m eval.conf_eval analyze --raw eval/results/conf_raw_<ts>.json            # dev only
    python -m eval.conf_eval analyze --raw eval/results/conf_raw_<ts>.json --final    # test split, ONCE

collect stores everything (baseline verdict, verbalized verdict+confidence, k sampled verdicts) so that
thresholds and curves are computed offline, with no further API calls. Protocol: eval/PROTOCOL.md.
"""
import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from agent import nodes
from agent.confidence import CONF_SUFFIX, consistency_score
from agent.llm import LLMUnavailable, structured_call
from agent.schemas import GrammarFeedback, GrammarFeedbackConf
from eval import gating
from eval.stats import auroc, bootstrap_ci, ece, mcnemar_exact

HERE = Path(__file__).parent
RESULTS_MD = HERE / "RESULTS.md"
SEM = asyncio.Semaphore(2)  # free tier: 8k tokens/min per model
RL_WAIT_S = 15
SYSTEMS = ("base", "verbalized", "consistency")


def load_items():
    split = json.load(open(HERE / "data/grammar_split.json"))["split"]
    return [dict(json.loads(l), split=split[json.loads(l)["id"]])
            for l in open(HERE / "data/grammar_eval.jsonl")]


async def _call(model, system, text, schema, temperature, retries=6):
    for attempt in range(retries + 1):
        try:
            return await structured_call("conf_eval", system, text, schema, models=[model],
                                         temperature=temperature)
        except LLMUnavailable as e:
            if "RateLimit" in str(e) and attempt < retries:
                await asyncio.sleep(RL_WAIT_S)
                continue
            raise


async def _safe(coro):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001 - a failed call is data, not a crash
        return e


async def collect_item(ex, model, k, temperature):
    system = nodes.GRAMMAR_SYSTEM.format(lang=ex["lang"])
    async with SEM:
        base = await _safe(_call(model, system, ex["text"], GrammarFeedback, 0.0))
        verb = await _safe(_call(model, system + CONF_SUFFIX, ex["text"], GrammarFeedbackConf, 0.0))
        samples = await asyncio.gather(*(_safe(_call(model, system, ex["text"], GrammarFeedback, temperature))
                                         for _ in range(k)))
    return ex["id"], {
        "split": ex["split"], "lang": ex["lang"], "text": ex["text"], "has_error": ex["has_error"],
        "base": ({"has_error": bool(base.errors)} if isinstance(base, GrammarFeedback) else {"error": str(base)[:200]}),
        "verb": ({"has_error": bool(verb.errors), "confidence": verb.confidence}
                 if isinstance(verb, GrammarFeedbackConf) else {"error": str(verb)[:200]}),
        "samples": [bool(s.errors) if isinstance(s, GrammarFeedback) else None for s in samples]}


async def collect(items, model, k=5, temperature=0.7):
    out = await asyncio.gather(*(collect_item(ex, model, k, temperature) for ex in items))
    return {"meta": {"model": model, "k": k, "temperature": temperature,
                     "ts": datetime.now().isoformat(timespec="seconds")}, "items": dict(out)}


def build_systems(raw):
    """-> ({system: [item dict(id, split, gold, flag, score)]}, n_dropped_base)"""
    systems = {s: [] for s in SYSTEMS}
    dropped = 0
    for id_, it in raw["items"].items():
        b, v = it.get("base") or {}, it.get("verb") or {}
        if "has_error" not in b:
            dropped += 1
            continue
        c = {"id": id_, "split": it["split"], "gold": it["has_error"]}
        systems["base"].append({**c, "flag": b["has_error"], "score": None})
        if "has_error" in v:
            systems["verbalized"].append({**c, "flag": v["has_error"], "score": v["confidence"]})
        sc = consistency_score(it.get("samples") or [])
        if sc is not None:
            systems["consistency"].append({**c, "flag": b["has_error"], "score": sc})
    return systems, dropped


def _fmt(row):
    return (f"P {row['precision']:.2f} / R {row['recall']:.2f} / false alarm {row['false_alarm']:.2f} "
            f"(shown TP {row['tp']}, FP {row['fp']}; missed {row['fn']}; correct sentences kept {row['tn']})")


def signal_quality(items, bins):
    """Among FLAGGED items: does the score separate real errors (gold) from false alarms?"""
    fl = [i for i in items if i["flag"] and i["score"] is not None]
    return {"n_flagged": len(fl), "n_false_alarms": sum(not i["gold"] for i in fl),
            "auroc": auroc([i["score"] for i in fl], [i["gold"] for i in fl]),
            "ece": ece([i["score"] for i in fl], [i["gold"] for i in fl], bins)}


def split_items(items, which):
    return [i for i in items if i["split"] == which]


def analyze(raw, final=False, beta=0.5, bins=3, n_boot=2000):
    systems, dropped = build_systems(raw)
    lines, final_rows = [], []
    lines.append(f"model={raw['meta']['model']} k={raw['meta']['k']} T={raw['meta']['temperature']}  "
                 f"dropped (baseline call failed): {dropped}")
    dev_base = split_items(systems["base"], "dev")
    lines.append(f"\nDEV (n={len(dev_base)}), baseline ungated: {_fmt(gating.evaluate(dev_base, None))}")
    chosen = {}
    for name in ("verbalized", "consistency"):
        dev = split_items(systems[name], "dev")
        if not dev:
            lines.append(f"\n[{name}] no dev data")
            continue
        row = gating.choose_tau(dev, beta=beta)
        chosen[name] = row["tau"]
        q = signal_quality(dev, bins)
        lines.append(f"\n[{name}] dev ungated: {_fmt(gating.evaluate(dev, 0.0))}")
        lines.append(f"[{name}] dev chosen tau={row['tau']:.2f} (max F{beta} on dev): {_fmt(row)}; "
                     f"flags suppressed {row['suppressed_flag_rate']:.0%}")
        lines.append(f"[{name}] dev signal quality: AUROC={q['auroc']:.2f} ECE({bins} bins)={q['ece']:.2f} "
                     f"over {q['n_flagged']} flagged items ({q['n_false_alarms']} are false alarms) -- rough, tiny n")
        lines.append(f"[{name}] dev curve (tau, precision, recall, false_alarm, flags suppressed):")
        for r in gating.curve(dev):
            lines.append(f"    tau={r['tau']:.2f}  P={r['precision']:.2f}  R={r['recall']:.2f}  "
                         f"FA={r['false_alarm']:.2f}  suppressed={r['suppressed_flag_rate']:.0%}")
    if not final:
        lines.append("\n(dev only. Run with --final to evaluate the test split with the taus above. Once.)")
        return "\n".join(lines), final_rows
    test_base = split_items(systems["base"], "test")
    base_eval = gating.evaluate(test_base, None)
    lines.append(f"\nTEST (n={len(test_base)}), baseline: {_fmt(base_eval)}")
    by_id_base = {i["id"]: i for i in test_base}
    for name, tau in chosen.items():
        test = split_items(systems[name], "test")
        ungated, gated = gating.evaluate(test, 0.0), gating.evaluate(test, tau)
        q = signal_quality(test, bins)

        def ci(metric, tau_=tau, test_=test):
            return bootstrap_ci(len(test_), lambda idx: gating.evaluate([test_[i] for i in idx], tau_)[metric],
                                n=n_boot)
        fa, rc = ci("false_alarm"), ci("recall")

        def paired(a_items, a_tau, b_items, b_tau):
            a = {i["id"]: i for i in a_items}
            b_ = 0
            c_ = 0
            for i in b_items:
                if i["id"] not in a:
                    continue
                ra = gating.evaluate([a[i["id"]]], a_tau)
                rb = gating.evaluate([i], b_tau)
                ok_a = (ra["tp"] + ra["tn"]) == 1
                ok_b = (rb["tp"] + rb["tn"]) == 1
                b_ += ok_a and not ok_b
                c_ += ok_b and not ok_a
            return b_, c_
        b1, c1 = paired(test_base, None, test, tau)
        b2, c2 = paired(test, 0.0, test, tau)
        lines.append(f"\n[{name}] TEST ungated: {_fmt(ungated)}")
        lines.append(f"[{name}] TEST gated tau={tau:.2f}: {_fmt(gated)}; flags suppressed {gated['suppressed_flag_rate']:.0%}")
        lines.append(f"[{name}] 95% bootstrap CI: false alarm [{fa[0]:.2f}, {fa[1]:.2f}], recall [{rc[0]:.2f}, {rc[1]:.2f}]  "
                     f"(n={len(test)}: intervals are WIDE; do not claim more than a direction)")
        lines.append(f"[{name}] TEST signal quality: AUROC={q['auroc']:.2f} ECE({bins} bins)={q['ece']:.2f} "
                     f"over {q['n_flagged']} flagged items ({q['n_false_alarms']} false alarms)")
        lines.append(f"[{name}] paired vs baseline: only-baseline-correct={b1}, only-gated-correct={c1}, "
                     f"exact McNemar p={mcnemar_exact(b1, c1):.3f}; vs own ungated: {b2}/{c2}, p={mcnemar_exact(b2, c2):.3f}")
        final_rows.append({"signal": name, "tau": tau, "n_test": len(test), "base": base_eval, "gated": gated,
                           "auroc": q["auroc"], "ece": q["ece"], "p_vs_base": mcnemar_exact(b1, c1)})
    return "\n".join(lines), final_rows


def append_results(raw, rows):
    md = RESULTS_MD
    head = "\n## Confidence gating (closed loop 1, held-out test split, run once)\n\n" \
           "| time | signal | model | tau (chosen on dev) | n test | baseline P / R / false alarm | " \
           "gated P / R / false alarm | AUROC (test) | ECE (test) | McNemar p vs baseline |\n|---|---|---|---|---|---|---|---|---|---|\n"
    text = md.read_text() if md.exists() else ""
    with open(md, "a") as f:
        if "## Confidence gating (closed loop 1" not in text:
            f.write(head)
        for r in rows:
            b, g = r["base"], r["gated"]
            f.write(f"| {raw['meta']['ts']} | {r['signal']} | {raw['meta']['model']} | {r['tau']:.2f} | {r['n_test']} | "
                    f"{b['precision']:.2f} / {b['recall']:.2f} / {b['false_alarm']:.2f} | "
                    f"{g['precision']:.2f} / {g['recall']:.2f} / {g['false_alarm']:.2f} | "
                    f"{r['auroc']:.2f} | {r['ece']:.2f} | {r['p_vs_base']:.3f} |\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", default="openai/gpt-oss-20b")
    c.add_argument("--k", type=int, default=5)
    c.add_argument("--temperature", type=float, default=0.7)
    a = sub.add_parser("analyze")
    a.add_argument("--raw", required=True)
    a.add_argument("--final", action="store_true", help="evaluate the TEST split (allowed once)")
    a.add_argument("--force", action="store_true", help="re-run --final (invalidates the held-out claim)")
    a.add_argument("--beta", type=float, default=0.5)
    a.add_argument("--bins", type=int, default=3)
    args = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    if args.cmd == "collect":
        raw = asyncio.run(collect(load_items(), args.model, args.k, args.temperature))
        path = HERE / "results" / f"conf_raw_{raw['meta']['ts'].replace(':', '')}.json"
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=1))
        print("saved", path)
        return
    raw = json.load(open(args.raw))
    lock = HERE / "results" / "FINAL_conf_eval.json"
    if args.final and lock.exists() and not args.force:
        raise SystemExit(f"{lock} exists: the test split was already used. Use --force only if you accept "
                         "that the test numbers are no longer held-out.")
    text, rows = analyze(raw, args.final, args.beta, args.bins)
    print(text)
    if args.final:
        (HERE / "results" / f"conf_analysis_{raw['meta']['ts'].replace(':', '')}.txt").write_text(text)
        append_results(raw, rows)
        lock.write_text(json.dumps({"raw": os.path.basename(args.raw), "ts": datetime.now().isoformat()}))
        print("\nappended to eval/RESULTS.md; test split is now used up.")


if __name__ == "__main__":
    main()
