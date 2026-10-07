"""Closed loop 2 (speech): do we trust the transcript before we grade the learner's grammar on it?

Two questions, both measured on recordings of YOUR OWN voice reading the eval sentences verbatim
(including their mistakes; see eval/audio/READING_SHEET.md):
  1. Fidelity: does ASR keep the learner's mistakes, or does it quietly "correct" them (so the grammar
     node never sees the error)? Also: does Azure pronunciation get scored against a corrected word?
  2. Gating: do Whisper's confidence features (and Azure/Whisper disagreement) predict a wrong
     transcript well enough to ask the learner to repeat instead of giving possibly wrong feedback?

    python -m eval.speech_eval collect --model openai/gpt-oss-20b [--condition clean|noisy] [--azure]
    python -m eval.speech_eval analyze --raw eval/results/speech_raw_<ts>.json            # dev + descriptive
    python -m eval.speech_eval analyze --raw ... --feature avg_prob --final               # test, ONCE

Protocol: eval/PROTOCOL.md.
"""
import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from agent import nodes
from agent.asr import transcribe_file
from agent.asr_conf import FEATURES
from agent.schemas import GrammarFeedback
from agent.textmetrics import norm_text, wer
from eval import gating
from eval.conf_eval import _call
from eval.stats import auroc, bootstrap_ci, mcnemar_exact

HERE = Path(__file__).parent
RESULTS_MD = HERE / "RESULTS.md"
AUDIO = HERE / "audio"
EXTS = (".wav", ".m4a", ".mp3", ".flac", ".ogg", ".webm", ".mp4", ".mpeg", ".mpga")
AZURE_LANG = {"English": "en-US", "German": "de-DE"}


def find_audio(id_, condition):
    suffix = "__noisy" if condition == "noisy" else ""
    for ext in EXTS:
        p = AUDIO / f"{id_}{suffix}{ext}"
        if p.exists():
            return p
    return None


def load_manifest():
    return [json.loads(l) for l in open(HERE / "data/speech_manifest.jsonl")]


async def collect(items, model, condition="clean", azure=False, pace=0.5):
    out, missing = {}, []
    for ex in items:
        path = find_audio(ex["id"], condition)
        if path is None:
            missing.append(ex["id"])
            continue
        rec = dict(ex, audio=str(path.name), condition=condition)
        try:
            rec["asr"] = await asyncio.to_thread(transcribe_file, str(path))
        except Exception as e:  # noqa: BLE001
            rec["asr_error"] = str(e)[:200]
            out[ex["id"]] = rec
            continue
        system = nodes.GRAMMAR_SYSTEM.format(lang=ex["lang"])
        for key, text in (("asr_flag", rec["asr"]["text"]), ("text_flag", ex["text"])):
            try:
                rec[key] = bool((await _call(model, system, text, GrammarFeedback, 0.0)).errors)
            except Exception as e:  # noqa: BLE001
                rec[key] = None
                rec[key + "_error"] = str(e)[:200]
        if azure:
            from agent.pronunciation import assess_pronunciation
            try:
                rec["azure"] = await asyncio.to_thread(assess_pronunciation, str(path), rec["asr"]["text"],
                                                       AZURE_LANG.get(ex["lang"], "en-US"))
            except Exception as e:  # noqa: BLE001
                rec["azure"] = {"error": str(e)[:200]}
        out[ex["id"]] = rec
        await asyncio.sleep(pace)
    return {"meta": {"model": model, "condition": condition, "azure": azure,
                     "ts": datetime.now().isoformat(timespec="seconds"), "missing_audio": missing},
            "items": out}


def prepare(raw):
    """Add derived fields. Only items with an ASR transcript and both grammar flags are usable."""
    rows = []
    for id_, r in raw["items"].items():
        if "asr" not in r or r.get("asr_flag") is None or r.get("text_flag") is None:
            continue
        asr_text = r["asr"]["text"]
        d = dict(r, id=id_, wer=wer(r["text"], asr_text), wrong=wer(r["text"], asr_text) > 0)
        feats = r["asr"].get("features") or {}
        d["scores"] = {f: feats.get(f) for f in FEATURES}
        az = r.get("azure") or {}
        if az.get("recognized_text") is not None:
            d["scores"]["azure_agree"] = 1.0 - min(1.0, wer(asr_text, az["recognized_text"]))
        if r["has_error"] and r.get("gold_correction"):
            d["autocorrected"] = wer(r["gold_correction"], asr_text) < wer(r["text"], asr_text)
        rows.append(d)
    return rows


def fidelity(rows):
    err = [r for r in rows if r["has_error"]]
    return {"n": len(rows), "mean_wer": sum(r["wer"] for r in rows) / len(rows) if rows else float("nan"),
            "exact": sum(not r["wrong"] for r in rows) / len(rows) if rows else float("nan"),
            "n_error_sentences": len(err),
            "autocorrected": sum(r.get("autocorrected", False) for r in err) / len(err) if err else float("nan")}


def downstream(rows):
    text_items = [{"gold": r["has_error"], "flag": r["text_flag"], "score": None} for r in rows]
    asr_items = [{"gold": r["has_error"], "flag": r["asr_flag"], "score": None} for r in rows]
    only_text = sum((t["flag"] == t["gold"]) and (a["flag"] != a["gold"]) for t, a in zip(text_items, asr_items))
    only_asr = sum((a["flag"] == a["gold"]) and (t["flag"] != t["gold"]) for t, a in zip(text_items, asr_items))
    return (gating.evaluate(text_items, None), gating.evaluate(asr_items, None),
            only_text, only_asr, mcnemar_exact(only_text, only_asr))


def signal_aurocs(rows):
    """AUROC of 'low confidence' for predicting a wrong transcript, per feature."""
    out = {}
    names = {k for r in rows for k, v in r["scores"].items() if v is not None}
    for f in sorted(names):
        sub = [r for r in rows if r["scores"].get(f) is not None]
        out[f] = (auroc([1 - r["scores"][f] for r in sub], [r["wrong"] for r in sub]), len(sub))
    return out


def gate_items(rows, feature):
    return [{"gold": r["has_error"], "flag": r["asr_flag"], "score": r["scores"].get(feature),
             "wrong": r["wrong"], "id": r["id"]} for r in rows]


def _fmt(row):
    return (f"P {row['precision']:.2f} / R {row['recall']:.2f} / false alarm {row['false_alarm']:.2f} "
            f"/ asked to repeat {row['repeat_rate']:.0%}")


def analyze(raw, feature=None, final=False, max_repeat=0.2, beta=0.5, n_boot=2000):
    rows = prepare(raw)
    split = {"dev": [r for r in rows if r["split"] == "dev"], "test": [r for r in rows if r["split"] == "test"]}
    lines = [f"condition={raw['meta'].get('condition')} model={raw['meta']['model']} usable items={len(rows)} "
             f"(missing audio: {len(raw['meta'].get('missing_audio', []))})"]
    fid = fidelity(rows)
    lines.append(f"\nFIDELITY (all items, descriptive): mean WER={fid['mean_wer']:.3f}; transcript identical to what was read: "
                 f"{fid['exact']:.0%}; of {fid['n_error_sentences']} sentences with a mistake, ASR moved toward the "
                 f"correction in {fid['autocorrected']:.0%}")
    t, a, ot, oa, p = downstream(rows)
    lines.append(f"\nDOWNSTREAM grammar detection on the written text : {_fmt(t)}")
    lines.append(f"DOWNSTREAM grammar detection on the ASR transcript: {_fmt(a)}")
    lines.append(f"  paired: only-text-correct={ot}, only-ASR-correct={oa}, exact McNemar p={p:.3f}")
    wrong = sum(r["wrong"] for r in rows)
    lines.append(f"\nTRANSCRIPT ERRORS: {wrong} of {len(rows)} transcripts differ from what was read")
    if wrong in (0, len(rows)):
        lines.append("  -> no variation in 'transcript wrong': gating cannot be evaluated. Record a noisy condition "
                     "(scripts/add_noise.py) and re-run collect.")
    sa = signal_aurocs(rows)
    for f, (v, n) in sa.items():
        lines.append(f"  AUROC of low '{f}' for 'transcript wrong' (all items, n={n}): {v:.2f}")
    feature = feature or max(sa, key=lambda k: (sa[k][0] if sa[k][0] == sa[k][0] else -1), default=None)
    if feature is None or wrong in (0, len(rows)):
        return "\n".join(lines), None
    dev = gate_items(split["dev"], feature)
    row = gating.choose_tau(dev, beta=beta, max_rate=max_repeat, rate_key="repeat_rate")
    lines.append(f"\nGATE feature={feature}: dev ungated {_fmt(gating.evaluate(dev, 0.0))}")
    lines.append(f"GATE dev chosen tau={row['tau']:.3f} (max F{beta}, repeat rate <= {max_repeat:.0%}): {_fmt(row)}")
    for r in gating.curve(dev):
        lines.append(f"    tau={r['tau']:.3f}  P={r['precision']:.2f}  R={r['recall']:.2f}  FA={r['false_alarm']:.2f}  repeat={r['repeat_rate']:.0%}")
    if not final:
        lines.append("\n(dev only. Run with --final to evaluate the test split with this tau. Once.)")
        return "\n".join(lines), None
    test = gate_items(split["test"], feature)
    ung, gat = gating.evaluate(test, 0.0), gating.evaluate(test, row["tau"])
    asked = [i for i in test if i["score"] is not None and i["score"] < row["tau"]]
    fa = bootstrap_ci(len(test), lambda idx: gating.evaluate([test[i] for i in idx], row["tau"])["false_alarm"], n=n_boot)
    rc = bootstrap_ci(len(test), lambda idx: gating.evaluate([test[i] for i in idx], row["tau"])["recall"], n=n_boot)
    lines.append(f"\nTEST (n={len(test)}) ungated: {_fmt(ung)}")
    lines.append(f"TEST gated tau={row['tau']:.3f}: {_fmt(gat)}")
    lines.append(f"  95% bootstrap CI: false alarm [{fa[0]:.2f}, {fa[1]:.2f}], recall [{rc[0]:.2f}, {rc[1]:.2f}] (WIDE, tiny n)")
    lines.append(f"  asked to repeat {len(asked)} items; transcript was really wrong in {sum(i['wrong'] for i in asked)} of them")
    return "\n".join(lines), {"feature": feature, "tau": row["tau"], "n_test": len(test), "ungated": ung, "gated": gat,
                              "asked": len(asked), "asked_wrong": sum(i["wrong"] for i in asked)}


def append_results(raw, res):
    md = RESULTS_MD
    text = md.read_text() if md.exists() else ""
    with open(md, "a") as f:
        if "## ASR confidence gating (closed loop 2" not in text:
            f.write("\n## ASR confidence gating (closed loop 2, held-out test split, run once)\n\n"
                    "| time | condition | feature | tau (dev) | n test | ungated P / R / false alarm | gated P / R / false alarm | asked to repeat (really wrong) |\n"
                    "|---|---|---|---|---|---|---|---|\n")
        u, g = res["ungated"], res["gated"]
        f.write(f"| {raw['meta']['ts']} | {raw['meta'].get('condition')} | {res['feature']} | {res['tau']:.3f} | {res['n_test']} | "
                f"{u['precision']:.2f} / {u['recall']:.2f} / {u['false_alarm']:.2f} | "
                f"{g['precision']:.2f} / {g['recall']:.2f} / {g['false_alarm']:.2f} | {res['asked']} ({res['asked_wrong']}) |\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", default="openai/gpt-oss-20b")
    c.add_argument("--condition", choices=["clean", "noisy"], default="clean")
    c.add_argument("--azure", action="store_true", help="also run Azure (needs AZURE_SPEECH_KEY/REGION)")
    a = sub.add_parser("analyze")
    a.add_argument("--raw", required=True)
    a.add_argument("--feature", choices=list(FEATURES) + ["azure_agree"])
    a.add_argument("--final", action="store_true")
    a.add_argument("--force", action="store_true")
    a.add_argument("--max-repeat", type=float, default=0.2)
    args = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    if args.cmd == "collect":
        raw = asyncio.run(collect(load_manifest(), args.model, args.condition, args.azure))
        path = HERE / "results" / f"speech_raw_{args.condition}_{raw['meta']['ts'].replace(':', '')}.json"
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=1))
        print("saved", path, "| missing audio:", len(raw["meta"]["missing_audio"]))
        return
    raw = json.load(open(args.raw))
    lock = HERE / "results" / f"FINAL_speech_eval_{raw['meta'].get('condition')}.json"
    if args.final and lock.exists() and not args.force:
        raise SystemExit(f"{lock} exists: the test split was already used for this condition.")
    text, res = analyze(raw, args.feature, args.final, args.max_repeat)
    print(text)
    if args.final and res:
        append_results(raw, res)
        lock.write_text(json.dumps({"raw": os.path.basename(args.raw), "ts": datetime.now().isoformat()}))
        print("\nappended to eval/RESULTS.md; test split is now used up for this condition.")


if __name__ == "__main__":
    main()
