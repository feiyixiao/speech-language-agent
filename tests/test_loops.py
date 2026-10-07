"""Closed loops 1 (grammar confidence gate) and 2 (ASR confidence gate): offline tests."""
import array
import json
import math
import wave
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent import llm, nodes
from agent.asr_conf import asr_features, gate
from agent.config import settings
from agent.confidence import consistency_score
from agent.textmetrics import wer
from eval import conf_eval, gating, speech_eval
from eval.stats import auroc, bootstrap_ci, ece, f_beta, mcnemar_exact


# ---------------------------------------------------------------- fake LLM
class Fake:
    """handler(messages, temperature) -> dict | str"""
    def __init__(self, handler):
        self.handler, self.calls = handler, []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, model, messages, temperature=0.0, **kw):
        self.calls.append((model, temperature))
        p = self.handler(messages, temperature)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=p if isinstance(p, str) else json.dumps(p)))], usage=None)


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "trace_dir", tmp_path)
    monkeypatch.setattr(settings, "llm_models", ["m1"])
    box = {}

    def install(handler):
        box["c"] = Fake(handler)
        monkeypatch.setattr(llm, "get_client", lambda: box["c"])
        return box["c"]
    return install


ERR = {"original": "goes", "correction": "go", "error_type": "subject_verb_agreement", "explanation": "x"}


def flagged(conf=None):
    d = {"has_error": True, "errors": [ERR], "corrected_sentence": "I go."}
    if conf is not None:
        d["confidence"] = conf
    return d


# ---------------------------------------------------------------- pure functions
def test_wer():
    assert wer("she goes to school", "she goes to school") == 0
    assert wer("she go to school", "She goes to school.") == 0.25   # 1 substitution / 4 words
    assert wer("", "") == 0.0 and wer("", "x") == 1.0
    assert wer("a b c d", "a c d") == 0.25                         # 1 deletion


def test_stats_worked_examples():
    assert math.isclose(mcnemar_exact(7, 0), 2 * 0.5 ** 7)        # the 0.016 on the CV
    conf = [0.95] * 40 + [0.75] * 40 + [0.55] * 20
    correct = [1] * 32 + [0] * 8 + [1] * 28 + [0] * 12 + [1] * 10 + [0] * 10
    assert math.isclose(ece(conf, correct, n_bins=10), 0.09, abs_tol=1e-9)
    assert auroc([.9, .8, .2, .1], [1, 1, 0, 0]) == 1.0 and auroc([.5, .5], [1, 0]) == 0.5
    assert math.isnan(auroc([.5], [1]))
    assert math.isclose(f_beta(1.0, 0.5, beta=0.5), 1.25 * 0.5 / (0.25 + 0.5))
    lo, hi = bootstrap_ci(20, lambda idx: sum(i < 15 for i in idx) / len(idx), n=300)
    assert lo <= 0.75 <= hi


def test_consistency_score():
    assert consistency_score([True, True, False, None]) == pytest.approx(2 / 3)
    assert consistency_score([None, None]) is None


def test_gating_removes_low_confidence_false_alarms():
    items = ([{"gold": True, "flag": True, "score": s, "id": i} for i, s in enumerate([.9, .9, .8, .8, .7])]
             + [{"gold": False, "flag": True, "score": s} for s in (.3, .4)]
             + [{"gold": False, "flag": False, "score": None} for _ in range(3)]
             + [{"gold": True, "flag": False, "score": None}])
    base = gating.evaluate(items, 0.0)
    assert (base["fp"], base["tp"]) == (2, 5) and base["false_alarm"] == pytest.approx(2 / 5)
    row = gating.choose_tau(items, beta=0.5)
    assert row["fp"] == 0 and row["tp"] == 5 and 0.4 < row["tau"] <= 0.7
    assert row["suppressed_flag_rate"] == pytest.approx(2 / 7)
    assert gating.evaluate(items, None)["fp"] == 2          # tau None = no gate


def test_choose_tau_respects_max_rate_and_ties_go_low():
    items = [{"gold": True, "flag": True, "score": 0.9} for _ in range(4)] + \
            [{"gold": False, "flag": False, "score": 0.1} for _ in range(4)]
    row = gating.choose_tau(items, max_rate=0.2, rate_key="repeat_rate")
    assert row["repeat_rate"] <= 0.2 and row["tau"] == 0.0   # nothing to gain -> lowest tau


# ---------------------------------------------------------------- ASR features
SEGS = [{"start": 0, "end": 2, "avg_logprob": -0.1, "no_speech_prob": 0.01, "compression_ratio": 1.2},
        {"start": 2, "end": 3, "avg_logprob": -1.5, "no_speech_prob": 0.3, "compression_ratio": 2.0}]


def test_asr_features_and_gate():
    f = asr_features(SEGS)
    assert f["min_prob"] == pytest.approx(math.exp(-1.5)) and f["no_speech"] == pytest.approx(0.7)
    assert f["compression"] == pytest.approx(0.5)
    assert f["avg_prob"] == pytest.approx((2 * math.exp(-0.1) + 1 * math.exp(-1.5)) / 3)
    obj = [SimpleNamespace(**s) for s in SEGS]
    assert asr_features(obj) == f                           # SDK objects work like dicts
    assert asr_features(None) == {} and asr_features([]) == {}
    assert gate(f, "min_prob", 0.5) == (f["min_prob"], True)
    assert gate(f, "min_prob", None)[1] is False            # gate off
    assert gate({}, "min_prob", 0.5) == (None, False)       # no features: never block
    assert gate(f, "nonsense", 0.5) == (None, False)


# ---------------------------------------------------------------- grammar node: loop 1
async def test_gate_off_changes_nothing(fake_llm):
    fake_llm(lambda m, t: flagged())
    out = await nodes.grammar_node({"transcript": "I goes", "error_history": []})
    g = out["grammar"]
    assert g["has_error"] and g["uncertain"] is False and g["confidence"] is None
    assert out["error_history"] == ["subject_verb_agreement"]


async def test_verbalized_low_confidence_is_suppressed_but_kept_for_audit(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "grammar_conf_mode", "verbalized")
    monkeypatch.setattr(settings, "grammar_conf_threshold", 0.7)
    fake_llm(lambda m, t: flagged(0.4))
    out = await nodes.grammar_node({"transcript": "I goes", "error_history": []})
    g = out["grammar"]
    assert g["uncertain"] and not g["has_error"] and g["errors"] == []
    assert g["suppressed_errors"][0]["correction"] == "go" and g["confidence"] == 0.4
    assert g["corrected_sentence"] == "I goes"              # no correction shown
    assert out["error_history"] == []                       # an uncertain flag is not logged as a repeat error


async def test_verbalized_high_confidence_is_shown(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "grammar_conf_mode", "verbalized")
    monkeypatch.setattr(settings, "grammar_conf_threshold", 0.7)
    fake_llm(lambda m, t: flagged(0.9))
    g = (await nodes.grammar_node({"transcript": "I goes", "error_history": []}))["grammar"]
    assert g["has_error"] and not g["uncertain"] and g["confidence"] == 0.9


async def test_verbalized_prompt_asks_for_confidence_only_in_that_mode(fake_llm, monkeypatch):
    seen = []
    fake_llm(lambda m, t: seen.append(m[0]["content"]) or flagged(0.9))
    await nodes.grammar_node({"transcript": "I goes", "error_history": []})
    monkeypatch.setattr(settings, "grammar_conf_mode", "verbalized")
    await nodes.grammar_node({"transcript": "I goes", "error_history": []})
    assert "confidence" not in seen[0] and "confidence" in seen[1]


async def test_consistency_uses_k_samples_at_temperature(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "grammar_conf_mode", "consistency")
    monkeypatch.setattr(settings, "grammar_conf_threshold", 0.6)
    monkeypatch.setattr(settings, "grammar_conf_k", 5)
    n = {"i": 0}

    def handler(m, t):
        if t == 0.0:
            return flagged()                                 # the verdict shown to the user
        n["i"] += 1
        return flagged() if n["i"] <= 2 else {"has_error": False, "errors": [], "corrected_sentence": "x"}
    c = fake_llm(handler)
    g = (await nodes.grammar_node({"transcript": "I goes", "error_history": []}))["grammar"]
    assert len(c.calls) == 6 and sum(t == settings.grammar_conf_temperature for _, t in c.calls) == 5
    assert g["confidence"] == pytest.approx(0.4) and g["uncertain"]


async def test_consistency_unavailable_degrades_without_blocking(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "grammar_conf_mode", "consistency")
    monkeypatch.setattr(settings, "grammar_conf_threshold", 0.6)
    monkeypatch.setattr(settings, "grammar_conf_k", 2)
    fake_llm(lambda m, t: flagged() if t == 0.0 else "not json")
    out = await nodes.grammar_node({"transcript": "I goes", "error_history": []})
    assert out["grammar"]["has_error"] and out["degraded"] == ["grammar_confidence"]


async def test_uncertain_transcript_suppresses_grammar_errors(fake_llm):
    fake_llm(lambda m, t: flagged())
    g = (await nodes.grammar_node({"transcript": "I goes", "error_history": [], "transcript_uncertain": True}))["grammar"]
    assert g["uncertain"] and g["errors"] == [] and len(g["suppressed_errors"]) == 1


# ---------------------------------------------------------------- API + pronunciation: loop 2
def test_audio_endpoint_gates_on_asr_confidence(fake_llm, monkeypatch):
    from api.main import app
    monkeypatch.setattr("agent.rag.retrieve", lambda q, k=3: [])
    fake_llm(lambda m, t: ({"intent": "practice", "reason": "x"} if m[0]["content"].startswith("You route")
                           else {"suggestions": []} if "word choices" in m[0]["content"] else flagged()))
    monkeypatch.setattr("api.main.transcribe_file", lambda p: {"text": "I goes", "language": "en",
                                                                "provider": "t", "features": asr_features(SEGS)})
    monkeypatch.setattr(settings, "asr_conf_feature", "min_prob")
    client = TestClient(app)
    post = lambda: client.post("/v1/feedback/audio", files={"file": ("a.wav", b"x", "audio/wav")},  # noqa: E731
                               data={"target_language": "English"}).json()
    monkeypatch.setattr(settings, "asr_conf_threshold", None)            # gate off
    assert post()["grammar"]["has_error"] and post()["transcript_uncertain"] is False
    monkeypatch.setattr(settings, "asr_conf_threshold", 0.5)             # min_prob = 0.22 < 0.5
    body = post()
    assert body["transcript_uncertain"] and body["transcript_confidence"] == pytest.approx(math.exp(-1.5))
    assert body["grammar"]["uncertain"] and body["grammar"]["errors"] == []


async def test_pronunciation_reports_whisper_azure_disagreement(monkeypatch):
    monkeypatch.setattr(settings, "azure_speech_key", "k")
    monkeypatch.setattr(settings, "azure_speech_region", "r")
    monkeypatch.setattr("agent.pronunciation.assess_pronunciation",
                        lambda p, t, l: {"accuracy_score": 90, "words": [], "recognized_text": "she go to school"})
    out = await nodes.pronunciation_node({"audio_path": "x.wav", "transcript": "She goes to school.",
                                          "target_language": "English"})
    assert out["pronunciation"]["reference_mismatch_wer"] == 0.25


# ---------------------------------------------------------------- data integrity
def test_split_is_a_stratified_partition():
    items = conf_eval.load_items()
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)) == 79
    dev, test = [i for i in items if i["split"] == "dev"], [i for i in items if i["split"] == "test"]
    assert (len(dev), len(test)) == (50, 29)
    for lang in ("English", "German"):
        for has_err in (True, False):
            assert any(i["lang"] == lang and i["has_error"] == has_err for i in test)
    manifest = speech_eval.load_manifest()
    assert [m["id"] for m in manifest] == ids and {m["id"]: m["split"] for m in manifest} == {i["id"]: i["split"] for i in items}


# ---------------------------------------------------------------- eval scripts end to end (synthetic)
def synthetic_conf_raw():
    items = {}
    for n, it in enumerate(conf_eval.load_items()):
        real = it["has_error"]
        # a plausible informative signal: real errors score high, ~1 in 3 correct sentences get a false alarm with low scores
        false_alarm = (not real) and n % 3 == 0
        flag = real or false_alarm
        conf = (0.9 if real else 0.35) if flag else 0.8
        k = 5
        n_flag = (5 if real else 1) if flag else 0
        items[it["id"]] = {"split": it["split"], "lang": it["lang"], "text": it["text"], "has_error": real,
                           "base": {"has_error": flag}, "verb": {"has_error": flag, "confidence": conf},
                           "samples": [True] * n_flag + [False] * (k - n_flag)}
    return {"meta": {"model": "synthetic", "k": 5, "temperature": 0.7, "ts": "2026-01-01T00:00:00"}, "items": items}


def test_conf_analysis_dev_then_final(tmp_path, monkeypatch):
    monkeypatch.setattr(conf_eval, "RESULTS_MD", tmp_path / "RESULTS.md")
    raw = synthetic_conf_raw()
    text, rows = conf_eval.analyze(raw, final=False)
    assert "DEV" in text and "TEST" not in text.split("(dev only")[0] and rows == []
    text, rows = conf_eval.analyze(raw, final=True, n_boot=50)
    assert {r["signal"] for r in rows} == {"verbalized", "consistency"} and "McNemar" in text
    for r in rows:
        assert r["gated"]["false_alarm"] <= r["base"]["false_alarm"] and r["gated"]["recall"] == r["base"]["recall"]
    conf_eval.append_results(raw, rows)
    md = (tmp_path / "RESULTS.md").read_text()
    assert md.count("## Confidence gating") == 1 and "| synthetic |" in md
    conf_eval.append_results(raw, rows)
    assert (tmp_path / "RESULTS.md").read_text().count("## Confidence gating") == 1


async def test_conf_collect_runs_all_signals(fake_llm):
    def handler(m, t):
        if "confidence" in m[0]["content"]:
            return flagged(0.8)
        return flagged()
    c = fake_llm(handler)
    items = [dict(i) for i in conf_eval.load_items()[:2]]
    raw = await conf_eval.collect(items, "m1", k=3, temperature=0.7)
    it = raw["items"][items[0]["id"]]
    assert it["base"] == {"has_error": True} and it["verb"] == {"has_error": True, "confidence": 0.8}
    assert it["samples"] == [True, True, True] and len(c.calls) == 2 * 5


def synthetic_speech_raw(all_right=False):
    items = {}
    for n, m in enumerate(speech_eval.load_manifest()):
        bad = (not all_right) and n % 6 == 0                   # every 6th transcript is wrong (~17%)
        asr_text = (m["gold_correction"] if (not all_right and m["has_error"] and n % 8 == 0 and m["gold_correction"]) else
                    "totally different words" if bad else m["text"])
        p = 0.5 if bad else 0.9
        items[m["id"]] = dict(m, asr={"text": asr_text, "language": "en", "provider": "t",
                                      "features": {"avg_prob": p, "min_prob": p, "no_speech": 0.9, "compression": 0.7}},
                              asr_flag=m["has_error"] if not bad else (not m["has_error"]),
                              text_flag=m["has_error"], condition="clean")
    return {"meta": {"model": "synthetic", "condition": "clean", "azure": False, "ts": "2026-01-01T00:00:00",
                     "missing_audio": []}, "items": items}


def test_speech_analysis_detects_autocorrection_and_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(speech_eval, "RESULTS_MD", tmp_path / "RESULTS.md")
    raw = synthetic_speech_raw()
    text, res = speech_eval.analyze(raw, feature=None, final=False)
    assert "FIDELITY" in text and "moved toward the correction" in text and res is None
    text, res = speech_eval.analyze(raw, feature="avg_prob", final=True, n_boot=50)
    assert res["feature"] == "avg_prob" and 0.5 < res["tau"] <= 0.9
    assert res["gated"]["false_alarm"] <= res["ungated"]["false_alarm"]  # (the 20% repeat cap applies to DEV only)
    assert res["asked_wrong"] == res["asked"]                  # in this synthetic set every uncertain one is wrong
    speech_eval.append_results(raw, res)
    assert "closed loop 2" in (tmp_path / "RESULTS.md").read_text()


def test_speech_analysis_explains_when_gate_cannot_be_evaluated():
    text, res = speech_eval.analyze(synthetic_speech_raw(all_right=True), final=True)
    assert "cannot be evaluated" in text and res is None


async def test_speech_collect_skips_missing_audio_and_uses_files(fake_llm, monkeypatch, tmp_path):
    fake_llm(lambda m, t: flagged())
    monkeypatch.setattr(speech_eval, "AUDIO", tmp_path)
    (tmp_path / "g000.wav").write_bytes(b"x")
    monkeypatch.setattr(speech_eval, "transcribe_file",
                        lambda p: {"text": "She goes to the gym every morning.", "language": "en", "provider": "t",
                                   "features": {"avg_prob": 0.9}})
    items = speech_eval.load_manifest()[:2]
    raw = await speech_eval.collect(items, "m1", pace=0)
    assert list(raw["items"]) == ["g000"] and raw["meta"]["missing_audio"] == ["g001"]
    assert raw["items"]["g000"]["asr_flag"] is True and raw["items"]["g000"]["text_flag"] is True


def test_add_noise_reaches_requested_snr(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("add_noise", "scripts/add_noise.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src, dst = tmp_path / "a.wav", tmp_path / "a__noisy.wav"
    sig = array.array("h", (int(8000 * math.sin(i / 10)) for i in range(16000)))
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(sig.tobytes())
    mod.add_noise(src, dst, snr_db=10)
    with wave.open(str(dst), "rb") as w:
        y = array.array("h")
        y.frombytes(w.readframes(w.getnframes()))
    noise = [b - a for a, b in zip(sig, y)]
    rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v))  # noqa: E731
    assert 9 < 20 * math.log10(rms(sig) / rms(noise)) < 11
