"""Unit tests — no network. The Groq client is replaced by a fake that can be
told to fail per model, so fallback / degradation paths are exercised."""
import json
from types import SimpleNamespace

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from agent import llm, nodes
from agent.config import settings
from agent.schemas import GrammarFeedback


def _status_error(cls, code):
    req = httpx.Request("POST", "https://api.groq.com")
    return cls("boom", response=httpx.Response(code, request=req), body=None)


class FakeCompletions:
    def __init__(self, behaviour):
        self.behaviour = behaviour  # model -> exception | callable(messages)->dict | dict
        self.calls = []

    async def create(self, model, messages, **kw):
        self.calls.append(model)
        b = self.behaviour.get(model)
        if isinstance(b, Exception):
            raise b
        payload = b(messages) if callable(b) else b
        content = payload if isinstance(payload, str) else json.dumps(payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "trace_dir", tmp_path)
    comp = FakeCompletions({})
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm.asyncio, "sleep", _nosleep)
    return comp


async def _nosleep(_):
    return None


def by_node(messages):
    """Answer according to which node's system prompt is being used."""
    sys = messages[0]["content"]
    user = messages[1]["content"]
    if sys.startswith("You route"):
        return {"intent": "question" if user.endswith("?") else "practice", "reason": "x"}
    if "grammar errors only" in sys:
        if "goes" in user:
            return {"has_error": True, "corrected_sentence": "I go.", "errors": [
                {"original": "goes", "correction": "go", "error_type": "subject_verb_agreement",
                 "explanation": "x"}]}
        return {"has_error": False, "errors": [], "corrected_sentence": user}
    if "word choices" in sys:
        return {"suggestions": []}
    return {"answer": "Use it for indirect objects.", "grounded": True}


M1, M2 = "m1", "m2"


@pytest.fixture(autouse=True)
def two_models(monkeypatch):
    monkeypatch.setattr(settings, "llm_models", [M1, M2])


# ---------------------------------------------------------------- llm fallback policy
async def test_retired_model_falls_back_without_retrying(fake):
    """INCIDENTS.md #1 regression: a 404 must go straight to the next model."""
    fake.behaviour = {M1: _status_error(groq.NotFoundError, 404), M2: by_node}
    out = await llm.structured_call("grammar", nodes.GRAMMAR_SYSTEM, "I goes", GrammarFeedback)
    assert out.has_error and fake.calls == [M1, M2]


async def test_rate_limit_is_retried_then_falls_back(fake):
    fake.behaviour = {M1: _status_error(groq.RateLimitError, 429), M2: by_node}
    await llm.structured_call("grammar", nodes.GRAMMAR_SYSTEM, "I goes", GrammarFeedback)
    assert fake.calls == [M1] * (settings.llm_max_retries + 1) + [M2]


def test_retry_after_header_is_honoured():
    e = _status_error(groq.RateLimitError, 429)
    e.response.headers["retry-after"] = "3"
    assert 3 <= llm._retry_delay(e, 0) < 3.3
    e.response.headers["retry-after"] = "999"
    assert llm._retry_delay(e, 0) < settings.llm_max_wait_s + 0.3


async def test_invalid_json_falls_back(fake):
    fake.behaviour = {M1: "not json", M2: by_node}
    out = await llm.structured_call("grammar", nodes.GRAMMAR_SYSTEM, "fine", GrammarFeedback)
    assert out.has_error is False and fake.calls == [M1, M2]


async def test_all_models_down_raises(fake):
    fake.behaviour = {M1: _status_error(groq.NotFoundError, 404),
                      M2: _status_error(groq.InternalServerError, 500)}
    with pytest.raises(llm.LLMUnavailable):
        await llm.structured_call("grammar", nodes.GRAMMAR_SYSTEM, "x", GrammarFeedback)


# ---------------------------------------------------------------- nodes
async def test_repeated_error_detected_from_history(fake):
    fake.behaviour = {M1: by_node}
    out = await nodes.grammar_node({"transcript": "I goes", "error_history": ["subject_verb_agreement"]})
    assert out["repeated_error_types"] == ["subject_verb_agreement"]
    assert out["error_history"] == ["subject_verb_agreement"]  # no duplicates


async def test_has_error_flag_forced_consistent(fake):
    fake.behaviour = {M1: {"has_error": True, "errors": [], "corrected_sentence": "ok"}}
    out = await nodes.grammar_node({"transcript": "ok", "error_history": []})
    assert out["grammar"]["has_error"] is False


async def test_router_degrades_to_heuristic(fake):
    fake.behaviour = {M1: _status_error(groq.NotFoundError, 404), M2: _status_error(groq.NotFoundError, 404)}
    out = await nodes.router_node({"transcript": "When do I use the dative?"})
    assert out == {"intent": "question", "degraded": ["router"]}


def test_heuristic_keeps_question_form_practice_sentences():
    assert nodes.heuristic_intent("Where you are going") == "practice"
    assert nodes.heuristic_intent("Wann benutzt man den Dativ?") == "question"


# ---------------------------------------------------------------- API
@pytest.fixture
def api(fake, monkeypatch):
    from api.main import app
    monkeypatch.setattr("agent.rag.retrieve", lambda q, k=3: [])
    return TestClient(app)  # not used as a context manager -> lifespan warm-up skipped


def test_feedback_practice_runs_grammar_and_vocab(api, fake):
    fake.behaviour = {M1: by_node}
    r = api.post("/v1/feedback", json={"text": "I goes to school"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "practice" and body["grammar"]["has_error"]
    assert body["vocabulary"] == {"suggestions": []} and body["degraded"] == []


def test_feedback_question_uses_rag(api, fake):
    fake.behaviour = {M1: by_node}
    body = api.post("/v1/feedback", json={"text": "When do I use the dative?"}).json()
    assert body["intent"] == "question" and body["answer"]["grounded"] is True


def test_partial_failure_still_200(api, fake):
    def vocab_down(messages):
        if "word choices" in messages[0]["content"]:
            raise _status_error(groq.InternalServerError, 500)
        return by_node(messages)
    fake.behaviour = {M1: vocab_down, M2: vocab_down}

    async def create(model, messages, **kw):  # let the callable raise
        fake.calls.append(model)
        payload = vocab_down(messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
                               usage=None)
    fake.create = create
    body = api.post("/v1/feedback", json={"text": "I goes"}).json()
    assert body["degraded"] == ["vocabulary"] and body["grammar"]["has_error"]


def test_everything_down_is_503(api, fake):
    fake.behaviour = {M1: _status_error(groq.NotFoundError, 404), M2: _status_error(groq.NotFoundError, 404)}
    r = api.post("/v1/feedback", json={"text": "I goes"})
    assert r.status_code == 503 and r.json()["detail"]["trace_id"]


def test_stream_emits_node_events(api, fake):
    fake.behaviour = {M1: by_node}
    text = api.post("/v1/feedback/stream", json={"text": "I goes"}).text
    events = [l.split(": ", 1)[1] for l in text.splitlines() if l.startswith("event: ")]
    assert events[0] == "router" and set(events[1:3]) == {"grammar", "vocabulary"} and events[-1] == "done"
