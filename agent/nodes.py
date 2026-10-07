"""LangGraph nodes. Each node is async, returns only the keys it owns, and
degrades (records itself in `degraded`) instead of failing the whole request."""
import asyncio
import re

from agent.confidence import CONF_SUFFIX, consistency_score
from agent.config import settings
from agent.llm import LLMUnavailable, structured_call
from agent.schemas import (GrammarFeedback, GrammarFeedbackConf, GrammarResult, KnowledgeAnswer,
                           RouterDecision, VocabFeedback)
from agent.textmetrics import wer

# v3 router (eval/RESULTS.md): v2 sent everyday questions written in the target
# language ("Kannst du mich helfen?") to the knowledge branch — INCIDENTS.md #2.
ROUTER_SYSTEM = (
    "You route inputs from a learner of {lang}. Decide the intent:\n"
    "- practice: any sentence the learner produced in {lang} — statements AND everyday questions "
    "(asking for help, time, prices, directions, feelings). These get grammar feedback.\n"
    "- question: the learner asks ABOUT {lang} itself — a grammar rule, which form is correct, "
    "a word's meaning, or pronunciation. Usually it mentions a word in quotes, a grammar term, "
    "or 'say', 'mean', 'use', 'pronounce', 'correct'.\n"
    "Examples (none of these are in the eval set): 'Could you open the window?' -> practice. "
    "'Wie viel kostet das Ticket?' -> practice. 'Is it \"much\" or \"many\" people?' -> question. "
    "'Was bedeutet \"Feierabend\"?' -> question. 'Is \"I am boring\" wrong when I mean bored?' -> question.\n"
    "Return JSON."
)

GRAMMAR_SYSTEM = (
    "You are a precise {lang} teacher. Check the learner's sentence for grammar errors only "
    "(not style, not punctuation, not capitalisation of a spoken transcript). "
    "Do NOT invent errors: if the sentence is grammatical, return has_error=false and an empty list. "
    "For each real error copy the wrong span into `original`, give the `correction`, pick the "
    "closest error_type, and explain in one sentence. Return JSON."
)

VOCAB_SYSTEM = (
    "You are a {lang} teacher. Suggest at most 3 more natural or more precise word choices for "
    "the learner's sentence. If the wording is already natural, return an empty list. Return JSON."
)

RAG_SYSTEM = (
    "You are a language teacher. Answer the learner's question in at most 4 sentences using the "
    "context below. Set grounded=true only if the context supports the answer; otherwise answer "
    "from general knowledge and set grounded=false. Return JSON.\n\nContext:\n{context}"
)

_QUESTION_RE = re.compile(
    r"^(when|how|what|why|which|is it|can i|should i|wann|wie|was|warum|welche|ist es)\b", re.I)


def heuristic_intent(text: str) -> str:
    """Fallback router used only when every LLM is down."""
    t = text.strip()
    return "question" if _QUESTION_RE.match(t) and t.endswith("?") else "practice"


async def router_node(state: dict) -> dict:
    try:
        system = ROUTER_SYSTEM.format(lang=state.get("target_language", "English"))
        d = await structured_call("router", system, state["transcript"], RouterDecision)
        return {"intent": d.intent}
    except LLMUnavailable:
        return {"intent": heuristic_intent(state["transcript"]), "degraded": ["router"]}


async def _consistency(system: str, text: str) -> float | None:
    """Closed loop 1, signal B: share of k sampled runs that also report an error."""
    runs = await asyncio.gather(*(
        structured_call("grammar_sample", system, text, GrammarFeedback,
                        temperature=settings.grammar_conf_temperature)
        for _ in range(settings.grammar_conf_k)), return_exceptions=True)
    return consistency_score([bool(r.errors) if isinstance(r, GrammarFeedback) else None for r in runs])


async def grammar_node(state: dict) -> dict:
    """Grammar feedback with an optional confidence gate (GRAMMAR_CONF_MODE, off by default).

    A flagged error is SUPPRESSED (kept in suppressed_errors, not shown, not added to the error
    history) when its confidence is below GRAMMAR_CONF_THRESHOLD, or when the ASR transcript
    itself is uncertain (closed loop 2). Nothing is suppressed when the gate is off."""
    lang = state.get("target_language", "English")
    mode = settings.grammar_conf_mode
    system = GRAMMAR_SYSTEM.format(lang=lang) + (CONF_SUFFIX if mode == "verbalized" else "")
    try:
        g = await structured_call("grammar", system, state["transcript"],
                                  GrammarFeedbackConf if mode == "verbalized" else GrammarFeedback)
    except LLMUnavailable:
        return {"degraded": ["grammar"]}
    # keep the flag and the list consistent — models sometimes disagree with themselves
    g.has_error = bool(g.errors)
    degraded = []
    conf = None
    if g.has_error and mode == "verbalized":
        conf = g.confidence
    elif g.has_error and mode == "consistency":
        conf = await _consistency(system, state["transcript"])
        if conf is None:
            degraded.append("grammar_confidence")
    below = (conf is not None and settings.grammar_conf_threshold is not None
             and conf < settings.grammar_conf_threshold)
    uncertain = g.has_error and (below or bool(state.get("transcript_uncertain")))
    out = GrammarResult(**g.model_dump(exclude={"confidence"}), confidence=conf, confidence_mode=mode,
                        uncertain=uncertain)
    if uncertain:
        out.suppressed_errors, out.errors = g.errors, []
        out.has_error, out.corrected_sentence = False, state["transcript"]
    history = list(state.get("error_history", []))
    types = sorted({e.error_type for e in out.errors})
    repeated = [t for t in types if t in history]
    res = {"grammar": out.model_dump(), "repeated_error_types": repeated,
           "error_history": history + [t for t in types if t not in history]}
    if degraded:
        res["degraded"] = degraded
    return res


async def vocabulary_node(state: dict) -> dict:
    lang = state.get("target_language", "English")
    try:
        v = await structured_call("vocabulary", VOCAB_SYSTEM.format(lang=lang),
                                  state["transcript"], VocabFeedback, temperature=0.3)
        return {"vocabulary": v.model_dump()}
    except LLMUnavailable:
        return {"degraded": ["vocabulary"]}


LANG_CODES = {"English": "en-US", "German": "de-DE", "Japanese": "ja-JP", "Mandarin Chinese": "zh-CN"}


async def pronunciation_node(state: dict) -> dict:
    if not state.get("audio_path"):
        return {}
    if not (settings.azure_speech_key and settings.azure_speech_region):
        return {"degraded": ["pronunciation"]}
    from agent.pronunciation import assess_pronunciation
    lang = LANG_CODES.get(state.get("target_language", "English"), "en-US")
    try:
        scores = await asyncio.wait_for(asyncio.to_thread(
            assess_pronunciation, state["audio_path"], state["transcript"], lang), timeout=30)
    except Exception:
        return {"degraded": ["pronunciation"]}
    if "error" in scores:
        return {"degraded": ["pronunciation"], "pronunciation": scores}
    # The reference text for Azure is the Whisper transcript, so a word Whisper "corrected" would be
    # scored against the corrected word. Surface how much the two recognisers disagree.
    if scores.get("recognized_text") is not None:
        scores["reference_mismatch_wer"] = round(wer(state["transcript"], scores["recognized_text"]), 3)
    return {"pronunciation": scores}


async def rag_node(state: dict) -> dict:
    from agent.rag import retrieve
    try:
        docs = await asyncio.to_thread(retrieve, state["transcript"], 3)
    except Exception:
        docs = []
    context = "\n\n".join(d.page_content for d in docs) or "(no context)"
    sections = [d.metadata.get("section", "?") for d in docs]
    try:
        a = await structured_call("rag", RAG_SYSTEM.format(context=context),
                                  state["transcript"], KnowledgeAnswer)
        return {"answer": a.model_dump(), "retrieved_sections": sections}
    except LLMUnavailable:
        return {"degraded": ["rag"], "retrieved_sections": sections}
