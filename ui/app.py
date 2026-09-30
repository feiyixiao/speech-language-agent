import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from agent.asr import transcribe_file
from agent.pipeline import run_feedback

st.set_page_config(page_title="Language Learning Agent", page_icon="🎙️", layout="centered")

st.title("Language Learning Agent")
st.caption("Speak or type a sentence for grammar, vocabulary and pronunciation feedback — "
           "or ask a grammar question.")

if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "error_history" not in st.session_state:
    st.session_state.error_history = []

language = st.selectbox("Target language", ["English", "German", "Japanese", "Mandarin Chinese"])
tab1, tab2 = st.tabs(["Upload audio", "Type text"])


def show(r):
    if r.degraded:
        st.warning(f"Some parts are temporarily unavailable: {', '.join(r.degraded)}")
    if r.intent == "question" and r.answer:
        st.subheader("Answer")
        st.markdown(r.answer.answer)
        if r.retrieved_sections:
            st.caption(("From the grammar notes: " if r.answer.grounded else
                        "Not covered by the grammar notes — general answer. Closest notes: ")
                       + " · ".join(r.retrieved_sections))
    if r.grammar:
        st.subheader("Grammar")
        if not r.grammar.errors:
            st.success("Your sentence is grammatically correct!")
        for e in r.grammar.errors:
            again = " — **you made this mistake before**" if e.error_type in r.repeated_error_types else ""
            st.markdown(f"- ~~{e.original}~~ → **{e.correction}** "
                        f"(`{e.error_type}`){again}  \n  {e.explanation}")
        if r.grammar.errors:
            st.markdown(f"**Corrected:** {r.grammar.corrected_sentence}")
    if r.vocabulary and r.vocabulary.suggestions:
        st.subheader("Vocabulary")
        for s in r.vocabulary.suggestions:
            st.markdown(f"- *{s.original}* → **{s.better}** — {s.why}")
    if r.pronunciation and "error" not in r.pronunciation:
        p = r.pronunciation
        st.subheader(f"Pronunciation: {p['pronunciation_score']}/100")
        st.caption(f"Accuracy {p['accuracy_score']} · Fluency {p['fluency_score']} · "
                   f"Completeness {p['completeness_score']}")
        weak = [w for w in p["words"] if w["accuracy_score"] < 80 or w["error_type"] != "None"]
        for w in weak:
            st.markdown(f"- ⚠️ **{w['word']}** — {w['accuracy_score']} ({w['error_type']})")
    st.caption(f"trace_id: `{r.trace_id}`")


def run(transcript, audio_path=None):
    with st.spinner("Thinking..."):
        r = asyncio.run(run_feedback(transcript, language, audio_path, st.session_state.error_history))
    st.session_state.error_history = r.error_history
    st.session_state.feedback_log.append(r)
    show(r)


with tab1:
    uploaded = st.file_uploader("Upload a .wav, .mp3, or .m4a file", type=["wav", "mp3", "m4a"])
    if uploaded:
        st.audio(uploaded)
        # keep the file for the whole session: Streamlit reruns the script on every click,
        # and the prototype deleted it before the pronunciation step could read it
        key = f"audio_{uploaded.file_id}"
        if key not in st.session_state:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.getvalue())
            with st.spinner("Transcribing..."):
                st.session_state[key] = (tmp.name, transcribe_file(tmp.name))
        path, asr = st.session_state[key]
        st.success(f"Transcribed ({asr['provider']}) — detected: **{asr['language']}**")
        transcript = st.text_area("Transcript (edit if needed)", value=asr["text"], height=80)
        if st.button("Get feedback", type="primary"):
            run(transcript, path)

with tab2:
    text_input = st.text_area("Type a sentence or a question", height=100,
                              placeholder="e.g. 'She don't like coffee.' or 'When do I use the dative?'")
    if st.button("Get feedback", key="text_btn", type="primary"):
        if text_input.strip():
            run(text_input.strip())
        else:
            st.warning("Please enter some text first.")

if st.session_state.feedback_log:
    st.divider()
    st.subheader("Session")
    if st.session_state.error_history:
        st.caption("Error types so far: " + ", ".join(st.session_state.error_history))
    for i, r in enumerate(reversed(st.session_state.feedback_log), 1):
        with st.expander(f"{i}. {r.transcript[:60]}"):
            show(r)
    if st.button("Clear session"):
        st.session_state.error_history = []
        st.session_state.feedback_log = []
        st.rerun()
