import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import tempfile
from agent.asr import transcribe_file
from agent.nodes import grammar_node, vocabulary_node, pronunciation_node

st.set_page_config(page_title="Language Learning Agent", page_icon="🎙️", layout="centered")

st.title("Language Learning Agent")
st.caption("Upload a recording to get grammar, vocabulary, and pronunciation feedback all at once.")

if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "error_history" not in st.session_state:
    st.session_state.error_history = []

st.divider()

language = st.selectbox("Target language", ["English", "German", "Japanese", "Mandarin Chinese"])

st.divider()

tab1, tab2 = st.tabs(["Upload audio", "Type text"])

def run_all_feedback(transcript, language, audio_path, error_history):
    base_state = {
        "transcript": transcript,
        "target_language": language,
        "task": "",
        "result": "",
        "error_history": error_history,
        "audio_path": audio_path,
    }

    with st.spinner("Checking grammar..."):
        g = grammar_node(base_state)
    with st.spinner("Checking vocabulary..."):
        v = vocabulary_node(base_state)

    st.subheader("Grammar")
    st.markdown(g["result"])

    st.subheader("Vocabulary")
    st.markdown(v["result"])

    if audio_path:
        with st.spinner("Assessing pronunciation..."):
            p = pronunciation_node({**base_state, "audio_path": audio_path})
        st.subheader("Pronunciation")
        st.markdown(p["result"])

    updated_history = g.get("error_history", error_history)
    st.session_state.error_history = updated_history
    st.session_state.feedback_log.append({
        "transcript": transcript,
        "grammar": g["result"],
        "vocabulary": v["result"],
        "pronunciation": p["result"] if audio_path else "—",
    })

with tab1:
    uploaded = st.file_uploader("Upload a .wav, .mp3, or .m4a file", type=["wav", "mp3", "m4a"])

    if uploaded:
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.audio(uploaded)

        with st.spinner("Transcribing with Whisper..."):
            asr_result = transcribe_file(tmp_path)

        transcript = asr_result["text"]
        detected_lang = asr_result["language"]
        st.success(f"Transcription complete — detected: **{detected_lang}**")
        transcript = st.text_area("Transcript (edit if needed)", value=transcript, height=80)

        if st.button("Get feedback", type="primary"):
            st.divider()
            run_all_feedback(transcript, language, tmp_path, st.session_state.error_history)

        os.unlink(tmp_path)

with tab2:
    text_input = st.text_area("Type or paste a sentence", height=100,
                               placeholder="e.g. 'I goes to school' or 'She don't like coffee.'")
    if st.button("Get feedback", key="text_btn", type="primary"):
        if text_input.strip():
            st.divider()
            run_all_feedback(text_input.strip(), language, "", st.session_state.error_history)
        else:
            st.warning("Please enter some text first.")

# Session history
if st.session_state.feedback_log:
    st.divider()
    st.subheader("Session history")
    for i, entry in enumerate(reversed(st.session_state.feedback_log), 1):
        with st.expander(f"{i}. {entry['transcript'][:60]}"):
            st.markdown(f"**Grammar:** {entry['grammar'][:120]}...")
            st.markdown(f"**Vocabulary:** {entry['vocabulary'][:120]}...")
            if entry['pronunciation'] != "—":
                st.markdown(f"**Pronunciation:** {entry['pronunciation'][:120]}...")

    if st.button("Clear session"):
        st.session_state.error_history = []
        st.session_state.feedback_log = []
        st.rerun()

st.divider()
st.caption("Whisper · LangGraph · Groq · Azure Speech · ChromaDB")
