import os
import json
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

def assess_pronunciation(audio_path: str, reference_text: str, language: str = "en-US") -> dict:
    """
    Assess pronunciation of an audio file against reference text.
    Returns scores for accuracy, fluency, completeness, and per-word details.
    """
    speech_config = speechsdk.SpeechConfig(
        subscription=os.getenv("AZURE_SPEECH_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION")
    )
    speech_config.speech_recognition_language = language

    audio_config = speechsdk.AudioConfig(filename=audio_path)

    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Word,
        enable_miscue=True
    )

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    pronunciation_config.apply_to(recognizer)

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        assessment = speechsdk.PronunciationAssessmentResult(result)
        words = []
        for word in assessment.words:
            words.append({
                "word": word.word,
                "accuracy_score": round(word.accuracy_score, 1),
                "error_type": word.error_type
            })
        return {
            "accuracy_score": round(assessment.accuracy_score, 1),
            "fluency_score": round(assessment.fluency_score, 1),
            "completeness_score": round(assessment.completeness_score, 1),
            "pronunciation_score": round(assessment.pronunciation_score, 1),
            "words": words,
            # what Azure itself heard; compared with the Whisper transcript in nodes.pronunciation_node
            "recognized_text": result.text,
        }
    else:
        return {"error": f"Recognition failed: {result.reason}"}


def format_pronunciation_feedback(scores: dict) -> str:
    if "error" in scores:
        return f"Pronunciation assessment failed: {scores['error']}"

    lines = [
        f"**Pronunciation Score: {scores['pronunciation_score']}/100**",
        f"- Accuracy: {scores['accuracy_score']}",
        f"- Fluency: {scores['fluency_score']}",
        f"- Completeness: {scores['completeness_score']}",
        "",
        "**Word-level feedback:**"
    ]

    for w in scores.get("words", []):
        if w["accuracy_score"] < 80 or w["error_type"] != "None":
            lines.append(f"- ⚠️ **{w['word']}** — score: {w['accuracy_score']}, issue: {w['error_type']}")
        else:
            lines.append(f"- ✓ {w['word']} ({w['accuracy_score']})")

    return "\n".join(lines)
