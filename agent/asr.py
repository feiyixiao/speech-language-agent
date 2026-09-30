import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os

_model = None

def get_model(size="base"):
    global _model
    if _model is None:
        _model = whisper.load_model(size)
    return _model

def transcribe_file(audio_path: str) -> dict:
    """Transcribe an audio file. Returns transcript text and detected language."""
    model = get_model()
    result = model.transcribe(audio_path)
    return {
        "text": result["text"].strip(),
        "language": result["language"]
    }

def record_audio(duration: int = 5, sample_rate: int = 16000) -> str:
    """Record from microphone for `duration` seconds, save to temp file, return path."""
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name
