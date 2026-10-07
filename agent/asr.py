"""Speech-to-text. Default: Groq-hosted Whisper (keeps the container small and
cold starts fast — local whisper pulls in torch). Local openai-whisper is the
fallback when the API fails or ASR_PROVIDER=local."""
import time

from agent.asr_conf import asr_features
from agent.config import settings
from agent.tracing import record_span

_local_model = None


def _local(audio_path: str) -> dict:
    global _local_model
    import whisper  # heavy, imported lazily
    if _local_model is None:
        _local_model = whisper.load_model("base")
    r = _local_model.transcribe(audio_path)
    return {"text": r["text"].strip(), "language": r["language"], "provider": "local-whisper-base",
            "features": asr_features(r.get("segments"))}


def _groq(audio_path: str) -> dict:
    from groq import Groq
    client = Groq(api_key=settings.groq_api_key, timeout=60, max_retries=1)
    with open(audio_path, "rb") as f:
        r = client.audio.transcriptions.create(file=f, model=settings.asr_model,
                                               response_format="verbose_json")
    return {"text": r.text.strip(), "language": getattr(r, "language", None),
            "provider": f"groq/{settings.asr_model}",
            "features": asr_features(getattr(r, "segments", None))}


def transcribe_file(audio_path: str) -> dict:
    """Returns {text, language, provider, features}. features = agent.asr_conf scores ({} if unavailable)."""
    order = [_groq, _local] if settings.asr_provider == "groq" else [_local]
    last = None
    for i, fn in enumerate(order):
        start = time.perf_counter()
        try:
            out = fn(audio_path)
            record_span(node="asr", model=out["provider"], status="ok", fallback=i > 0,
                        latency_ms=round((time.perf_counter() - start) * 1000, 1))
            return out
        except Exception as e:
            last = e
            record_span(node="asr", model=fn.__name__, status="error", fallback=i > 0,
                        error=f"{type(e).__name__}: {e}"[:200],
                        latency_ms=round((time.perf_counter() - start) * 1000, 1))
    raise RuntimeError(f"ASR failed: {last}")
