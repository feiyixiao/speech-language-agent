"""Runtime configuration, read once from the environment.

Model names live here (not hard-coded in nodes) because the vendor can retire
a model at any time — see INCIDENTS.md #1.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _list(name: str, default: str) -> list[str]:
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


@dataclass
class Settings:
    # Ordered fallback chain: first model that answers with valid JSON wins.
    llm_models: list[str] = field(default_factory=lambda: _list(
        "LLM_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b,openai/gpt-oss-120b"))
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "20"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))  # per model, for 429/5xx
    llm_max_wait_s: float = float(os.getenv("LLM_MAX_WAIT_S", "10"))  # cap on a single retry-after wait
    # closed loop 1: confidence gate for grammar errors (off until eval/PROTOCOL.md justifies a threshold)
    grammar_conf_mode: str = os.getenv("GRAMMAR_CONF_MODE", "off")  # off | verbalized | consistency
    grammar_conf_threshold: float | None = (float(os.environ["GRAMMAR_CONF_THRESHOLD"])
                                            if os.getenv("GRAMMAR_CONF_THRESHOLD") else None)
    grammar_conf_k: int = int(os.getenv("GRAMMAR_CONF_K", "5"))
    grammar_conf_temperature: float = float(os.getenv("GRAMMAR_CONF_TEMPERATURE", "0.7"))
    # closed loop 2: ASR confidence gate (off by default)
    asr_conf_feature: str = os.getenv("ASR_CONF_FEATURE", "avg_prob")  # avg_prob|min_prob|no_speech|compression
    asr_conf_threshold: float | None = (float(os.environ["ASR_CONF_THRESHOLD"])
                                        if os.getenv("ASR_CONF_THRESHOLD") else None)
    asr_provider: str = os.getenv("ASR_PROVIDER", "groq")  # groq | local
    asr_model: str = os.getenv("ASR_MODEL", "whisper-large-v3-turbo")
    trace_dir: Path = Path(os.getenv("TRACE_DIR", str(ROOT / "traces")))
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    azure_speech_key: str = os.getenv("AZURE_SPEECH_KEY", "")
    azure_speech_region: str = os.getenv("AZURE_SPEECH_REGION", "")


settings = Settings()

# USD per 1M tokens (input, output) — APPROXIMATE, check console.groq.com/pricing
# before quoting numbers. Used only for relative cost-per-request comparisons.
PRICES = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "qwen/qwen3.8-27b": (0.29, 0.59),
}
