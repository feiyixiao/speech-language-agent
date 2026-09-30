"""Fail (exit 1) if any configured LLM model is gone. Used in CI and before deploys."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import groq  # noqa: E402

from agent.config import settings  # noqa: E402

available = {m.id for m in groq.Groq(api_key=settings.groq_api_key).models.list().data}
missing = [m for m in settings.llm_models + [settings.asr_model] if m not in available]
print("configured:", settings.llm_models + [settings.asr_model])
if missing:
    print("MISSING (retired or renamed):", missing)
    sys.exit(1)
print("all models available")
