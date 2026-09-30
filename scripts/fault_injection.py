"""Replay the same requests with a healthy config vs. a retired primary model.

    LLM_MODELS=llama-3.1-8b-instant,openai/gpt-oss-20b TRACE_DIR=traces/fi_retired python scripts/fault_injection.py
    TRACE_DIR=traces/fi_healthy python scripts/fault_injection.py
    python scripts/trace_report.py   (with the same TRACE_DIR)
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.pipeline import run_feedback  # noqa: E402

REQUESTS = [
    ("She don't like coffee.", "English"), ("Where you are going?", "English"),
    ("When do I use the present perfect?", "English"), ("I enjoy to swim in the lake.", "English"),
    ("Kannst du mich helfen?", "German"), ("Gestern ich habe Fußball gespielt.", "German"),
    ("Wann benutzt man den Dativ?", "German"), ("Ich fahre mit der Bus zur Uni.", "German"),
    ("He is good in mathematics.", "English"), ("Ich interessiere mich an Musik.", "German"),
]


async def main():
    for text, lang in REQUESTS:
        r = await run_feedback(text, lang)
        print(f"{r.intent:9} degraded={r.degraded} {text}")
        await asyncio.sleep(3)  # free-tier token budget


asyncio.run(main())
