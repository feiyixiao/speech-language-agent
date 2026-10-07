"""HTTP API.

POST /v1/feedback          JSON text in  -> FeedbackResult
POST /v1/feedback/audio    multipart audio -> ASR -> FeedbackResult
POST /v1/feedback/stream   JSON text in  -> Server-Sent Events, one event per finished node
GET  /healthz              liveness (cheap)
GET  /readyz               readiness: every configured model answers (catches retired models)
"""
import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.asr import transcribe_file
from agent.asr_conf import gate as asr_gate
from agent.config import settings
from agent.llm import get_client
from agent.pipeline import run_feedback, stream_feedback
from agent.schemas import FeedbackResult


@asynccontextmanager
async def lifespan(_app):
    # load the embedding model + vector index before the first request:
    # without this the first question took ~9 s (trace p95, 2026-09-30)
    from agent.rag import get_vectorstore
    await asyncio.to_thread(get_vectorstore)
    yield


app = FastAPI(title="Speech Language Learning Agent", version="0.2.0", lifespan=lifespan)

MAX_AUDIO_BYTES = 10 * 1024 * 1024


class FeedbackRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    target_language: str = "English"
    error_history: list[str] = Field(default_factory=list)


def _check_not_all_down(r: FeedbackResult) -> FeedbackResult:
    needed = {"grammar", "vocabulary"} if r.intent == "practice" else {"rag"}
    if needed <= set(r.degraded):
        raise HTTPException(503, detail={"message": "all LLM providers unavailable",
                                         "trace_id": r.trace_id})
    return r


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    async def ping(model):
        try:
            await get_client().chat.completions.create(
                model=model, max_tokens=64, messages=[{"role": "user", "content": "ok"}])
            return model, "ok"
        except Exception as e:
            return model, f"{type(e).__name__}: {str(e)[:120]}"
    results = dict(await asyncio.gather(*(ping(m) for m in settings.llm_models)))
    ok = any(v == "ok" for v in results.values())
    if not ok:
        raise HTTPException(503, detail=results)
    return {"status": "ok" if all(v == "ok" for v in results.values()) else "degraded",
            "models": results}


@app.post("/v1/feedback", response_model=FeedbackResult)
async def feedback(req: FeedbackRequest):
    r = await run_feedback(req.text, req.target_language, None, req.error_history)
    return _check_not_all_down(r)


@app.post("/v1/feedback/audio", response_model=FeedbackResult)
async def feedback_audio(file: UploadFile = File(...), target_language: str = Form("English"),
                         error_history: str = Form("[]")):
    data = await file.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio larger than 10 MB")
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        try:
            asr = await asyncio.to_thread(transcribe_file, path)
        except RuntimeError as e:
            raise HTTPException(502, f"transcription failed: {e}")
        if not asr["text"]:
            raise HTTPException(422, "no speech detected")
        score, uncertain = asr_gate(asr.get("features") or {}, settings.asr_conf_feature,
                                    settings.asr_conf_threshold)
        r = await run_feedback(asr["text"], target_language, path, json.loads(error_history),
                               asr={"score": score, "uncertain": uncertain})
        return _check_not_all_down(r)
    finally:
        os.unlink(path)


@app.post("/v1/feedback/stream")
async def feedback_stream(req: FeedbackRequest):
    async def events():
        async for node, payload in stream_feedback(req.text, req.target_language, None,
                                                   req.error_history):
            yield f"event: {node}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")
