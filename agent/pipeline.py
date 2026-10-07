"""Single entry point used by the API, the Streamlit UI and the eval scripts."""
from collections.abc import AsyncIterator

from agent.graph import graph
from agent.schemas import FeedbackResult
from agent.tracing import current_trace_id, trace


def _initial(transcript, target_language, audio_path, error_history, asr=None) -> dict:
    state = {"transcript": transcript, "target_language": target_language,
             "audio_path": audio_path or "", "error_history": list(error_history or []),
             "degraded": []}
    if asr:  # {"score": float|None, "uncertain": bool} from agent.asr_conf.gate
        if asr.get("score") is not None:
            state["transcript_confidence"] = asr["score"]
        state["transcript_uncertain"] = bool(asr.get("uncertain"))
    return state


def _to_result(state: dict) -> FeedbackResult:
    return FeedbackResult(
        trace_id=current_trace_id(),
        **{k: state[k] for k in FeedbackResult.model_fields if k in state and k != "trace_id"})


async def run_feedback(transcript: str, target_language: str = "English",
                       audio_path: str | None = None,
                       error_history: list[str] | None = None,
                       asr: dict | None = None) -> FeedbackResult:
    with trace("feedback", target_language=target_language, has_audio=bool(audio_path)) as t:
        state = await graph.ainvoke(_initial(transcript, target_language, audio_path, error_history, asr))
        result = _to_result(state)
        t["attrs"].update(intent=result.intent, degraded=result.degraded)
        return result


async def stream_feedback(transcript: str, target_language: str = "English",
                          audio_path: str | None = None,
                          error_history: list[str] | None = None,
                          asr: dict | None = None) -> AsyncIterator[tuple[str, dict]]:
    """Yield (node_name, node_output) as each node finishes, then ("done", full_result)."""
    with trace("feedback_stream", target_language=target_language, has_audio=bool(audio_path)) as t:
        state = _initial(transcript, target_language, audio_path, error_history, asr)
        async for update in graph.astream(state, stream_mode="updates"):
            for node, out in update.items():
                out = out or {}
                for k, v in out.items():
                    state[k] = state.get(k, []) + v if k == "degraded" else v
                yield node, out
        result = _to_result(state)
        t["attrs"].update(intent=result.intent, degraded=result.degraded)
        yield "done", result.model_dump()
