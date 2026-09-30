"""Structured LLM calls with retries and a cross-model fallback chain.

Failure policy (why each branch exists):
- 404 / model decommissioned / 400 bad request -> not retryable, go to next model
  immediately (INCIDENTS.md #1: the retired default model 404'd every request).
- 429 / 5xx / timeout -> retry the same model with exponential backoff, then fall back.
- Output that fails the pydantic schema -> next model (a different model is a
  better bet than re-sampling the same one).
If every model fails, LLMUnavailable is raised and the caller decides how to degrade.
"""
import asyncio
import json
import random
import time
from typing import TypeVar

import groq
from pydantic import BaseModel, ValidationError

from agent.config import PRICES, settings
from agent.tracing import record_span

T = TypeVar("T", bound=BaseModel)

_client: groq.AsyncGroq | None = None


class LLMUnavailable(RuntimeError):
    """All models in the fallback chain failed."""


def get_client() -> groq.AsyncGroq:
    global _client
    if _client is None:
        # we do our own retries so they are visible in traces
        _client = groq.AsyncGroq(api_key=settings.groq_api_key, max_retries=0,
                                 timeout=settings.llm_timeout_s)
    return _client


def _model_kwargs(model: str) -> dict:
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if model.startswith("qwen/"):
        return {"reasoning_format": "hidden"}
    return {}


def _cost(model: str, usage) -> float:
    if usage is None or model not in PRICES:
        return 0.0
    pin, pout = PRICES[model]
    return (usage.prompt_tokens * pin + usage.completion_tokens * pout) / 1e6


def _retry_delay(e: Exception, attempt: int) -> float:
    """Honour the provider's retry-after on 429 (INCIDENTS.md #3); else exponential backoff."""
    resp = getattr(e, "response", None)
    ra = resp.headers.get("retry-after") if resp is not None else None
    try:
        if ra is not None:
            return min(float(ra), settings.llm_max_wait_s) + random.random() * 0.25
    except ValueError:
        pass
    return min(8.0, 0.5 * 2 ** attempt) + random.random() * 0.25


def _retryable(e: Exception) -> bool:
    if isinstance(e, (groq.APITimeoutError, groq.APIConnectionError, groq.RateLimitError)):
        return True
    return isinstance(e, groq.APIStatusError) and e.status_code >= 500


async def structured_call(node: str, system: str, user: str, schema: type[T],
                          models: list[str] | None = None, temperature: float = 0.0) -> T:
    """Call the chain of models until one returns JSON that validates against `schema`."""
    models = models or settings.llm_models
    response_format = {"type": "json_schema", "json_schema": {
        "name": schema.__name__, "schema": schema.model_json_schema(), "strict": False}}
    errors = []
    for i, model in enumerate(models):
        for attempt in range(settings.llm_max_retries + 1):
            start = time.perf_counter()
            span = {"node": node, "model": model, "attempt": attempt, "fallback": i > 0}
            try:
                resp = await get_client().chat.completions.create(
                    model=model, temperature=temperature, response_format=response_format,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    **_model_kwargs(model))
                out = schema.model_validate(json.loads(resp.choices[0].message.content))
                u = resp.usage
                record_span(**span, status="ok",
                            latency_ms=round((time.perf_counter() - start) * 1000, 1),
                            prompt_tokens=u.prompt_tokens if u else None,
                            completion_tokens=u.completion_tokens if u else None,
                            cost_usd=round(_cost(model, u), 7))
                return out
            except (ValidationError, json.JSONDecodeError) as e:
                record_span(**span, status="invalid_output", error=str(e)[:200],
                            latency_ms=round((time.perf_counter() - start) * 1000, 1))
                errors.append(f"{model}: invalid output")
                break  # next model
            except Exception as e:
                record_span(**span, status="error", error=f"{type(e).__name__}: {e}"[:200],
                            latency_ms=round((time.perf_counter() - start) * 1000, 1))
                errors.append(f"{model}: {type(e).__name__}")
                if not _retryable(e) or attempt == settings.llm_max_retries:
                    break  # next model
                await asyncio.sleep(_retry_delay(e, attempt))
    raise LLMUnavailable("; ".join(errors))
