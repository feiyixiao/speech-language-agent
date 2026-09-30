"""Typed contracts between the LLM and the rest of the system.

Every LLM node returns one of these (JSON-schema constrained + pydantic
validated), so downstream logic never parses free text. The old prototype
decided "was there an error?" by string-matching "grammatically correct" in
the reply — the eval in eval/ measures what that cost.
"""
from typing import Literal

from pydantic import BaseModel, Field

ErrorType = Literal[
    "subject_verb_agreement", "tense", "verb_form", "article", "preposition",
    "noun_number", "word_order", "gender_case", "pronoun", "spelling", "other",
]


class RouterDecision(BaseModel):
    intent: Literal["practice", "question"] = Field(
        description="practice = the learner produced a sentence to be checked; "
                    "question = the learner asks about the language (grammar, meaning, pronunciation)")
    reason: str = Field(description="one short sentence")


class GrammarError(BaseModel):
    original: str = Field(description="the wrong span, copied from the input")
    correction: str
    error_type: ErrorType
    explanation: str = Field(description="one sentence, learner-friendly")


class GrammarFeedback(BaseModel):
    has_error: bool
    errors: list[GrammarError] = Field(default_factory=list)
    corrected_sentence: str


class VocabSuggestion(BaseModel):
    original: str
    better: str
    why: str


class VocabFeedback(BaseModel):
    suggestions: list[VocabSuggestion] = Field(default_factory=list, max_length=3)


class KnowledgeAnswer(BaseModel):
    answer: str
    grounded: bool = Field(description="true only if the answer is supported by the given context")


class FeedbackResult(BaseModel):
    """What the API returns."""
    trace_id: str | None = None
    intent: str
    transcript: str
    grammar: GrammarFeedback | None = None
    repeated_error_types: list[str] = Field(default_factory=list)
    vocabulary: VocabFeedback | None = None
    pronunciation: dict | None = None
    answer: KnowledgeAnswer | None = None
    retrieved_sections: list[str] = Field(default_factory=list)
    error_history: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list,
                                description="components that failed and were skipped")
