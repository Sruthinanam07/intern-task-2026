from typing import Literal
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=2000)
    target_language: str = Field(..., min_length=1)
    native_language: str = Field(..., min_length=1)


class ErrorItem(BaseModel):
    original: str
    correction: str
    error_type: Literal[
        "grammar",
        "spelling",
        "word_choice",
        "punctuation",
        "word_order",
        "missing_word",
        "extra_word",
        "conjugation",
        "gender_agreement",
        "number_agreement",
        "tone_register",
        "other",
    ]
    explanation: str


class FeedbackResponse(BaseModel):
    corrected_sentence: str
    is_correct: bool
    errors: list[ErrorItem]
    difficulty: Literal["A1", "A2", "B1", "B2", "C1", "C2"]
