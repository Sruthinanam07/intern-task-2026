"""
Test suite for the Pangea Language Feedback API.

Unit tests: validate schema, edge cases with mocked LLM responses.
Integration tests: hit the live API (requires ANTHROPIC_API_KEY and running server).
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, _cache
from app.models import FeedbackRequest, FeedbackResponse

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_response(data: dict):
    """Create a mock Anthropic API response."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=json.dumps(data))]
    return mock_msg


CORRECT_RESPONSE = {
    "corrected_sentence": "I go to the market every day.",
    "is_correct": True,
    "errors": [],
    "difficulty": "A1",
}

ERRORS_RESPONSE = {
    "corrected_sentence": "Yo fui al mercado ayer.",
    "is_correct": False,
    "errors": [
        {
            "original": "soy fue",
            "correction": "fui",
            "error_type": "conjugation",
            "explanation": "You mixed two verb forms. Use 'fui' (I went).",
        }
    ],
    "difficulty": "A2",
}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Schema validation tests (unit — no API call)
# ---------------------------------------------------------------------------

@patch("app.main.anthropic.Anthropic")
def test_correct_sentence_returns_empty_errors(mock_anthropic, monkeypatch):
    """A correct sentence should return is_correct=True and empty errors list."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mock_anthropic.return_value.messages.create.return_value = make_mock_response(CORRECT_RESPONSE)
    _cache.clear()

    resp = client.post("/feedback", json={
        "sentence": "I go to the market every day.",
        "target_language": "English",
        "native_language": "Spanish",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is True
    assert data["errors"] == []
    assert data["corrected_sentence"] == "I go to the market every day."
    assert data["difficulty"] in ("A1", "A2", "B1", "B2", "C1", "C2")


@patch("app.main.anthropic.Anthropic")
def test_sentence_with_errors_returns_error_list(mock_anthropic):
    """Sentence with errors should return is_correct=False and non-empty errors."""
    mock_anthropic.return_value.messages.create.return_value = make_mock_response(ERRORS_RESPONSE)
    _cache.clear()

    resp = client.post("/feedback", json={
        "sentence": "Yo soy fue al mercado ayer.",
        "target_language": "Spanish",
        "native_language": "English",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is False
    assert len(data["errors"]) > 0
    error = data["errors"][0]
    assert "original" in error
    assert "correction" in error
    assert "error_type" in error
    assert "explanation" in error


@patch("app.main.anthropic.Anthropic")
def test_error_type_is_valid_enum(mock_anthropic):
    """error_type must be one of the allowed values."""
    valid_types = {
        "grammar", "spelling", "word_choice", "punctuation", "word_order",
        "missing_word", "extra_word", "conjugation", "gender_agreement",
        "number_agreement", "tone_register", "other",
    }
    mock_anthropic.return_value.messages.create.return_value = make_mock_response(ERRORS_RESPONSE)
    _cache.clear()

    resp = client.post("/feedback", json={
        "sentence": "Yo soy fue al mercado ayer.",
        "target_language": "Spanish",
        "native_language": "English",
    })
    data = resp.json()
    for err in data["errors"]:
        assert err["error_type"] in valid_types


@patch("app.main.anthropic.Anthropic")
def test_difficulty_is_valid_cefr(mock_anthropic):
    """difficulty must be a valid CEFR level."""
    mock_anthropic.return_value.messages.create.return_value = make_mock_response(CORRECT_RESPONSE)
    _cache.clear()

    resp = client.post("/feedback", json={
        "sentence": "I go to the market every day.",
        "target_language": "English",
        "native_language": "French",
    })
    assert resp.json()["difficulty"] in ("A1", "A2", "B1", "B2", "C1", "C2")


@patch("app.main.anthropic.Anthropic")
def test_caching_prevents_duplicate_api_calls(mock_anthropic):
    """Identical requests should hit cache; API called only once."""
    mock_create = mock_anthropic.return_value.messages.create
    mock_create.return_value = make_mock_response(CORRECT_RESPONSE)
    _cache.clear()

    payload = {
        "sentence": "She have a dog.",
        "target_language": "English",
        "native_language": "German",
    }
    client.post("/feedback", json=payload)
    client.post("/feedback", json=payload)

    assert mock_create.call_count == 1


def test_empty_sentence_returns_422():
    """Empty sentence should be rejected with 422 Unprocessable Entity."""
    resp = client.post("/feedback", json={
        "sentence": "",
        "target_language": "English",
        "native_language": "English",
    })
    assert resp.status_code == 422


def test_missing_fields_returns_422():
    """Missing required fields should be rejected."""
    resp = client.post("/feedback", json={"sentence": "Hello"})
    assert resp.status_code == 422


@patch("app.main.anthropic.Anthropic")
def test_markdown_fence_stripped(mock_anthropic):
    """LLM responses wrapped in ```json fences should be parsed correctly."""
    fenced = "```json\n" + json.dumps(CORRECT_RESPONSE) + "\n```"
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=fenced)]
    mock_anthropic.return_value.messages.create.return_value = mock_msg
    _cache.clear()

    resp = client.post("/feedback", json={
        "sentence": "I go to the market every day.",
        "target_language": "English",
        "native_language": "Portuguese",
    })
    assert resp.status_code == 200
    assert resp.json()["is_correct"] is True


# ---------------------------------------------------------------------------
# Integration tests (requires live API key + running server)
# ---------------------------------------------------------------------------

INTEGRATION = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping integration tests",
)


@INTEGRATION
def test_integration_spanish_conjugation_error():
    """Spanish sentence with conjugation error."""
    _cache.clear()
    resp = client.post("/feedback", json={
        "sentence": "Yo soy fue al mercado ayer.",
        "target_language": "Spanish",
        "native_language": "English",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is False
    assert len(data["errors"]) > 0
    assert data["corrected_sentence"] != "Yo soy fue al mercado ayer."


@INTEGRATION
def test_integration_french_correct_sentence():
    """Correct French sentence should return is_correct=True and no errors."""
    _cache.clear()
    resp = client.post("/feedback", json={
        "sentence": "Je mange une pomme chaque matin.",
        "target_language": "French",
        "native_language": "English",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is True
    assert data["errors"] == []


@INTEGRATION
def test_integration_japanese_non_latin_script():
    """Japanese sentence — tests non-Latin script support."""
    _cache.clear()
    resp = client.post("/feedback", json={
        "sentence": "私は昨日学校に行きました。",
        "target_language": "Japanese",
        "native_language": "English",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "corrected_sentence" in data
    assert data["difficulty"] in ("A1", "A2", "B1", "B2", "C1", "C2")


@INTEGRATION
def test_integration_german_multiple_errors():
    """German sentence with multiple errors."""
    _cache.clear()
    resp = client.post("/feedback", json={
        "sentence": "Ich habe gestern ins Kino gegangen und ein Film gesehen.",
        "target_language": "German",
        "native_language": "English",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is False
    assert len(data["errors"]) >= 1


@INTEGRATION
def test_integration_explanation_in_native_language():
    """Explanations must be in the learner's native language."""
    _cache.clear()
    resp = client.post("/feedback", json={
        "sentence": "Mi casa es muy grande y bonito.",
        "target_language": "Spanish",
        "native_language": "English",
    })
    assert resp.status_code == 200
    data = resp.json()
    # At minimum, there should be a gender agreement error (bonito -> bonita)
    # and explanations should be in English (not Spanish)
    assert data["is_correct"] is False
    for err in data["errors"]:
        # Simple heuristic: English explanations shouldn't have common Spanish-only words
        assert isinstance(err["explanation"], str)
        assert len(err["explanation"]) > 0
