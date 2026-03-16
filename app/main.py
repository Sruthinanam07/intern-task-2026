import hashlib
import json
import logging
import os
import time
from functools import lru_cache

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models import FeedbackRequest, FeedbackResponse
from app.prompt import build_prompt

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pangea Language Feedback API", version="1.0.0")

# In-memory cache: sha256(sentence+target+native) -> FeedbackResponse dict
_cache: dict[str, dict] = {}

def _cache_key(req: FeedbackRequest) -> str:
    raw = f"{req.sentence.strip()}|{req.target_language.lower()}|{req.native_language.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _call_claude(req: FeedbackRequest) -> FeedbackResponse:
    api_key = os.getenv("ANTHROPIC_API_KEY", "test-key")
    client = anthropic.Anthropic(api_key=api_key)

    system_prompt, user_message = build_prompt(req)

    for attempt in range(3):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = message.content[0].text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data = json.loads(raw)
            return FeedbackResponse(**data)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Parse error on attempt {attempt + 1}: {e}")
            if attempt == 2:
                raise HTTPException(status_code=502, detail="LLM returned unparseable response after 3 attempts.")
            time.sleep(1)
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise HTTPException(status_code=502, detail=f"LLM API error: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    key = _cache_key(req)
    if key in _cache:
        logger.info("Cache hit")
        return FeedbackResponse(**_cache[key])

    result = _call_claude(req)
    _cache[key] = result.model_dump()
    return result


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
