"""
The only point the system talks to a language model.

Every LLM call in the project goes through 'complete()'. Concentrating 
LLM API calls in one place. The agents on agents that don't import this module are deterministic.
The test in 'tests/test_agent_boundaries.py' asserts this.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# --- environment ------------------------------------------------------------
# A .env file is only text until something parses it, so it is loaded here,
# at the one module that needs the key. The path is derived from __file__
# rather than the working directory, so it resolves the same whether the
# process was started by uvicorn from backend/, by a script from the repo
# root, or by the test runner.
#
# override=False means a real environment variable always wins over the file.
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError: 
        return
    load_dotenv(ENV_FILE, override=False)

_load_env_file()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

# Per-role model routing. For future flexibility if needed.
MODEL_BY_ROLE = {
    "market_analyst": DEFAULT_MODEL,
    "sentiment_analyser": DEFAULT_MODEL,
    "coordinator": DEFAULT_MODEL,
}


class LLMUnavailable(RuntimeError):
    """Raised when no API key is configured or the provider call fails."""


@dataclass
class LLMResponse:
    text: str
    model: str


def is_configured():
    """True when a live call could be attempted."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def model_for(role):
    return os.environ.get("OPENROUTER_MODEL") or MODEL_BY_ROLE.get(role, DEFAULT_MODEL)


def complete(
    prompt: str,
    role: str = "coordinator",
    system: str | None = None,
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> LLMResponse:
    """Send one prompt and return the completion. Raises 'LLMUnavailable' if the LLM is unavailable."""

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise LLMUnavailable(
            f"OPENROUTER_API_KEY is not set, so LLM-backed agents cannot run"
        )

    # Imported lazily so the API and the offline pipeline can run, and the
    # test suite can execute, without the openai package installed.
    try:
        from openai import OpenAI
    except ImportError as error: 
        raise LLMUnavailable(f"openai package not installed: {error}") from error


    # record the system message first so the user prompt can override it if needed
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})


    model = model_for(role)
    try:
        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature, # 0.0 is classification, 0.2 is exploratory
        )
    except Exception as error:
        raise LLMUnavailable(f"OpenRouter call failed: {error}") from error

    return LLMResponse(text=completion.choices[0].message.content or "",
                       model=model)


def complete_json(
    prompt: str,
    role: str = "coordinator",
    system: str | None = None,
    max_tokens: int = 800,
    temperature: float = 0.0,
):
    """
    Ask for a JSON reply and parse it.

    Models wrap JSON in prose or a fenced code block even when
    told not to, so the first balanced JSON object or array in the reply is
    extracted rather than parsing the whole response. Temperature defaults to
    zero because these calls are classification.

    Raises 'LLMUnavailable' if nothing parseable comes back.
    """
    
    response = complete(
        prompt, 
        role=role, 
        system=system,
        max_tokens=max_tokens, 
        temperature=temperature, 
    )
    
    parsed = extract_json(response.text)
    
    if parsed is None:
        raise LLMUnavailable(
            f"model returned no parseable JSON (first 200 chars: {response.text[:200]!r})"
        )
        
    return parsed

# First balanced JSON object or array in 'text', or None.
def extract_json(text):
    candidates = [(opening, closing, text.find(opening)) for opening, closing in (("{", "}"), ("[", "]"))]
    for opening, closing, start in sorted((candidate for candidate in candidates if candidate[2] != -1),key=lambda cad: cad[2]):

        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == opening:
                depth += 1
            elif character == closing:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        break
    
    return None
