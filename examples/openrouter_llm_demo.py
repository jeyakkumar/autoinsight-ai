"""Call Claude Opus 5.5 through OpenRouter.

Run from the repository root:
    uv run python examples/openrouter_llm_demo.py --check-key
    uv run python examples/openrouter_llm_demo.py --evaluate
    uv run python examples/openrouter_llm_demo.py --dry-run

The request uses OpenRouter's OpenAI-compatible Chat Completions endpoint. It is
separate from the JEV Decisions API because Claude is a generative model.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_URL = "https://openrouter.ai/api/v1/key"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class DemoSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_llm_model: str = Field(default="anthropic/claude-opus-5.5", min_length=1)


class Usage(BaseModel):
    prompt_tokens: int = Field(ge=0, strict=True)
    completion_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(ge=0, strict=True)
    cost: float | None = Field(default=None, ge=0)


class CompletionResponse(BaseModel):
    id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    choices: list[dict] = Field(min_length=1)
    usage: Usage | None = None


def build_request(model: str) -> dict:
    """Ask for an explanation grounded in deterministic vehicle evidence."""
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You explain verified analytics results. Use only the supplied evidence. "
                    "Do not invent numbers, vehicle facts, recalls, or safety conclusions."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "vehicle": "Example SUV 2022",
                        "verified_insights": [
                            {
                                "claim": "Listing price is 17.9% below the peer median.",
                                "evidence": {
                                    "listing_price_usd": 24800,
                                    "peer_median_price_usd": 30200,
                                    "peer_count": 143,
                                },
                            }
                        ],
                        "unverified_or_missing": [
                            "No recall data was supplied.",
                            "No NHTSA safety rating was supplied.",
                        ],
                        "task": "Write a concise analyst explanation and next investigation step.",
                    },
                    indent=2,
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }


def extract_text(choice: dict) -> str:
    message = choice.get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def run_request(client: httpx.Client, *, evaluate: bool, model: str) -> dict:
    started = perf_counter()
    response = client.post(CHAT_URL, json=build_request(model)) if evaluate else client.get(KEY_URL)
    response.raise_for_status()
    data = response.json()
    if not evaluate:
        if not isinstance(data.get("data"), dict):
            raise ValueError("unexpected OpenRouter key response")
        safe = data["data"]
        return {
            "status": "api_key_check_succeeded",
            "elapsed_ms": round((perf_counter() - started) * 1000, 1),
            "limit_remaining": safe.get("limit_remaining"),
            "usage": safe.get("usage"),
            "is_free_tier": safe.get("is_free_tier"),
        }
    parsed = CompletionResponse.model_validate(data)
    text = extract_text(parsed.choices[0])
    if not text:
        raise ValueError("response contained no assistant text")
    return {
        "status": "llm_evaluation_succeeded",
        "elapsed_ms": round((perf_counter() - started) * 1000, 1),
        "id": parsed.id,
        "model": parsed.model,
        "text": text,
        "usage": parsed.usage.model_dump() if parsed.usage else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-key", action="store_true", help="Check OpenRouter access")
    actions.add_argument("--evaluate", action="store_true", help="Run one billed Claude request")
    actions.add_argument("--dry-run", action="store_true", help="Print the request without calling")
    args = parser.parse_args(argv)
    try:
        settings = DemoSettings()
    except ValidationError:
        print(
            "Invalid configuration. Check OPENROUTER_API_KEY and OPENROUTER_LLM_MODEL.",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        print(json.dumps(build_request(settings.openrouter_llm_model), indent=2))
        return 0
    api_key = settings.openrouter_api_key.get_secret_value().strip()
    if not api_key:
        print("Set OPENROUTER_API_KEY in .env or your environment first.", file=sys.stderr)
        return 2
    try:
        with httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=90.0,
            follow_redirects=False,
        ) as client:
            result = run_request(
                client, evaluate=args.evaluate, model=settings.openrouter_llm_model
            )
    except httpx.HTTPStatusError as exc:
        guidance = {
            401: "Authentication failed. Check your OpenRouter key.",
            402: "Check your OpenRouter balance or key limit.",
            403: "Access denied for this model or account.",
            404: "Endpoint or model not found. Check OPENROUTER_LLM_MODEL.",
            429: "Rate limited. Wait and retry.",
            529: "Provider overloaded. Try again later.",
        }.get(exc.response.status_code, "OpenRouter request failed.")
        print(f"HTTP {exc.response.status_code}: {guidance}", file=sys.stderr)
        return 1
    except (httpx.TimeoutException, httpx.RequestError):
        print(
            "OpenRouter connection failed or timed out; no automatic retry was made.",
            file=sys.stderr,
        )
        return 1
    except (ValueError, ValidationError):
        print("Unexpected OpenRouter response; expected chat-completions JSON.", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2).replace(api_key, "[REDACTED]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
