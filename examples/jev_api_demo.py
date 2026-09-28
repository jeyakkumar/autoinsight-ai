"""Check an OpenRouter key or run a small Jev evaluation without the full pipeline.

Run from the repository root:
    uv run python examples/jev_api_demo.py
    uv run python examples/jev_api_demo.py --evaluate
    uv run python examples/jev_api_demo.py --dry-run

Reference: https://openrouter.ai/docs/guides/community/jev
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Literal

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

KEY_URL = "https://openrouter.ai/api/v1/key"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class DemoSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    openrouter_api_key: SecretStr = SecretStr("")
    jev_model: str = Field(default="~typesafe/jev-latest", min_length=1)


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1, strict=True)


class DemoAnswers(BaseModel):
    price_claim_supported: NoulAnswer
    safety_claim_supported: NoulAnswer


class Usage(BaseModel):
    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)
    cost: float | None = Field(default=None, ge=0)


class EvaluationResponse(BaseModel):
    id: str
    provider: str | None = None
    model: str = Field(min_length=1)
    answers: DemoAnswers
    usage: Usage


class KeyInfo(BaseModel):
    # Deliberately omit the key label, hash and other account metadata from output.
    limit_remaining: float | None
    usage: float = Field(ge=0)
    is_free_tier: bool


class KeyResponse(BaseModel):
    data: KeyInfo


def build_demo_request(model: str) -> dict:
    """Python calculates the difference; Jev judges two narrow claims about evidence."""
    listing_price, peer_median = 24800, 30200
    difference_pct = round((listing_price / peer_median - 1) * 100, 1)
    return {
        "model": model,
        "state": {
            "data_kind": "synthetic demonstration, not a real listing",
            "vehicle": "Example SUV 2022",
            "evidence": {
                "listing_price_usd": listing_price,
                "peer_median_price_usd": peer_median,
                "difference_pct": difference_pct,
                "peer_count": 143,
                "peer_definition": "Same make, model and model year",
                "recall_data_available": False,
                "safety_rating_available": False,
            },
            "price_claim": f"This listing is {abs(difference_pct)}% below its peer median price.",
            "safety_claim": "This vehicle has no safety concerns.",
        },
        "questions": {
            "price_claim_supported": {
                "type": "noul",
                "instructions": "Does the provided price evidence support the price_claim?",
            },
            "safety_claim_supported": {
                "type": "noul",
                "instructions": (
                    "Does the provided evidence support the safety_claim? "
                    "Missing recall data or safety ratings are not evidence of safety."
                ),
            },
        },
    }


def run_demo(client: httpx.Client, *, evaluate: bool, model: str) -> dict:
    """Make exactly one request; report transport/contract failures to the caller."""
    started = perf_counter()
    if evaluate:
        response = client.post(DECISIONS_URL, json=build_demo_request(model))
    else:
        response = client.get(KEY_URL)
    response.raise_for_status()
    data = response.json()
    schema = EvaluationResponse if evaluate else KeyResponse
    parsed = schema.model_validate(data)
    return {
        "status": "evaluation_succeeded" if evaluate else "api_key_check_succeeded",
        "elapsed_ms": round((perf_counter() - started) * 1000, 1),
        **parsed.model_dump(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-key", action="store_true", help="Check OpenRouter key (default)")
    actions.add_argument("--evaluate", action="store_true", help="Run one billed Jev evaluation")
    actions.add_argument(
        "--dry-run", action="store_true", help="Print demo input; make no API call"
    )
    args = parser.parse_args(argv)

    try:
        settings = DemoSettings()
    except ValidationError:
        print(
            "Invalid configuration. Check OPENROUTER_API_KEY and JEV_MODEL in .env.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print(json.dumps(build_demo_request(settings.jev_model), indent=2))
        return 0

    api_key = settings.openrouter_api_key.get_secret_value().strip()
    if not api_key:
        print("Set OPENROUTER_API_KEY in .env or your environment before running.", file=sys.stderr)
        return 2

    try:
        with httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=30.0,
            follow_redirects=False,
        ) as client:
            result = run_demo(client, evaluate=args.evaluate, model=settings.jev_model)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        guidance = {
            401: "Authentication failed. Check that your OpenRouter key is valid.",
            402: "Check your account balance or billing status.",
            403: "Access denied. Check your account permissions.",
            400: "Request rejected. Check the model and request format.",
            404: "Endpoint or model not found. Check JEV_MODEL.",
            422: "Request rejected. Check the model and request format.",
            429: "Rate limited. Wait before trying again.",
            529: "Service overloaded. Try again later.",
        }.get(status, "Request failed. Check provider status and account access.")
        print(f"HTTP {status}: {guidance}", file=sys.stderr)
        return 1
    except httpx.TimeoutException:
        print("Request timed out. No automatic retry was made.", file=sys.stderr)
        return 1
    except httpx.RequestError:
        print("Connection failed. Check your network and OpenRouter availability.", file=sys.stderr)
        return 1
    except ValueError:
        print("Unexpected API response; expected the documented Jev JSON format.", file=sys.stderr)
        return 1

    # Never print headers, credentials, or raw provider error bodies.
    print(json.dumps(result, indent=2).replace(api_key, "[REDACTED]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
