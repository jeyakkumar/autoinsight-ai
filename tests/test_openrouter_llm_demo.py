"""Offline tests for the OpenRouter Claude example."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

spec = importlib.util.spec_from_file_location(
    "openrouter_llm_demo",
    Path(__file__).resolve().parents[1] / "examples" / "openrouter_llm_demo.py",
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_request_uses_claude_chat_completions_shape():
    request = demo.build_request("anthropic/claude-opus-5.5")
    assert request["model"] == "anthropic/claude-opus-5.5"
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    assert "questions" not in request


def test_evaluation_parses_text_usage_and_cost():
    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == demo.CHAT_URL
        body = json.loads(request.content)
        assert body["model"] == "anthropic/claude-opus-5.5"
        assert body["max_tokens"] == 300
        return httpx.Response(
            200,
            json={
                "id": "chat-test",
                "model": "anthropic/claude-opus-5.5",
                "choices": [{"message": {"role": "assistant", "content": "Grounded explanation."}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cost": 0.01,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = demo.run_request(client, evaluate=True, model="anthropic/claude-opus-5.5")
    assert result["text"] == "Grounded explanation."
    assert result["usage"]["cost"] == 0.01


def test_key_check_does_not_return_sensitive_label():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "data": {
                    "label": "secret-label",
                    "limit_remaining": None,
                    "usage": 0,
                    "is_free_tier": False,
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = demo.run_request(client, evaluate=False, model="unused")
    assert result["status"] == "api_key_check_succeeded"
    assert "secret-label" not in json.dumps(result)


def test_empty_assistant_content_is_rejected():
    with pytest.raises(ValidationError):
        demo.CompletionResponse.model_validate(
            {"id": "x", "model": "m", "choices": [], "usage": None}
        )
