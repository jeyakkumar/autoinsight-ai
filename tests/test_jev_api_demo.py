"""Offline HTTP contract tests; no API keys or paid requests are used."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

spec = importlib.util.spec_from_file_location(
    "jev_api_demo", Path(__file__).resolve().parents[1] / "examples" / "jev_api_demo.py"
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_key_check_uses_authenticated_key_endpoint():
    def handler(request):
        assert request.method == "GET"
        assert str(request.url) == "https://openrouter.ai/api/v1/key"
        assert request.headers["Authorization"] == "Bearer test-only-key"
        return httpx.Response(
            200,
            json={
                "data": {
                    "label": "private-label",
                    "limit_remaining": None,
                    "usage": 0.25,
                    "is_free_tier": False,
                }
            },
        )

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-only-key"},
    ) as client:
        result = demo.run_demo(client, evaluate=False, model="~typesafe/jev-latest")
    assert result["status"] == "api_key_check_succeeded"
    assert result["data"]["limit_remaining"] is None
    assert "private-label" not in json.dumps(result)


@pytest.mark.parametrize("probability", [0.98, 1.5, True])
def test_evaluation_uses_decisions_api_and_validates_probability(probability):
    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
        body = json.loads(request.content)
        assert body["model"] == "~typesafe/jev-latest"
        assert "messages" not in body
        assert body["state"]["evidence"]["difference_pct"] == -17.9
        return httpx.Response(
            200,
            json={
                "id": "test-response",
                "model": "typesafe/jev-test-version",
                "provider": "TypeSafe",
                "answers": {
                    "price_claim_supported": {"type": "noul", "noul": probability},
                    "safety_claim_supported": {"type": "noul", "noul": 0.01},
                },
                "usage": {"input_tokens": 300, "output_tokens": 20, "cost": 0.00001},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        if probability == 0.98:
            result = demo.run_demo(client, evaluate=True, model="~typesafe/jev-latest")
            assert result["status"] == "evaluation_succeeded"
            assert result["usage"]["cost"] == 0.00001
        else:
            with pytest.raises(ValidationError):
                demo.run_demo(client, evaluate=True, model="~typesafe/jev-latest")


@pytest.mark.parametrize("status", [401, 402, 429, 503])
def test_http_failure_is_not_reported_as_success_or_retried(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "Test failure"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            demo.run_demo(client, evaluate=True, model="~typesafe/jev-latest")
    assert len(calls) == 1


def test_cli_dry_run_and_missing_key_make_no_requests(monkeypatch, capsys):
    settings = demo.DemoSettings(_env_file=None, openrouter_api_key="")
    monkeypatch.setattr(demo, "DemoSettings", lambda: settings)

    def unexpected_client(**kwargs):
        pytest.fail("No HTTP client should be created")

    monkeypatch.setattr(demo.httpx, "Client", unexpected_client)
    assert demo.main(["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["model"] == "~typesafe/jev-latest"
    assert demo.main([]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_cli_does_not_print_secret_from_provider_error(monkeypatch, capsys):
    secret = "test-secret-never-print"
    settings = demo.DemoSettings(_env_file=None, openrouter_api_key=secret)
    monkeypatch.setattr(demo, "DemoSettings", lambda: settings)

    def fail(*args, **kwargs):
        response = httpx.Response(
            401, json={"error": secret}, request=httpx.Request("GET", demo.KEY_URL)
        )
        response.raise_for_status()

    monkeypatch.setattr(demo, "run_demo", fail)
    assert demo.main([]) == 1
    captured = capsys.readouterr()
    assert "HTTP 401" in captured.err
    assert secret not in captured.out + captured.err
