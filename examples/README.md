# Jev API key check and evaluation

This example targets **Jev on OpenRouter**, using the project's existing
HTTPX dependency. It is independent of the unfinished analytics pipeline.

1. Get an API key from [OpenRouter](https://openrouter.ai/settings/keys).
2. Add these lines to your local, Git-ignored `.env` (or set environment variables):

   ```dotenv
   OPENROUTER_API_KEY=your_key_here
   JEV_MODEL=~typesafe/jev-latest
   ```

3. From the repository root, check authentication:

   ```sh
   uv run python examples/jev_api_demo.py --check-key
   ```

   Success prints `api_key_check_succeeded`, elapsed time, usage and remaining key limit.
   This calls `GET https://openrouter.ai/api/v1/key`; it does not prove Jev inference
   is available. `limit_remaining: null` means no per-key cap, not unlimited account
   credits. Running without flags also checks the key.

4. Run one small model evaluation (uses your provider's API credits):

   ```sh
   uv run python examples/jev_api_demo.py --evaluate
   ```

   The example sends synthetic vehicle evidence and two `noul` (yes/no probability)
   questions. Python calculates the price difference. Jev evaluates whether the price
   claim and an unsupported safety claim are backed by that evidence. The output
   includes probabilities, the returned model version, request ID, provider, token
   usage, cost in USD when supplied, and elapsed time.
   Actual probabilities vary; success means a valid API response, not proven accuracy.
   `noul` is a number from 0 to 1, not a Boolean or a separate confidence field.

To inspect the payload without a key or a network call:

```sh
uv run python examples/jev_api_demo.py --dry-run
```

The script reads the repository's `.env`; exported variables take precedence. It
does not print keys or raw error bodies, follow redirects, or automatically retry.
Exit codes: `0` success, `1` API/network/response failure, `2` missing/invalid configuration.
An evaluation timeout does not prove the provider did not process the request.

The model defaults to [`~typesafe/jev-latest`](https://openrouter.ai/~typesafe/jev-latest),
including the leading tilde. The endpoint is
`POST https://openrouter.ai/api/alpha/decisions`, using `state` and `questions`.
This is a decision-model request, not a chat-completions request with `messages`.
No separate TypeSafe API key is needed.

Provider contract verified against the official [OpenRouter Jev guide](https://openrouter.ai/docs/guides/community/jev)
and [key-check documentation](https://openrouter.ai/docs/api_reference/limits) on 2026-09-27.

## Claude Opus 5.5 through OpenRouter

The generative LLM example is separate from JEV:

```sh
uv run python examples/openrouter_llm_demo.py --check-key
uv run python examples/openrouter_llm_demo.py --evaluate
uv run python examples/openrouter_llm_demo.py --dry-run
```

It defaults to `anthropic/claude-opus-5.5` and calls
`POST https://openrouter.ai/api/v1/chat/completions` with `messages`, `temperature`
and `max_tokens`. The prompt contains synthetic vehicle evidence and asks Claude
to explain it without inventing numbers. This is the explanation stage of the
AutoInsight design; Python should calculate metrics first, and JEV should verify
claims before they are passed to Claude. The request uses OpenRouter's
OpenAI-compatible format, not JEV's `state`/`questions` format.

Set `OPENROUTER_LLM_MODEL` to another OpenRouter model if needed. `--evaluate` is
billed to your OpenRouter account; `--dry-run` makes no network request.
