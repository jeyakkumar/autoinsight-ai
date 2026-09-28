# CLM on Apple Silicon (experimental)

This standalone experiment ranks AutoInsight analysis candidates locally. It keeps
the original **Qwen/Qwen3-8B** encoder and released **CLM-v0.1-8B** heads, while
replacing the reference vLLM embedding server with Transformers on PyTorch MPS.
The small projection heads run on CPU using the upstream CLM implementation.
It does not call OpenRouter or require an API key.

The inspected development Mac is an M4 Pro with 64 GiB unified memory. That is
promising for a short-context unquantized trial, but available runtime memory and
actual model execution determine whether it fits. Initial free disk space was
about 32 GiB. The selected external volume `/Volumes/mmext` has about 1.7 TiB free.
The first evaluation downloads roughly 16 GB of encoder weights plus the heads
and tokenizer. Leave additional space for dependencies and temporary files.

## Run from the repository root

This environment is independent of the main AutoInsight `.venv` and lockfile:

```sh
uv sync --project experiments/clm_mac
uv run --project experiments/clm_mac python experiments/clm_mac/demo.py --check
```

`--check` loads no models and makes no inference call. It reports Apple GPU
availability, installed versions, cache location and free disk space.

When ready to download the full weights and try ranking:

```sh
uv run --project experiments/clm_mac python experiments/clm_mac/demo.py --evaluate
```

By your selection, weights default to `/Volumes/mmext/autoinsight-models/clm/`,
outside the repository. Keep the `mmext` drive mounted. You can override the path:

```sh
uv run --project experiments/clm_mac python experiments/clm_mac/demo.py \
  --evaluate --cache-dir /Volumes/mmext/autoinsight-models/clm
```

For subsequent offline runs, use the same cache directory and add `--local-only`.
Output includes ranked actions, relative probabilities, separate loading and
evaluation timings, token count and encoder revision. It reports
`reference_backend_parity: not_verified` until compared with reference embeddings.

## Why a separate environment

The current `contrastive-lm` package declares vLLM as a dependency. This project's
uv override restricts that dependency to Linux: the Mac demo injects its own
embedder and calls only the CLM engine/heads. This is a deliberate local adaptation,
not an upstream-supported Mac installation. Do not use this environment for the
reference `vllm serve` instructions. Versions are recorded in its own `uv.lock`.

## Correctness and limits

- Use Qwen3-8B, not Qwen3-Embedding, a smaller Qwen model, or a generic embedding API.
- Extract the last non-padding token from the final hidden state and L2-normalize
  in float32 before applying the original heads. The encoder uses bfloat16 on MPS.
- Inputs are plain text with no chat template or additional special tokens. CLM's
  upstream engine formats state/questions and candidates before encoding them.
- The initial token cap is 512 per text, processing one text at a time. Long or
  empty inputs fail explicitly. `--max-tokens` permits up to 2048 for exploration.
- Embeddings are cached only for the lifetime of the demo. There is no HTTP server.
- CLM's candidate probabilities are relative to the supplied alternatives. A high
  probability does not establish evidence quality or absolute task correctness.
- MPS execution, successful loading and sensible rankings do not prove numerical
  parity with vLLM. Compare identical token IDs, normalized embeddings and rankings
  against the reference backend before treating evaluations as interchangeable.

## Offline backend tests

```sh
uv run --project experiments/clm_mac python -m pytest -c experiments/clm_mac/pyproject.toml \
  experiments/clm_mac/tests
```

These check pooling, empty-input rejection and a **tiny randomly initialized**
Qwen3 forward pass on the Apple GPU. They do not download model weights or establish
CLM ranking quality. The GPU test skips if MPS is unavailable; check the test summary.
Run from a normal terminal if a restricted execution environment hides the GPU.

## Observed local run

The full released model ran successfully on the M4 Pro/64 GiB development machine.
All 3 backend tests also passed with Apple GPU access. Runtime versions:
PyTorch 2.14.0, Transformers 4.57.6, contrastive-lm 0.1.0.

- Initial download plus loading: 990.61 seconds; subsequent cached runs avoid the download.
- Ranking evaluation: 2447.9 ms for 101 encoder tokens across the state and four candidates.
- Ranking: peer price 0.8903, paint colors 0.0797, recalls 0.0254, mileage 0.0045.
- Encoder revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- Head revision: `e939398d4556fcd9400c76fa8c5a513202f42b0a`.

The price action ranked first, but the irrelevant paint-color action ranked ahead
of recalls and mileage. This demonstrates successful local execution, not validated
action-selection quality. Evaluate prompt wording and labeled domain examples,
and compare reference-backend embeddings before drawing conclusions about CLM quality.
The earlier JEV request used different inputs/questions, so these timings are not
an apples-to-apples benchmark.

The initial output is saved locally in `evaluation/results/clm_mac_first_run.json`
(ignored by Git). Edit `STATE` and `CANDIDATES` in `demo.py` to try another question.
For a fully cached run:

```sh
uv run --project experiments/clm_mac python experiments/clm_mac/demo.py --evaluate --local-only
```

## Run CLM for the chat UI

The local HTTP adapter loads the same cached model and serves the ranking
contract expected by the Streamlit chat. Start it in a separate terminal:

```sh
uv run --project experiments/clm_mac python experiments/clm_mac/server.py \
  --local-only --cache-dir /Volumes/mmext/autoinsight-models/clm
```

When `http://localhost:8700/health` reports `model_loaded: true`, enable
**Call live CLM/JEV** in the dashboard. The chat trace will then show
`status: live` for CLM instead of `heuristic_fallback`.

## References

- [CLM model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
- [Upstream engine and embedder injection](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/engine.py)
- [Reference embedding normalization](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/embedder.py)
- [PyTorch Apple GPU backend](https://docs.pytorch.org/docs/stable/notes/mps.html)

Reviewed on 2026-09-27. The full 8B evaluation and reference-backend comparison are
separate from the small offline backend checks.
