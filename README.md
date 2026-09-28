# AutoInsight AI

A reproducible vehicle analytics project that separates deterministic calculations,
LLM planning and explanation, CLM action selection, and JEV evidence verification.

## Status

**Project scaffold.** The API health endpoint and dashboard shell are running, but
data ingestion, analytics, provider integrations, and end-to-end analysis remain
in progress. No mock output is presented as a real CLM or JEV decision.

## Overview

AutoInsight AI is designed to answer used-vehicle questions through a structured
pipeline:

- deterministic market and safety evidence
- LLM planning for allowable analytical actions
- CLM action selection
- JEV verification of claims before surfacing them

This keeps the system grounded in evidence instead of allowing unsupported claims
or model-generated conclusions to masquerade as verified results.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Run commands from
the repository root:

```sh
uv sync --extra dev
cp .env.example .env
uv run uvicorn autoinsight_ai.api.main:app --reload
```

In another terminal:

```sh
uv run streamlit run app/streamlit_app.py
```

- API docs: <http://127.0.0.1:8000/docs>
- Dashboard: <http://localhost:8501>

The `.python-version` selects Python 3.12; `uv` may download it if missing.
Alternatively, use a Python 3.12+ virtual environment and run:

```sh
pip install -e '.[dev]'
```

## Dashboard behavior

The dashboard includes a **Chat analysis** panel. Ask a question such as:

```text
Is the 2022 Ford Escape a good value, and are there recalls?
```

This panel exposes:

- CLM action selection
- deterministic peer evidence
- JEV verification decisions
- optional NHTSA source requests

Enable **Call live CLM/JEV** only when a local CLM server and
`OPENROUTER_API_KEY` are configured. Enable **Query NHTSA .gov source** to make
the vehicle-specific API request.

In the chat pipeline, CLM is responsible for selecting the allowed analytical
actions from the question and available data. It does not calculate prices or
write the answer. JEV receives the resulting evidence and independently marks
claims as supported before they are surfaced. When either provider is
unavailable, the trace labels the deterministic fallback and keeps unsupported
claims out of the response.

To load realistic listing data, download the public `vehicles_us_cleaned.csv`
snapshot into `data/raw/`:

```bash
uv run python scripts/download_real_dataset.py
```

The dashboard uses that file automatically on its next start. If a requested
make/model/year is absent, the chat reports that no exact price conclusion can be
made instead of substituting another vehicle. The upload control remains
available for a newer or private listing export.

## Project layout

```text
app/streamlit_app.py             Dashboard shell
src/autoinsight_ai/
  api/                          FastAPI entry point and health endpoint
  config.py                     Validated environment settings
  ingestion/                    Listing, EPA, NHTSA, and normalization interfaces
  profiling/                    Deterministic profiling interface
  planning/                     LLM planning interface and shared typed contracts
  decision/                     CLM and JEV provider interfaces
  analysis/                     Deterministic metric interfaces and value weights
  insights/                     Evidence generation, routing, and LLM explanation
  orchestration/                Four comparison modes and pipeline interface
  observability/                Structured request metrics and JSON logging
data/{raw,processed,metadata}/  Local datasets and source provenance
evaluation/                     Twenty draft golden cases and evaluation plan
tests/                          Contract, routing, mode, and API checks
docs/                           Implementation sequence and boundaries
logs/                           Ignored runtime decision logs
```

The requirements' suggested modules live under `src/autoinsight_ai/` so the
application installs as a single importable package without generic top-level
package names such as `analysis` or `config`.

## Data and providers

For local CLM exploration on Apple Silicon, see
[the isolated Mac experiment](experiments/clm_mac/README.md). It uses the original
Qwen3-8B encoder with an experimental PyTorch MPS backend and upstream CLM heads.

To test an OpenRouter API key and run a small JEV Python evaluation, see
[the standalone JEV example](examples/README.md). It supports an authenticated key
check, a synthetic vehicle-evidence evaluation, and an offline payload preview.

The chat source step is catalogued in `data/metadata/sources.json` and uses the
official NHTSA vehicle endpoint pattern for recalls. An empty response means no
records were returned by that endpoint; it is not a safety conclusion.

Begin with used-car listings, EPA fuel economy, and NHTSA recalls. Complaints and
safety have reserved modules for version 2. Raw data, generated data, and secrets
are ignored by Git. Dataset URLs and license-review status are recorded in
`data/metadata/sources.json`; no datasets have been downloaded or redistributed.
Inspect actual CSV columns during ingestion; do not assume the example schema.

LLM, CLM, and JEV are dependency-injected protocols. Their inputs and outputs are
**internal application contracts**, not claims about external model APIs. Verify
provider documentation before implementing adapters. The standalone JEV example uses
the verified OpenRouter Decisions API contract; pipeline adapters remain
unimplemented. Placeholder functions explicitly raise `NotImplementedError`.

## Configuration and verification

`.env.example` documents settings. Routing thresholds, minimum peer count, and
the maximum number of follow-up analyses are configurable. Value-index weights
have a separate validated configuration contract; their formula still needs
definition.

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The four comparison modes are `analytics_only`, `llm_analytics`, `llm_clm`, and
`llm_clm_jev`. They are configuration choices, not completed pipelines. The
dashboard displays this limitation explicitly.

Read [the requirements](requirements.md), [implementation plan](docs/implementation.md),
and [evaluation plan](evaluation/README.md) before extending the scaffold.

## Example prompts

### Single vehicle

- Is the 2024 Honda CR-V a good value?
- Is the 2025 Toyota RAV4 priced fairly?
- Is the 2022 Ford F-150 a good buy based on mileage and price?

Expected result: price percentile, mileage percentile, value index, and JEV
verification.

### Model-year comparison

- Compare Honda CR-V 2024 vs 2025
- Compare Toyota RAV4 2023 versus 2024
- What is the price and mileage difference between Ford F-150 2023 and 2024?

Expected result: side-by-side listing count, median price, median mileage, and
comparison verification.

### Ranking

- Show me the top selling cars
- What are the most popular SUVs in the dataset?
- Which vehicle has the highest listing volume?

Expected result: listing-frequency ranking. The answer clarifies that listing
count is not confirmed sales volume.

### Recall and source requests

- Are there recalls for the 2024 Honda CR-V?
- Check NHTSA safety information for the 2025 Toyota RAV4
- Does the 2023 Ford F-150 have complaint records?

Enable **Query NHTSA .gov source** for these requests.

### Typo and missing-year handling

- Is the 2021 Toyato RAV4 a good one?
- Is the 2021 Honda CRV good?
- Compare Honda CRV 2024 vs 2025

