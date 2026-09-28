"""Run with uvicorn autoinsight_ai.api.main:app --reload."""

from fastapi import FastAPI

from autoinsight_ai import __version__

app = FastAPI(title="AutoInsight AI", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "pipeline_status": "not_implemented"}
