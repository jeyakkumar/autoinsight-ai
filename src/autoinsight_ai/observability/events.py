"""Request telemetry schema and JSON logging utility; pipeline instrumentation is pending."""

import logging
from datetime import UTC, datetime

from pydantic import Field

from autoinsight_ai.config import PipelineMode
from autoinsight_ai.planning.schemas import Contract


class RequestMetrics(Contract):
    request_id: str
    question: str
    mode: PipelineMode
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    candidate_actions: int = Field(default=0, ge=0)
    clm_selected: int = Field(default=0, ge=0)
    analysis_executed: int = Field(default=0, ge=0)
    candidate_insights: int = Field(default=0, ge=0)
    jev_verified: int = Field(default=0, ge=0)
    insights_surface: int = Field(default=0, ge=0)
    llm_latency_ms: float | None = Field(default=None, ge=0)
    clm_latency_ms: float | None = Field(default=None, ge=0)
    jev_latency_ms: float | None = Field(default=None, ge=0)
    analysis_latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    selection_rate: float | None = Field(default=None, ge=0, le=1)
    verification_rate: float | None = Field(default=None, ge=0, le=1)


def log_metrics(metrics: RequestMetrics, logger: logging.Logger) -> None:
    """Emit one JSON event; the caller configures handlers and log destination."""
    logger.info(metrics.model_dump_json())
