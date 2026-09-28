"""Environment configuration; provider-specific settings are intentionally deferred."""

from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineMode(StrEnum):
    ANALYTICS_ONLY = "analytics_only"
    LLM_ANALYTICS = "llm_analytics"
    LLM_CLM = "llm_clm"
    LLM_CLM_JEV = "llm_clm_jev"

    @property
    def uses_llm(self) -> bool:
        return self != self.ANALYTICS_ONLY

    @property
    def uses_clm(self) -> bool:
        return self in {self.LLM_CLM, self.LLM_CLM_JEV}

    @property
    def uses_jev(self) -> bool:
        return self == self.LLM_CLM_JEV


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUTOINSIGHT_", env_file=".env", extra="ignore")

    pipeline_mode: PipelineMode = PipelineMode.LLM_CLM_JEV
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
    surface_threshold: float = Field(default=0.85, ge=0, le=1)
    investigate_threshold: float = Field(default=0.60, ge=0, le=1)
    min_peer_count: int = Field(default=10, ge=2)
    max_additional_analysis: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def check_thresholds(self) -> "Settings":
        if self.investigate_threshold >= self.surface_threshold:
            raise ValueError("investigate_threshold must be less than surface_threshold")
        return self
