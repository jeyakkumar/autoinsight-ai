"""Internal application contracts, not external CLM/JEV API specifications."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from autoinsight_ai.config import PipelineMode


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActionId(StrEnum):
    PRICE_PERCENTILE = "price_percentile"
    PRICE_PER_MILE = "price_per_mile"
    DEPRECIATION = "depreciation"
    MILEAGE_PERCENTILE = "mileage_percentile"
    FUEL_ECONOMY = "fuel_economy"
    RECALL_ANALYSIS = "recall_analysis"
    COMPLAINT_ANALYSIS = "complaint_analysis"
    SAFETY_RATING = "safety_rating"
    PRICE_ANOMALY = "price_anomaly"
    VALUE_INDEX = "value_index"


class CandidateAction(Contract):
    id: ActionId
    description: str = Field(min_length=1)
    required_data: list[str] = Field(min_length=1)


class AnalysisPlan(Contract):
    question: str = Field(min_length=1)
    candidate_actions: list[CandidateAction] = Field(min_length=1)


class SelectionDecision(Contract):
    action_id: ActionId
    # Score scale is provider-defined until the CLM contract is verified.
    score: float = Field(allow_inf_nan=False)
    selected: bool
    reason: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Insight(Contract):
    insight_id: str = Field(min_length=1)
    vehicle_key: str = Field(min_length=1)
    action_id: ActionId
    claim: str = Field(min_length=1)
    evidence: dict[str, JsonValue] = Field(min_length=1)
    sources: list[str] = Field(min_length=1)


class VerificationDecision(Contract):
    insight_id: str = Field(min_length=1)
    evidence_sufficient: bool
    surface_to_user: bool
    request_more_analysis: bool
    surface_probability: float = Field(ge=0, le=1)
    investigate_probability: float | None = Field(default=None, ge=0, le=1)
    reason: str


class Route(StrEnum):
    SURFACE = "surface"
    ADDITIONAL_ANALYSIS = "additional_analysis"
    SUPPRESS_OR_REVIEW = "suppress_or_review"


class AnalysisRequest(Contract):
    question: str = Field(min_length=1)
    mode: PipelineMode = PipelineMode.LLM_CLM_JEV


class PipelineResult(Contract):
    request_id: str
    mode: PipelineMode
    plan: AnalysisPlan
    selections: list[SelectionDecision]
    candidate_insights: list[Insight]
    verifications: list[VerificationDecision]
    surfaced_insights: list[Insight]
    analysis_steps: list[str]
    explanation: str | None = None
