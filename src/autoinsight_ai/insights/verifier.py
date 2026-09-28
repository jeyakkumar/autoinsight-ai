"""Route validated verification decisions using application configuration."""

from autoinsight_ai.config import Settings
from autoinsight_ai.planning.schemas import Route, VerificationDecision


def route_insight(decision: VerificationDecision, settings: Settings) -> Route:
    if decision.request_more_analysis:
        return Route.ADDITIONAL_ANALYSIS
    if (
        decision.evidence_sufficient
        and decision.surface_to_user
        and decision.surface_probability >= settings.surface_threshold
    ):
        return Route.SURFACE
    if decision.surface_probability >= settings.investigate_threshold:
        return Route.ADDITIONAL_ANALYSIS
    return Route.SUPPRESS_OR_REVIEW
