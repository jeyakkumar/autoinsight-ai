import pytest
from pydantic import ValidationError

from autoinsight_ai.config import Settings
from autoinsight_ai.insights.verifier import route_insight
from autoinsight_ai.planning.schemas import Route, VerificationDecision


@pytest.mark.parametrize(
    ("probability", "sufficient", "surface", "more", "expected"),
    [
        (0.85, True, True, False, Route.SURFACE),
        (0.60, True, True, False, Route.ADDITIONAL_ANALYSIS),
        (0.59, True, True, False, Route.SUPPRESS_OR_REVIEW),
        (0.99, False, True, False, Route.ADDITIONAL_ANALYSIS),
        (0.99, True, False, False, Route.ADDITIONAL_ANALYSIS),
        (0.99, True, True, True, Route.ADDITIONAL_ANALYSIS),
    ],
)
def test_routing_requires_evidence_and_respects_boundaries(
    probability, sufficient, surface, more, expected
):
    decision = VerificationDecision(
        insight_id="I-1",
        evidence_sufficient=sufficient,
        surface_to_user=surface,
        request_more_analysis=more,
        surface_probability=probability,
        reason="Synthetic routing test",
    )
    settings = Settings(_env_file=None, surface_threshold=0.85, investigate_threshold=0.60)
    assert route_insight(decision, settings) == expected


def test_rejects_inverted_thresholds():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, investigate_threshold=0.9, surface_threshold=0.8)


def test_rejects_invalid_probability():
    with pytest.raises(ValidationError):
        VerificationDecision(
            insight_id="I-1",
            evidence_sufficient=True,
            surface_to_user=True,
            request_more_analysis=False,
            surface_probability=1.1,
            reason="Invalid probability",
        )
