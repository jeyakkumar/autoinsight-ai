import pytest
from pydantic import ValidationError

from autoinsight_ai.planning.schemas import AnalysisPlan


def test_planner_cannot_request_arbitrary_executable_action():
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate(
            {
                "question": "Find good value",
                "candidate_actions": [
                    {"id": "execute_python", "description": "Run code", "required_data": ["price"]}
                ],
            }
        )
