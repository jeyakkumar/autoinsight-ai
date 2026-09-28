import pytest

from autoinsight_ai.config import PipelineMode, Settings
from autoinsight_ai.orchestration.pipeline import AnalysisPipeline
from autoinsight_ai.planning.schemas import AnalysisRequest


@pytest.mark.parametrize(
    ("mode", "llm", "clm", "jev"),
    [
        (PipelineMode.ANALYTICS_ONLY, False, False, False),
        (PipelineMode.LLM_ANALYTICS, True, False, False),
        (PipelineMode.LLM_CLM, True, True, False),
        (PipelineMode.LLM_CLM_JEV, True, True, True),
    ],
)
def test_comparison_modes(mode, llm, clm, jev):
    assert (mode.uses_llm, mode.uses_clm, mode.uses_jev) == (llm, clm, jev)


def test_scaffold_does_not_produce_fabricated_results():
    with pytest.raises(NotImplementedError):
        AnalysisPipeline(Settings(_env_file=None)).run(AnalysisRequest(question="Find value"))
