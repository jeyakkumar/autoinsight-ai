"""Pipeline wiring point. No fake analytical or model decisions are returned."""

from autoinsight_ai.config import Settings
from autoinsight_ai.decision.clm import Selector
from autoinsight_ai.decision.jev import Verifier
from autoinsight_ai.insights.explainer import Explainer
from autoinsight_ai.planning.llm_planner import Planner
from autoinsight_ai.planning.schemas import AnalysisRequest, PipelineResult


class AnalysisPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        planner: Planner | None = None,
        selector: Selector | None = None,
        verifier: Verifier | None = None,
        explainer: Explainer | None = None,
    ) -> None:
        self.settings = settings
        self.planner = planner
        self.selector = selector
        self.verifier = verifier
        self.explainer = explainer

    def run(self, request: AnalysisRequest) -> PipelineResult:
        raise NotImplementedError(
            "Wire ingestion, planning, selection, deterministic analysis, evidence verification, "
            "bounded follow-up analysis, telemetry and explanation before running the pipeline"
        )
