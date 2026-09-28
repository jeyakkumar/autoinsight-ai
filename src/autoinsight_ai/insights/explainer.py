"""Final explanation boundary; pass only surfaced evidence and validate numeric claims."""

from typing import Protocol

from autoinsight_ai.planning.schemas import Insight


class Explainer(Protocol):
    def explain(self, insights: list[Insight], analysis_steps: list[str]) -> str: ...
