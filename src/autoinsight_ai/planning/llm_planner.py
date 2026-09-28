"""LLM planning boundary. Implement a provider adapter returning validated JSON."""

from typing import Protocol

from pydantic import JsonValue

from autoinsight_ai.planning.schemas import AnalysisPlan


class Planner(Protocol):
    def plan(self, question: str, state: dict[str, JsonValue]) -> AnalysisPlan: ...
