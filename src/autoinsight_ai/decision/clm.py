"""CLM selects analytical actions; it does not calculate metrics or write answers."""

from typing import Protocol

from pydantic import JsonValue

from autoinsight_ai.planning.schemas import CandidateAction, SelectionDecision


class Selector(Protocol):
    def select(
        self, state: dict[str, JsonValue], candidates: list[CandidateAction]
    ) -> list[SelectionDecision]: ...
