"""JEV independently evaluates insight evidence. Verify its API before integration."""

from typing import Protocol

from pydantic import JsonValue

from autoinsight_ai.planning.schemas import Insight, VerificationDecision


class Verifier(Protocol):
    def verify(self, state: dict[str, JsonValue], insight: Insight) -> VerificationDecision: ...
