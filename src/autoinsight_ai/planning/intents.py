"""Small typed intent boundary between chat language and analysis actions."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatIntent:
    kind: str
    years: tuple[int, ...] = ()
    wants_sources: bool = False


def parse_intent(question: str) -> ChatIntent:
    text = question.lower()
    years = tuple(sorted({int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", text)}))
    if any(
        phrase in text for phrase in ("top selling", "best selling", "most popular", "popular cars")
    ):
        kind = "ranking"
    elif re.search(r"\b(compare|versus|vs\.?|difference)\b", text) and len(years) >= 2:
        kind = "comparison"
    else:
        kind = "vehicle"
    wants_sources = any(
        word in text for word in ("recall", "complaint", "safety", "nhtsa", ".gov", "source")
    )
    return ChatIntent(kind=kind, years=years, wants_sources=wants_sources)
