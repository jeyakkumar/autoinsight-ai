"""Small, inspectable chat orchestrator for the Streamlit demonstration."""

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from autoinsight_ai.analysis.engine import execute_actions
from autoinsight_ai.config import Settings
from autoinsight_ai.ingestion.nhtsa import lookup_vehicle
from autoinsight_ai.planning.actions import action_description
from autoinsight_ai.planning.intents import parse_intent
from autoinsight_ai.planning.schemas import ActionId


def load_source_catalog(path: Path | None = None) -> dict[str, Any]:
    """Read the repository's source registry used to authorize source requests."""
    catalog_path = path or Path(__file__).resolve().parents[3] / "data/metadata/sources.json"
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def _display_model(value: Any) -> str:
    text = str(value)
    return (
        text.upper()
        if any(character.isdigit() for character in text) or "-" in text
        else text.title()
    )


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _vehicle_from_question(question: str, frame: pd.DataFrame) -> dict[str, Any]:
    year_match = re.search(r"\b(19|20)\d{2}\b", question)
    requested_year = int(year_match.group(0)) if year_match else None
    question_lower = question.lower()
    question_words = re.findall(r"[a-z0-9-]+", question_lower)
    for _, candidate in frame[["make", "model"]].drop_duplicates().iterrows():
        make = str(candidate.make).lower()
        model = str(candidate.model).lower()
        make_match = (
            make in question_lower
            or max(
                (SequenceMatcher(None, make, word).ratio() for word in question_words),
                default=0,
            )
            >= 0.60
        )
        if make_match and _normalized_name(model) in _normalized_name(question_lower):
            return {
                "model_year": requested_year or int(frame.iloc[0].model_year),
                "make": str(candidate.make),
                "model": str(candidate.model),
            }
    match = re.search(r"(\d{4})\s+([A-Za-z-]+)\s+([A-Za-z0-9-]+)", question)
    if match:
        return {"model_year": int(match.group(1)), "make": match.group(2), "model": match.group(3)}
    row = frame.iloc[0]
    return {"model_year": int(row.model_year), "make": str(row.make), "model": str(row.model)}


def _comparison_from_question(question: str, frame: pd.DataFrame) -> dict[str, Any] | None:
    if not re.search(r"\b(compare|versus|vs\.?|difference)\b", question.lower()):
        return None
    years = sorted({int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", question)})
    if len(years) < 2:
        return None
    question_normalized = _normalized_name(question)
    question_words = re.findall(r"[a-z0-9-]+", question.lower())
    for _, candidate in frame[["make", "model"]].drop_duplicates().iterrows():
        make = str(candidate.make).lower()
        model = str(candidate.model).lower()
        make_match = (
            make in question.lower()
            or max(
                (SequenceMatcher(None, make, word).ratio() for word in question_words),
                default=0,
            )
            >= 0.60
        )
        if make_match and _normalized_name(model) in question_normalized:
            return {"make": str(candidate.make), "model": str(candidate.model), "years": years}
    return None


def _select_actions(question: str) -> tuple[list[str], dict[str, Any]]:
    text = question.lower()
    intent = parse_intent(question)
    actions = [
        ActionId.PRICE_PERCENTILE.value,
        ActionId.MILEAGE_PERCENTILE.value,
        ActionId.VALUE_INDEX.value,
    ]
    if "fuel" in text or "mpg" in text or "economy" in text:
        actions.append(ActionId.FUEL_ECONOMY.value)
    if intent.wants_sources:
        actions.append(ActionId.RECALL_ANALYSIS.value)
    if "complaint" in text:
        actions.append(ActionId.COMPLAINT_ANALYSIS.value)
    if "safety" in text or "rating" in text:
        actions.append(ActionId.SAFETY_RATING.value)
    return list(dict.fromkeys(actions)), {
        "status": "heuristic_fallback",
        "reason": "live CLM was not requested or is unavailable",
        "intent": intent.kind,
    }


def _live_clm(question: str, candidates: list[str], base_url: str) -> dict[str, Any]:
    try:
        candidate_prompts = [f"{action}: {action_description(action)}" for action in candidates]
        response = httpx.post(
            f"{base_url.rstrip('/')}/v1/rank",
            json={"model": "clm-latest", "question": question, "candidates": candidate_prompts},
            timeout=8,
        )
        response.raise_for_status()
        body = response.json()
        ranked = body.get("ranked", body.get("selections", []))
        selected = []
        for item in ranked:
            if isinstance(item, dict):
                value = item.get("id", item.get("candidate"))
            else:
                value = item
            if isinstance(value, str):
                selected.append(value)
        selected_ids = []
        for value in selected:
            selected_ids.extend(
                action
                for action, prompt in zip(candidates, candidate_prompts, strict=True)
                if value == action or value == prompt or value.startswith(f"{action}:")
            )
        return {"status": "live", "response": body, "selected": selected_ids}
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        return {
            "status": "live_unavailable",
            "endpoint": base_url,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _live_jev(evidence: dict[str, Any]) -> dict[str, Any] | None:
    settings = Settings()
    key = os.getenv("OPENROUTER_API_KEY") or settings.openrouter_api_key
    if not key:
        return {"status": "live_unavailable", "reason": "OPENROUTER_API_KEY is not configured"}
    try:
        response = httpx.post(
            "https://openrouter.ai/api/alpha/decisions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("JEV_MODEL") or settings.jev_model,
                "state": evidence,
                "questions": {
                    "price_claim_supported": {
                        "type": "noul",
                        "instructions": "Does the supplied peer evidence support the price claim?",
                    },
                    "value_claim_supported": {
                        "type": "noul",
                        "instructions": "Does the supplied evidence support the value-index claim?",
                    },
                    "safety_claim_supported": {
                        "type": "noul",
                        "instructions": (
                            "Does the supplied evidence support a safety claim? "
                            "Missing recalls or ratings are not evidence of safety."
                        ),
                    },
                    "ranking_claim_supported": {
                        "type": "noul",
                        "instructions": (
                            "Does the supplied dataset evidence support the ranking claim?"
                        ),
                    },
                },
            },
            timeout=15,
        )
        response.raise_for_status()
        return {"status": "live", "response": response.json()}
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        return {"status": "live_unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def _fallback_jev(evidence: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    price_supported = evidence.get("price_peer_count", 0) >= 2
    value_supported = isinstance(evidence.get("value_index"), (int, float))
    safety_supported = bool(source and source.get("status") == "ok")
    return {
        "status": "demo_fallback",
        "decisions": {
            "price_claim_supported": price_supported,
            "value_claim_supported": value_supported,
            "safety_claim_supported": safety_supported,
        },
        "surface": price_supported or value_supported,
        "reason": (
            "Only claims with peer evidence are eligible for the answer; "
            "safety requires a source response."
        ),
    }


def _normalize_jev(raw: dict[str, Any]) -> dict[str, Any]:
    answers = raw.get("response", {}).get("answers", {})
    decisions = {
        name: float(value.get("noul", 0)) >= 0.5
        for name, value in answers.items()
        if isinstance(value, dict) and "noul" in value
    }
    return {
        "status": "live",
        "decisions": decisions,
        "surface": decisions.get("price_claim_supported", False),
        "raw": raw,
    }


def run_chat(
    question: str,
    vehicles: pd.DataFrame,
    *,
    live_models: bool = False,
    live_sources: bool = False,
    clm_url: str = "http://localhost:8700",
) -> dict[str, Any]:
    """Execute the demo flow and return every stage for display in the UI."""
    intent = parse_intent(question)
    candidates, clm = _select_actions(question)
    if live_models:
        live = _live_clm(question, candidates, clm_url)
        if live.get("status") == "live" and live.get("selected"):
            selected = [
                item for item in live["selected"] if item in {action.value for action in ActionId}
            ]
            if selected:
                candidates = selected
                clm = live
            else:
                clm = {
                    "status": "live_invalid_response",
                    "reason": "CLM returned no recognized ActionId values",
                    "response": live.get("response"),
                }
        elif live.get("status") == "live_unavailable":
            clm = live
    executable = [
        item
        for item in candidates
        if item
        in {
            a.value
            for a in (
                ActionId.PRICE_PERCENTILE,
                ActionId.PRICE_PER_MILE,
                ActionId.MILEAGE_PERCENTILE,
                ActionId.FUEL_ECONOMY,
                ActionId.VALUE_INDEX,
            )
        }
    ]
    result = execute_actions(vehicles, executable)
    if intent.kind == "ranking":
        ranking = (
            result.groupby(["make", "model", "model_year"], dropna=False)
            .agg(
                listing_count=("vehicle_key", "size"),
                median_price=("price", "median"),
                median_mileage=("mileage", "median"),
            )
            .reset_index()
            .sort_values(["listing_count", "median_price"], ascending=[False, True])
            .head(5)
        )
        top = ranking.iloc[0]
        top_make = str(top.make).title()
        top_model = str(top.model).title()
        ranking_evidence = {
            "claim": "most represented vehicle by listing frequency",
            "vehicle": f"{int(top.model_year)} {top_make} {top_model}",
            "listing_count": int(top.listing_count),
            "dataset_rows": int(len(result)),
        }
        if live_models:
            ranking_live = _live_jev(ranking_evidence)
            ranking_jev = (
                _normalize_jev(ranking_live)
                if ranking_live.get("status") == "live"
                else ranking_live
            )
        else:
            ranking_jev = {
                "status": "demo_fallback",
                "decisions": {"ranking_claim_supported": True},
                "surface": True,
                "reason": "The ranking is computed directly from the loaded listing rows.",
            }
        if not ranking_jev.get("decisions", {}).get("ranking_claim_supported", True):
            answer = (
                "I found a candidate listing ranking, but JEV could not verify the ranking claim."
            )
            return {
                "question": question,
                "answer": answer,
                "clm": clm,
                "selected_actions": candidates,
                "analysis": {"ranking": ranking.to_dict(orient="records")},
                "jev": ranking_jev,
                "source": None,
            }
        answer = (
            "Based on listing frequency in the loaded dataset, the most represented "
            f"vehicle is the {int(top.model_year)} {top_make} {top_model} "
            f"with {int(top.listing_count):,} listings. "
            "This measures market listings, not confirmed sales volume."
        )
        return {
            "question": question,
            "answer": answer,
            "clm": clm,
            "selected_actions": candidates,
            "analysis": {"ranking": ranking.to_dict(orient="records")},
            "jev": ranking_jev,
            "source": None,
        }
    comparison = _comparison_from_question(question, vehicles)
    if comparison:
        same_model = result[
            result["make"].map(_normalized_name).eq(_normalized_name(comparison["make"]))
            & result["model"].map(_normalized_name).eq(_normalized_name(comparison["model"]))
            & result["model_year"].isin(comparison["years"])
        ]
        comparison_rows = (
            same_model.groupby("model_year")
            .agg(
                listing_count=("vehicle_key", "size"),
                median_price=("price", "median"),
                median_mileage=("mileage", "median"),
                median_value_index=("value_index", "median"),
            )
            .reset_index()
        )
        available_years = set(int(year) for year in comparison_rows["model_year"])
        missing_years = [year for year in comparison["years"] if year not in available_years]
        comparison_evidence = {
            "make": comparison["make"],
            "model": comparison["model"],
            "requested_years": comparison["years"],
            "missing_years": missing_years,
            "rows": comparison_rows.to_dict(orient="records"),
        }
        if len(comparison_rows) < 2:
            answer = (
                f"I could not compare both requested {comparison['make']} "
                f"{_display_model(comparison['model'])} model years. "
                f"Available years in this dataset: {sorted(available_years) or 'none'}."
            )
        else:
            first, second = comparison_rows.iloc[0], comparison_rows.iloc[1]
            price_change = float(second.median_price - first.median_price)
            answer = (
                f"For the {comparison['make']} {_display_model(comparison['model'])}, "
                f"{int(first.model_year)} has a median price of ${first.median_price:,.0f} "
                f"across {int(first.listing_count)} listings, while {int(second.model_year)} "
                f"is ${abs(price_change):,.0f} {'higher' if price_change >= 0 else 'lower'} "
                f"at ${second.median_price:,.0f} across {int(second.listing_count)} listings."
            )
        if live_models:
            comparison_live = _live_jev(comparison_evidence)
            comparison_jev = (
                _normalize_jev(comparison_live)
                if comparison_live.get("status") == "live"
                else comparison_live
            )
        else:
            comparison_jev = {
                "status": "demo_fallback",
                "decisions": {"comparison_claim_supported": len(comparison_rows) >= 2},
                "surface": len(comparison_rows) >= 2,
            }
        return {
            "question": question,
            "answer": answer,
            "clm": clm,
            "selected_actions": candidates,
            "analysis": comparison_evidence,
            "jev": comparison_jev,
            "source": None,
        }
    vehicle = _vehicle_from_question(question, vehicles)
    same_model = result[
        result["make"].map(_normalized_name).eq(_normalized_name(vehicle["make"]))
        & result["model"].map(_normalized_name).eq(_normalized_name(vehicle["model"]))
    ]
    row = same_model[same_model["model_year"] == vehicle["model_year"]].head(1)
    matched = not row.empty
    evidence = row.iloc[0].to_dict() if matched else {}
    source = None
    if live_sources and ActionId.RECALL_ANALYSIS.value in candidates:
        source = lookup_vehicle(vehicle["make"], vehicle["model"], vehicle["model_year"])
        source["catalog_entry"] = next(
            (item for item in load_source_catalog()["sources"] if item["id"] == "nhtsa_recalls"),
            None,
        )
    if live_models:
        live_jev = _live_jev({"question": question, "evidence": evidence, "source": source})
        jev = _normalize_jev(live_jev) if live_jev.get("status") == "live" else live_jev
    else:
        jev = _fallback_jev(evidence, source)
    value = evidence.get("value_index")
    if not matched:
        if not same_model.empty:
            nearest_year = int(
                same_model.iloc[
                    (same_model["model_year"] - vehicle["model_year"]).abs().argsort()[:1]
                ]["model_year"].iloc[0]
            )
            nearest = same_model[same_model["model_year"] == nearest_year]
            evidence = {
                "match_status": "nearby_year",
                "requested_model_year": vehicle["model_year"],
                "nearest_available_year": nearest_year,
                "available_years": sorted(
                    int(year) for year in same_model["model_year"].dropna().unique()
                ),
                "listing_count": int(len(nearest)),
                "median_price": float(nearest["price"].median()),
                "median_mileage": float(nearest["mileage"].median()),
            }
            display_make = str(vehicle["make"]).title()
            display_model = _display_model(vehicle["model"])
            answer = (
                f"I couldn't find a {vehicle['model_year']} {display_make} "
                f"{display_model} listing. "
                f"Using the nearest available {nearest_year} listings for this model, "
                f"the median price is ${evidence['median_price']:,.0f} across "
                f"{evidence['listing_count']} listings. This is directional evidence, "
                "not an exact-year conclusion."
            )
            return {
                "question": question,
                "answer": answer,
                "clm": clm,
                "selected_actions": candidates,
                "analysis": evidence,
                "jev": {"status": "not_run", "reason": "exact model year unavailable"},
                "source": source,
            }
        display_make = str(vehicle["make"]).title()
        display_model = _display_model(vehicle["model"])
        answer = (
            f"I couldn't find an exact {vehicle['model_year']} {display_make} "
            f"{display_model} listing in this dataset. No price conclusion was made. "
            "Load a broader listing file or upload a CSV containing that vehicle."
        )
        return {
            "question": question,
            "answer": answer,
            "clm": clm,
            "selected_actions": candidates,
            "analysis": evidence,
            "jev": {"status": "not_run", "reason": "no exact vehicle row"},
            "source": source,
        }
    verified_price = jev.get("decisions", {}).get(
        "price_claim_supported", jev.get("surface", False)
    )
    if not verified_price:
        answer = (
            f"I found a {vehicle['model_year']} {str(vehicle['make']).title()} "
            f"{_display_model(vehicle['model'])} listing, but JEV could not verify "
            "the price claim from the available peer evidence."
        )
        return {
            "question": question,
            "answer": answer,
            "clm": clm,
            "selected_actions": candidates,
            "analysis": evidence,
            "jev": jev,
            "source": source,
        }
    answer = (
        f"{vehicle['model_year']} {str(vehicle['make']).title()} "
        f"{_display_model(vehicle['model'])} "
        f"is priced at ${evidence.get('price', 0):,.0f}."
    )
    if evidence.get("price_difference_pct") is not None:
        direction = "below" if evidence["price_difference_pct"] < 0 else "above"
        answer += (
            f" That is {abs(evidence['price_difference_pct']):.1f}% "
            f"{direction} its filtered peer median."
        )
    if value is not None:
        answer += f" Its deterministic value index is {value:.2f}."
    if source:
        answer += (
            f" NHTSA returned {source.get('recall_count', 0)} recall record(s); "
            "an empty result is not a safety rating."
        )
    return {
        "question": question,
        "answer": answer,
        "clm": clm,
        "selected_actions": candidates,
        "analysis": evidence,
        "jev": jev,
        "source": source,
    }
