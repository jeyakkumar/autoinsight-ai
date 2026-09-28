"""Build candidate claims with traceable machine-readable evidence."""

from pydantic import JsonValue

from autoinsight_ai.planning.schemas import Insight


def generate_insights(results: dict[str, JsonValue]) -> list[Insight]:
    insights: list[Insight] = []
    for item in results.get("vehicles", []):
        if not isinstance(item, dict):
            continue
        vehicle_key = item.get("vehicle_key")
        if not isinstance(vehicle_key, str):
            continue
        if _is_number(item.get("price_difference_pct")) and item.get("price_peer_count", 0) >= 2:
            difference = float(item["price_difference_pct"])
            insights.append(
                Insight(
                    insight_id=f"{vehicle_key}:price",
                    vehicle_key=vehicle_key,
                    action_id="price_percentile",
                    claim=(
                        f"Listing price is {abs(difference):.1f}% "
                        f"{'below' if difference < 0 else 'above'} peer median."
                    ),
                    evidence={
                        "listing_price": item.get("price"),
                        "peer_median": item.get("peer_price_median"),
                        "difference_pct": round(difference, 2),
                        "peer_count": item.get("price_peer_count"),
                    },
                    sources=["used_car_listings"],
                )
            )
        if _is_number(item.get("value_index")):
            insights.append(
                Insight(
                    insight_id=f"{vehicle_key}:value",
                    vehicle_key=vehicle_key,
                    action_id="value_index",
                    claim="Deterministic value index is available for this listing.",
                    evidence={
                        "value_index": round(float(item["value_index"]), 4),
                        "components": item.get("value_index_components", {}),
                    },
                    sources=["used_car_listings"],
                )
            )
    return insights


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
