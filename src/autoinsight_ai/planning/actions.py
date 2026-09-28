"""Allow-listed actions and the descriptions sent to CLM."""

from autoinsight_ai.planning.schemas import ActionId

ACTION_DESCRIPTIONS: dict[ActionId, str] = {
    ActionId.PRICE_PERCENTILE: "Compare listing price with same make, model and year peers.",
    ActionId.MILEAGE_PERCENTILE: "Compare mileage with same make, model and year peers.",
    ActionId.FUEL_ECONOMY: "Compare fuel economy with comparable vehicle peers.",
    ActionId.VALUE_INDEX: "Calculate a bounded value index from price, mileage and fuel evidence.",
    ActionId.RECALL_ANALYSIS: "Request vehicle-specific NHTSA recall records for the model year.",
    ActionId.COMPLAINT_ANALYSIS: (
        "Request vehicle-specific NHTSA complaint records for the model year."
    ),
    ActionId.SAFETY_RATING: "Request an official safety-rating source before making safety claims.",
}


def action_description(action: str) -> str:
    try:
        return ACTION_DESCRIPTIONS[ActionId(action)]
    except (ValueError, KeyError):
        return action
