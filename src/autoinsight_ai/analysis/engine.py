"""Registry for deterministic vehicle actions selected by CLM or a baseline."""

from collections.abc import Iterable

import pandas as pd

from autoinsight_ai.analysis.fuel_economy import analyze_fuel_economy
from autoinsight_ai.analysis.mileage import analyze_mileage
from autoinsight_ai.analysis.price import analyze_price
from autoinsight_ai.analysis.value import ValueWeights, calculate_value_index
from autoinsight_ai.planning.schemas import ActionId


def execute_actions(
    vehicles: pd.DataFrame,
    actions: Iterable[ActionId | str],
    *,
    value_weights: ValueWeights | None = None,
) -> pd.DataFrame:
    """Execute an allow-listed action sequence and return one enriched listing frame."""
    result = vehicles.copy()
    selected = {ActionId(action) for action in actions}
    if ActionId.PRICE_PERCENTILE in selected or ActionId.PRICE_PER_MILE in selected:
        result = analyze_price(result)
    if ActionId.MILEAGE_PERCENTILE in selected:
        result = analyze_mileage(result)
    if ActionId.FUEL_ECONOMY in selected:
        result = analyze_fuel_economy(result)
    if ActionId.VALUE_INDEX in selected:
        if "price_percentile" not in result or "mileage_percentile" not in result:
            result = analyze_price(result)
            result = analyze_mileage(result)
        result = calculate_value_index(result, value_weights or ValueWeights())
    return result
