"""Value-index configuration; define and document the actual formula before use."""

import pandas as pd
from pydantic import Field, model_validator

from autoinsight_ai.planning.schemas import Contract


class ValueWeights(Contract):
    price: float = Field(default=0.6, ge=0, le=1)
    mileage: float = Field(default=0.3, ge=0, le=1)
    fuel_economy: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def normalized_weights(self) -> "ValueWeights":
        if abs(self.price + self.mileage + self.fuel_economy - 1.0) > 1e-9:
            raise ValueError("Value weights must sum to one")
        return self


def calculate_value_index(metrics: pd.DataFrame, weights: ValueWeights) -> pd.DataFrame:
    required = {"price_percentile", "mileage_percentile"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"value index requires columns: {missing}")
    result = metrics.copy()
    components = {
        "price": (1 - result["price_percentile"]) * weights.price,
        "mileage": (1 - result["mileage_percentile"]) * weights.mileage,
    }
    available_weight = pd.Series(
        weights.price + weights.mileage, index=result.index, dtype="Float64"
    )
    if "fuel_economy_percentile" in result:
        components["fuel_economy"] = result["fuel_economy_percentile"] * weights.fuel_economy
        available_weight = (
            available_weight + result["fuel_economy_percentile"].notna() * weights.fuel_economy
        )
    numerator = pd.concat(components, axis=1).sum(axis=1, min_count=1)
    result["value_index"] = numerator / available_weight.where(available_weight > 0)
    result["value_index_components"] = [
        {
            name: float(value)
            for name, value in ((key, frame.iloc[i]) for key, frame in components.items())
            if pd.notna(value)
        }
        for i in range(len(result))
    ]
    return result
