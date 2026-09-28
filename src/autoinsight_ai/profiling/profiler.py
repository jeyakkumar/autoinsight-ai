"""Profile counts, missing/duplicate data, ranges and suspicious vehicle attributes."""

import pandas as pd
from pydantic import JsonValue


def profile_vehicles(vehicles: pd.DataFrame) -> dict[str, JsonValue]:
    if vehicles.empty:
        return {"rows": 0, "duplicate_rows": 0, "missing_values": {}}
    profile: dict[str, JsonValue] = {
        "rows": int(len(vehicles)),
        "duplicate_rows": int(vehicles.duplicated().sum()),
        "missing_values": {str(k): int(v) for k, v in vehicles.isna().sum().items() if v},
        "unique_makes": int(
            vehicles["normalized_make" if "normalized_make" in vehicles else "make"].nunique(
                dropna=True
            )
        )
        if "make" in vehicles
        else 0,
        "unique_models": int(
            vehicles["normalized_model" if "normalized_model" in vehicles else "model"].nunique(
                dropna=True
            )
        )
        if "model" in vehicles
        else 0,
    }
    for column, name in (("model_year", "year"), ("price", "price"), ("mileage", "mileage")):
        if column in vehicles:
            values = pd.to_numeric(vehicles[column], errors="coerce").dropna()
            if not values.empty:
                profile[f"{name}_min"] = float(values.min())
                profile[f"{name}_max"] = float(values.max())
    suspicious: dict[str, int] = {}
    if "mileage" in vehicles:
        suspicious["negative_mileage"] = int(
            (pd.to_numeric(vehicles["mileage"], errors="coerce") < 0).sum()
        )
    if "price" in vehicles:
        prices = pd.to_numeric(vehicles["price"], errors="coerce")
        suspicious["non_positive_price"] = int((prices <= 0).sum())
        if prices.notna().sum() >= 4:
            upper = prices.quantile(0.99)
            suspicious["extreme_price"] = int((prices > upper).sum())
    if "model_year" in vehicles:
        years = pd.to_numeric(vehicles["model_year"], errors="coerce")
        suspicious["future_model_year"] = int((years > pd.Timestamp.now().year + 1).sum())
    profile["suspicious_values"] = suspicious
    return profile
