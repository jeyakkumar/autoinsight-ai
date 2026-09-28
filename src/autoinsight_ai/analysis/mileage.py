"""Deterministic analysis interface for mileage percentiles within documented peer groups."""

import pandas as pd


def analyze_mileage(vehicles: pd.DataFrame) -> pd.DataFrame:
    required = {"make", "model", "model_year", "mileage"}
    missing = sorted(required.difference(vehicles.columns))
    if missing:
        raise ValueError(f"mileage analysis requires columns: {missing}")
    result = vehicles.copy()
    valid = result["mileage"].notna() & (result["mileage"] >= 0)
    groups = [
        "normalized_make" if "normalized_make" in result else "make",
        "normalized_model" if "normalized_model" in result else "model",
        "model_year",
    ]
    result["mileage_peer_count"] = (
        valid.groupby([result[column] for column in groups]).transform("sum").astype("Int64")
    )
    result["peer_mileage_median"] = (
        result["mileage"]
        .where(valid)
        .groupby([result[column] for column in groups], dropna=False)
        .transform("median")
    )
    result["mileage_percentile"] = (
        result["mileage"]
        .where(valid)
        .groupby([result[column] for column in groups], dropna=False)
        .rank(pct=True, method="average")
    )
    result["mileage_difference_pct"] = (result["mileage"] / result["peer_mileage_median"] - 1) * 100
    return result
