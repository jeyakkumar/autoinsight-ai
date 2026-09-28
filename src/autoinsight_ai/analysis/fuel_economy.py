"""Deterministic analysis interface for fuel economy comparisons with explicit units."""

import pandas as pd


def analyze_fuel_economy(vehicles: pd.DataFrame) -> pd.DataFrame:
    required = {"make", "model", "model_year", "fuel_economy"}
    missing = sorted(required.difference(vehicles.columns))
    if missing:
        raise ValueError(f"fuel-economy analysis requires columns: {missing}")
    result = vehicles.copy()
    valid = result["fuel_economy"].notna() & (result["fuel_economy"] > 0)
    groups = ["normalized_make" if "normalized_make" in result else "make", "model_year"]
    result["fuel_economy_peer_count"] = (
        valid.groupby([result[column] for column in groups]).transform("sum").astype("Int64")
    )
    result["peer_fuel_economy_median"] = (
        result["fuel_economy"]
        .where(valid)
        .groupby([result[column] for column in groups], dropna=False)
        .transform("median")
    )
    result["fuel_economy_percentile"] = (
        result["fuel_economy"]
        .where(valid)
        .groupby([result[column] for column in groups], dropna=False)
        .rank(pct=True, method="average")
    )
    result["fuel_economy_difference_pct"] = (
        result["fuel_economy"] / result["peer_fuel_economy_median"] - 1
    ) * 100
    return result
