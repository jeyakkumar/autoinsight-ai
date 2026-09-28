"""Canonical identity and fuel normalization boundary."""

import pandas as pd

CANONICAL_COLUMNS = {
    "make",
    "model",
    "model_year",
    "mileage",
    "fuel_type",
    "price",
}


def _clean_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def normalize_vehicles(vehicles: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Map inspected source columns; preserve raw fields and use an unambiguous vehicle key."""
    if vehicles.empty:
        raise ValueError("vehicles must contain at least one row")
    missing_map = CANONICAL_COLUMNS.difference(column_map)
    if missing_map:
        raise ValueError(f"column_map is missing canonical columns: {sorted(missing_map)}")
    missing_source = {column_map[name] for name in CANONICAL_COLUMNS}.difference(vehicles.columns)
    if missing_source:
        raise ValueError(f"source columns are missing: {sorted(missing_source)}")

    result = vehicles.copy()
    result["make"] = _clean_text(result[column_map["make"]])
    result["model"] = _clean_text(result[column_map["model"]])
    result["fuel_type"] = _clean_text(result[column_map["fuel_type"]]).str.lower()
    result["model_year"] = pd.to_numeric(result[column_map["model_year"]], errors="coerce").astype(
        "Int64"
    )
    result["mileage"] = _numeric_measure(result[column_map["mileage"]])
    result["price"] = _numeric_measure(result[column_map["price"]])
    result["normalized_make"] = (
        result["make"].str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
    )
    result["normalized_model"] = (
        result["model"].str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
    )
    result["vehicle_key"] = (
        result["normalized_make"]
        + "|"
        + result["normalized_model"]
        + "|"
        + result["model_year"].astype("string")
    )
    result.loc[
        result[["normalized_make", "normalized_model", "model_year"]].isna().any(axis=1),
        "vehicle_key",
    ] = pd.NA
    return result


def _numeric_measure(values: pd.Series) -> pd.Series:
    """Parse numbers and common currency/mileage text without guessing units."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").astype("Float64")
    cleaned = values.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype("Float64")
