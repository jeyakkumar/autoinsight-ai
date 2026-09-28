"""Deterministic analysis interface for peer-price percentiles, price-per-mile and depreciation."""

import pandas as pd


def analyze_price(vehicles: pd.DataFrame) -> pd.DataFrame:
    _require_columns(vehicles, "make", "model", "model_year", "price")
    result = vehicles.copy()
    valid = result["price"].notna() & (result["price"] > 0)
    groups = _peer_columns(result)
    result["price_peer_count"] = (
        valid.groupby([result[column] for column in groups]).transform("sum").astype("Int64")
    )
    result["peer_price_median"] = (
        result["price"]
        .where(valid)
        .groupby([result[column] for column in groups], dropna=False)
        .transform("median")
    )
    result["price_percentile"] = (
        result["price"]
        .where(valid)
        .groupby([result[column] for column in groups], dropna=False)
        .rank(pct=True, method="average")
    )
    result["price_difference_pct"] = (result["price"] / result["peer_price_median"] - 1) * 100
    if "mileage" in result:
        result["price_per_mile"] = result["price"].where(
            valid & result["mileage"].notna() & (result["mileage"] > 0)
        ) / result["mileage"].where(result["mileage"] > 0)
    return result


def _require_columns(frame: pd.DataFrame, *columns: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"price analysis requires columns: {missing}")


def _peer_columns(frame: pd.DataFrame) -> list[str]:
    return [
        "normalized_make" if "normalized_make" in frame else "make",
        "normalized_model" if "normalized_model" in frame else "model",
        "model_year",
    ]
