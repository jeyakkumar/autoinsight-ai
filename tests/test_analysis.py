"""Configuration checks; numerical analytics tests belong here once implemented."""

import pandas as pd
import pytest
from pydantic import ValidationError

from autoinsight_ai.analysis.engine import execute_actions
from autoinsight_ai.analysis.value import ValueWeights
from autoinsight_ai.ingestion.normalization import normalize_vehicles
from autoinsight_ai.profiling.profiler import profile_vehicles


def test_rejects_unnormalized_value_weights():
    with pytest.raises(ValidationError):
        ValueWeights(price=0.9, mileage=0.9, fuel_economy=0.1)


@pytest.fixture
def listings():
    raw = pd.DataFrame(
        {
            "Brand": ["Ford", " ford ", "Toyota", "Ford", "Ford"],
            "Vehicle Model": ["Escape", "ESCAPE", "RAV4", "Escape", "Escape"],
            "Year": [2022, 2022, 2022, 2022, 2022],
            "Miles": [20_000, 40_000, 25_000, 10_000, 30_000],
            "Fuel": ["Gas", "gas", "Hybrid", "Gas", "Gas"],
            "Price": [24_800, 22_000, 28_000, 30_000, 26_000],
        }
    )
    return normalize_vehicles(
        raw,
        {
            "make": "Brand",
            "model": "Vehicle Model",
            "model_year": "Year",
            "mileage": "Miles",
            "fuel_type": "Fuel",
            "price": "Price",
        },
    )


def test_normalization_creates_stable_vehicle_key_and_numeric_values(listings):
    assert listings.loc[0, "vehicle_key"] == "ford|escape|2022"
    assert listings.loc[1, "vehicle_key"] == listings.loc[0, "vehicle_key"]
    assert listings.loc[0, "price"] == 24800
    assert listings.loc[0, "fuel_type"] == "gas"


def test_price_and_mileage_analysis_are_peer_based(listings):
    result = execute_actions(listings, ["price_percentile", "mileage_percentile"])
    escape = result[result["vehicle_key"] == "ford|escape|2022"]
    assert set(escape["price_peer_count"].dropna()) == {4}
    assert escape.loc[escape["price"].idxmin(), "price_percentile"] == 0.25
    assert escape.loc[escape["mileage"].idxmax(), "mileage_percentile"] == 1.0
    assert escape["mileage_peer_count"].iloc[0] == 4


def test_value_index_is_bounded_and_missing_features_do_not_become_zero(listings):
    result = execute_actions(listings, ["value_index"])
    assert result["value_index"].between(0, 1).all()
    assert result["value_index"].notna().all()


def test_profiler_reports_quality_signals(listings):
    profile = profile_vehicles(listings)
    assert profile["rows"] == 5
    assert profile["unique_makes"] == 2
    assert profile["suspicious_values"]["non_positive_price"] == 0
