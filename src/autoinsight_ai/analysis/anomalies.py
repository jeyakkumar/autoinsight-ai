"""Deterministic analysis interface for price anomalies controlling for year and mileage."""

import pandas as pd


def analyze_anomalies(vehicles: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Implement price anomalies controlling for year and mileage")
