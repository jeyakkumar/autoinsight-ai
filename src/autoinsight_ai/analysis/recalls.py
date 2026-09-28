"""Recall-count interface, including source coverage and campaign deduplication."""

import pandas as pd


def analyze_recalls(vehicles: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError(
        "Implement recall counts with source coverage and campaign deduplication"
    )
