"""Inspect source columns and normalize used-car listings."""

from pathlib import Path

import pandas as pd


def load_used_cars(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported vehicle data format: {path.suffix}")
