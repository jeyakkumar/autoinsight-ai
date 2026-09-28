"""Inspect EPA schema, units and granularity before vehicle enrichment."""

from pathlib import Path

import pandas as pd


def load_epa(path: Path) -> pd.DataFrame:
    raise NotImplementedError("Inspect EPA schema, units and granularity before vehicle enrichment")
