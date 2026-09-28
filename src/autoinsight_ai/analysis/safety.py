"""V2 safety-rating interface; missing ratings must remain distinct from zero."""

import pandas as pd


def analyze_safety(vehicles: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError(
        "Implement v2 safety ratings with missing ratings kept distinct from zero"
    )
