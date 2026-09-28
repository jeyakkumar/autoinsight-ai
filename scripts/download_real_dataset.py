"""Fetch the current public AutoTrader preview and build the app's CSV schema."""

import subprocess
import tempfile
from pathlib import Path

import pandas as pd

REPOSITORY = "https://github.com/rebrowser/autotrader-dataset.git"
DESTINATION = Path("data/raw/vehicles_us_cleaned.csv")


def build_dataset(source_dir: Path, destination: Path = DESTINATION) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((source_dir / "car-listings" / "data").glob("*")):
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix == ".csv":
            frame = pd.read_csv(path)
        else:
            continue
        sale_price = pd.to_numeric(frame.get("salePrice"), errors="coerce")
        fair_low = pd.to_numeric(frame.get("kbbFairPriceLow"), errors="coerce")
        fair_high = pd.to_numeric(frame.get("kbbFairPriceHigh"), errors="coerce")
        frame["price"] = sale_price.fillna((fair_low + fair_high) / 2)
        frame["price_source"] = sale_price.notna().map(
            {True: "listing_price", False: "kbb_fair_midpoint"}
        )
        columns = {
            "make": "makeName",
            "model": "modelName",
            "model_year": "year",
            "mileage": "mileage",
            "fuel_type": "fuelTypeGroup",
            "price": "price",
            "fuel_economy": "mpgHighway",
            "price_source": "price_source",
            "listing_type": "listingType",
            "body_style": "bodyStyle",
            "state": "sellerState",
            "days_on_market": "daysOnMarket",
            "deal_indicator": "dealIndicator",
            "listing_seen_at": "_lastSeenAt",
            "listing_id": "listingId",
        }
        available = {name: source for name, source in columns.items() if source in frame}
        renamed = frame[list(available.values())].rename(
            columns={v: k for k, v in available.items()}
        )
        frames.append(renamed)
    result = pd.concat(frames, ignore_index=True)
    for column in ["model_year", "mileage", "price", "fuel_economy"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["make", "model", "model_year", "mileage", "price"])
    result["model_year"] = result["model_year"].astype(int)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="autoinsight-autotrader-") as temporary:
        checkout = Path(temporary) / "autotrader-dataset"
        subprocess.run(["git", "clone", "--depth", "1", REPOSITORY, str(checkout)], check=True)
        result = build_dataset(checkout)
    print(
        f"Wrote {len(result):,} rows to {DESTINATION} "
        f"(model years {result.model_year.min()}–{result.model_year.max()})."
    )


if __name__ == "__main__":
    main()
