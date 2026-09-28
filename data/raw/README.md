# Real listing data

The dashboard looks for `vehicles_us_cleaned.csv` in this directory. Refresh it
from the current public AutoTrader preview with:

```bash
uv run python scripts/download_real_dataset.py
```

The refresh combines the recent CSV and Parquet files in the preview, maps
`year`, `mileage`, `makeName`, and `modelName` to the canonical fields, and
uses the KBB fair-price midpoint when the preview masks the listing price.
`price_source` records that distinction. The generated file is ignored by Git;
review the provider's license and terms before redistributing it.
