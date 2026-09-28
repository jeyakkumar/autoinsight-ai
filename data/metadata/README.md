# Data provenance

`sources.json` records intended sources from the requirements, not verified current
licenses or downloaded datasets. Before ingestion, extend each entry with retrieval
timestamp, source filename, SHA-256, license/terms URL, inspected columns, unit/currency
information, row count and normalization mappings. Keep data files out of version control.
Record unmatched and ambiguous enrichment joins; a missing match is not a zero count.
