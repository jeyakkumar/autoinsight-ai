# Evaluation

`golden_questions.json` contains twenty **draft** cases. Each records relevant and
irrelevant actions, important insights and insufficient-evidence situations. These
are seed expectations, not validated golden labels. Curate them against a versioned
dataset, replace ambiguous vehicle references with listing IDs and record phase
availability before scoring. Questions involving complaints/safety require v2 data.

Run the same fixed questions and source snapshot through all four pipeline modes.
Store outputs in ignored `results/`, including configuration, source checksums,
provider/model versions, request IDs, selected actions, evidence, verification and timing.

Track planning precision/recall, selected-action precision, unnecessary-analysis rate,
supported-insight rate, unsupported-insight rejection rate, escalation rate, factual
and numerical accuracy, relevance, latency, tokens and cost. Mark unavailable metrics
as null rather than zero. Use human-adjudicated labels to avoid evaluating a model
solely with its own decisions. The evaluation runner awaits pipeline implementation.
