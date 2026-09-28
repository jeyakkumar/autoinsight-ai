# Implementation status

## Available now

- Installable Python package and backend/UI development entry points.
- Strict Pydantic contracts for plans, selections, evidence, verification and outputs.
- Four comparison modes with explicit CLM/JEV/LLM capability switches.
- Configurable confidence routing that requires sufficient evidence to surface.
- Structured telemetry contract and JSON logger utility.
- Provider protocols and clearly marked implementation stubs.
- Twenty draft evaluation cases, source manifest and local-data directories.

## Implementation sequence

1. Review dataset licenses and download listings, EPA data and recall data locally.
   Record retrieval time, license, checksum, schema, units and source versions.
2. Inspect column names; map source columns to canonical make/model/year/fuel fields.
   Use an unambiguous vehicle key and preserve listing identifiers and provenance.
   Validate enrichment cardinality; EPA summaries must not become vehicle-level matches.
3. Implement deterministic profiling, including missing/duplicate values and suspect
   mileage, price, year and fuel types. Distinguish missing enrichment from zero events.
4. Implement peer-price, mileage, fuel and recall metrics, then a documented configurable
   value formula. Expose peer sample size, units, reference populations and source dates.
5. Verify LLM/CLM/JEV provider contracts and implement adapters. Validate structured model
   outputs and restrict executable action IDs to the deterministic action registry.
   Treat listing text as untrusted data. Models must not execute arbitrary Python or SQL.
6. Wire orchestration: plan → select → calculate → generate evidence → verify → route.
   Respect mode switches, log every decision and cap follow-up analysis using configuration.
   Reverify new evidence after follow-up analysis; do not loop indefinitely.
7. Add explanation from surfaced evidence only. Check generated numerical claims against
   evidence before display; structured schemas alone do not ensure numerical accuracy.
8. Connect the dashboard and add an analysis API only once orchestration works. Display
   candidate actions, CLM selections, deterministic results, JEV decisions and provenance.
9. Curate and label the twenty evaluation cases against a fixed dataset snapshot, then
   compare all four modes on the same questions and record latency, tokens and cost.
10. Add complaints and safety in v2; live APIs, vector search and historical trends in v3.

## Integration boundaries

`Planner`, `Selector`, `Verifier`, and `Explainer` are internal protocols. CLM must
select actions; JEV must evaluate evidence independently; neither calculates metrics
or writes the final answer. The orchestration stub intentionally produces no results.
For non-JEV baseline modes, record insights as unverified rather than labeling them
JEV-verified. Recall and complaint counts alone cannot establish vehicle safety.

Automated CI, deployment, model adapters and real data acquisition are not configured.
The initial tests validate the scaffold's contracts and routing, not model quality.
