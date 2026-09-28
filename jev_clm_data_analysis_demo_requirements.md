# JEV + CLM Data Analysis Demo — Requirements

## 1. Purpose

Build a small, reproducible **AI Data Analyst / Decision Pipeline** demonstrating how JEV and CLM can sit around a conventional analytics engine.

The goal is **not** to ask JEV or CLM to calculate business metrics or generate long-form answers. Instead:

- Python/SQL performs deterministic data analysis.
- An LLM converts a user question into candidate analysis plans and explains results.
- CLM ranks/selects candidate analytical actions or insights.
- JEV independently evaluates evidence, relevance, and whether an insight/action should be surfaced.
- The application combines these decisions into a transparent analysis workflow.

The demo should make it easy to compare:

1. Traditional analytics only
2. LLM + analytics
3. LLM + CLM + analytics
4. LLM + CLM + JEV + analytics

---

# 2. Recommended Demo Domain

## Vehicle Intelligence / Used-Car Decision Analysis

Instead of building another generic sales dashboard, use a vehicle dataset because it provides several different types of analytical decisions:

- price/depreciation
- mileage vs price
- fuel economy
- EV/hybrid/gas comparisons
- manufacturer/model trends
- safety
- recalls
- consumer complaints
- anomaly detection
- evidence-based recommendation/routing

### Primary demo question

> "Which vehicles look attractive from a value perspective, and which ones require additional investigation?"

The system should **not simply predict a car price**.

It should discover and verify multiple candidate insights, then decide which ones are sufficiently supported to show the analyst.

---

# 3. Data Sources

## 3.1 Used-car listings — MVP dataset

Use a publicly available used-car listing dataset such as the Kaggle **Used Car Price Prediction Dataset**.

The dataset contains approximately 4,009 vehicle listings and includes fields such as brand/model, model year, mileage, fuel type and other vehicle attributes.

Source:

- Kaggle: https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset

Use this dataset only as the initial market/listing dataset. Review the dataset's current license/terms before redistribution.

### Expected core fields

```text
brand
model
model_year
mileage
fuel_type
engine
transmission
price
```

The exact schema must be inspected at ingestion time rather than hard-coded.

---

## 3.2 EPA Automotive Trends / Fuel Economy — enrichment

Use public EPA automotive data to enrich vehicle characteristics.

EPA provides summary automotive trends data covering model years 1975–2025 and detailed vehicle-level data, with downloadable CSV files.

Source:

https://www.epa.gov/automotive-trends/explore-automotive-trends-data

Potential enrichment fields:

```text
manufacturer
model_year
vehicle_type
fuel_type
fuel_economy
vehicle_attributes
technology
```

EPA also provides annual fuel-economy test data.

Source:

https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy

---

## 3.3 NHTSA Recalls

Use NHTSA recall data to enrich each vehicle/model with recall information.

NHTSA provides public recall datasets and APIs covering vehicle recalls from 1949 to the present.

Source:

https://www.nhtsa.gov/nhtsa-datasets-and-apis

Example API pattern:

```text
/recalls/recallsByVehicle?make={MAKE}&model={MODEL}&modelYear={MODEL_YR}
```

Potential fields:

```text
recall_count
recall_campaigns
recall_component
recall_summary
recall_date
```

---

## 3.4 NHTSA Complaints

NHTSA also provides public vehicle-owner complaint data.

Potential fields:

```text
complaint_count
complaint_component
complaint_summary
complaint_date
```

Source:

https://www.nhtsa.gov/nhtsa-datasets-and-apis

This is especially useful because the pipeline can distinguish:

> "Low price"

from:

> "Low price but unusually high complaint/recall activity."

---

## 3.5 NHTSA Safety Ratings

NHTSA provides vehicle safety-rating data by model year, make and model.

Potential enrichment:

```text
overall_safety_rating
frontal_crash_rating
side_crash_rating
rollover_rating
```

Source:

https://www.nhtsa.gov/nhtsa-datasets-and-apis

---

# 4. Demo Business Scenario

Build a fictional vehicle marketplace/consumer analytics product called:

## AutoInsight AI

A user can ask:

> "Find interesting used vehicles under $30K."

or:

> "Why is this vehicle cheaper than similar vehicles?"

or:

> "Which SUVs provide good value without unusual safety signals?"

or:

> "What should I investigate before buying this vehicle?"

The system does not directly produce a recommendation from the LLM.

Instead, it creates candidate analytical actions and candidate insights.

---

# 5. High-Level Architecture

```text
                         User
                           |
                           v
                  +----------------+
                  | Question Parser|
                  |      LLM       |
                  +-------+--------+
                          |
                          v
                 Candidate Analysis
                      Plans
                          |
              +-----------+-----------+
              |                       |
              v                       v
        +-----------+           +-----------+
        |    CLM    |           |    JEV    |
        | Candidate |           | Evidence  |
        | Selection |           | Decision  |
        +-----+-----+           +-----+-----+
              |                       |
              +-----------+-----------+
                          |
                          v
                 Analysis Orchestrator
                          |
             +------------+------------+
             |            |            |
             v            v            v
          Pandas        SQL/API      NHTSA
          Metrics       Queries      Enrichment
             |            |            |
             +------------+------------+
                          |
                          v
                   Verified Results
                          |
                          v
                   Insight Generator
                       LLM
                          |
                          v
                     Dashboard
```

---

# 6. Core Pipeline

## Stage 1 — Data ingestion

Load:

```text
used_car_listings
epa_vehicle_data
nhtsa_recalls
nhtsa_complaints
nhtsa_safety
```

Normalize:

```text
make
model
model_year
fuel_type
```

Create a canonical vehicle key:

```text
vehicle_key =
    normalized_make
    + normalized_model
    + model_year
```

---

# 7. Stage 2 — Data Profiling

The deterministic profiler calculates:

```text
row_count
missing_values
duplicate_count
unique_makes
unique_models
year_range
price_range
mileage_range
```

Example:

```json
{
  "rows": 4009,
  "duplicate_rows": 12,
  "missing_price": 0,
  "missing_mileage": 37,
  "year_min": 2010,
  "year_max": 2025
}
```

The profiler should also detect suspicious values:

```text
negative mileage
zero price
extreme price
future model year
invalid fuel type
```

---

# 8. Stage 3 — Candidate Analysis Generation

The LLM converts the user's question into candidate analytical actions.

Example:

```text
User:
"Which SUVs under $30K are interesting?"

Candidate actions:

A. Calculate price percentile by model
B. Calculate price-per-mile
C. Compare model year depreciation
D. Compare fuel economy
E. Calculate recall count
F. Calculate complaint count
G. Compare safety ratings
H. Detect price anomalies
I. Compare mileage against peer vehicles
```

The LLM must output structured JSON.

Example:

```json
{
  "question": "Which SUVs under $30K are interesting?",
  "candidate_actions": [
    {
      "id": "price_percentile",
      "description": "Compare listing price against peer vehicles",
      "required_data": ["price", "make", "model", "model_year"]
    },
    {
      "id": "recall_analysis",
      "description": "Calculate recall activity by vehicle",
      "required_data": ["make", "model", "model_year"]
    }
  ]
}
```

---

# 9. Stage 4 — CLM Candidate Selection

CLM receives the current analytical state and candidate actions.

Purpose:

> Determine which analytical actions are worth executing.

Example:

```text
Candidate actions:

1. Price percentile
2. Price-per-mile
3. Fuel economy
4. Recall analysis
5. Complaint analysis
6. Safety rating
7. Dealer location analysis
8. Color distribution
```

CLM produces a ranking/selection signal.

The application should retain:

```text
candidate_action
CLM score
selected
reason/state metadata
```

Do not ask CLM to produce the final natural-language answer.

---

# 10. Stage 5 — Deterministic Analysis

Execute only the selected analytical actions.

Example calculations:

### Price percentile

```text
vehicle price percentile within
make + model + model_year
```

### Mileage percentile

```text
vehicle mileage percentile within peer group
```

### Value index

Example:

```text
value_index =
    peer_price_percentile
    + mileage_adjustment
    + fuel_economy_adjustment
```

The exact scoring formula should remain deterministic and configurable.

### Recall signal

```text
recall_count by make/model/model_year
```

### Complaint signal

```text
complaint_count by make/model/model_year
```

### Safety signal

```text
NHTSA safety rating
```

---

# 11. Stage 6 — Candidate Insight Generation

Generate candidate insights from deterministic results.

Example:

```text
Insight A:
Vehicle price is 18% below peer median.

Insight B:
Vehicle mileage is 62nd percentile.

Insight C:
Vehicle has higher-than-peer recall activity.

Insight D:
Vehicle has above-average fuel economy.

Insight E:
Vehicle price appears anomalously low after controlling for year and mileage.
```

Each insight must contain machine-readable evidence.

Example:

```json
{
  "insight_id": "I-104",
  "vehicle": "Example SUV 2022",
  "claim": "Listing price is 18% below peer median",
  "evidence": {
    "listing_price": 24800,
    "peer_median": 30200,
    "difference_pct": -17.9,
    "peer_count": 143
  }
}
```

---

# 12. Stage 7 — JEV Evidence Verification

JEV receives the analytical state and candidate insight.

Questions should be narrow and typed.

Example:

```text
Question 1:
Is this price difference meaningful enough to investigate?

Question 2:
Is there sufficient peer data?

Question 3:
Does the evidence support surfacing this insight?

Question 4:
Should the system request additional analysis?

Question 5:
Should this insight be escalated to the user?
```

Example structured result:

```json
{
  "insight_id": "I-104",
  "evidence_sufficient": true,
  "surface_to_user": true,
  "request_more_analysis": false,
  "probabilities": {
    "surface": 0.91,
    "investigate": 0.87
  }
}
```

The exact JEV API/model contract must be verified during implementation.

---

# 13. Stage 8 — Confidence-Based Routing

Use JEV output to determine workflow routing.

```text
                  Candidate Insight
                         |
                         v
                       JEV
                         |
             +-----------+-----------+
             |                       |
        Sufficient              Insufficient
        evidence                  evidence
             |                       |
             v                       v
        Show insight          Run additional
                              analysis
                                    |
                                    v
                                  JEV
```

Example:

```text
JEV surface_probability >= 0.85
    -> surface

0.60 <= probability < 0.85
    -> perform additional analysis

probability < 0.60
    -> suppress or request human review
```

Thresholds must be configuration, not hard-coded into model prompts.

---

# 14. Stage 9 — Final LLM Explanation

Only after deterministic calculations and JEV verification should the LLM generate the user-facing explanation.

Input:

```json
{
  "vehicle": "...",
  "verified_insights": [...],
  "evidence": [...],
  "analysis_steps": [...]
}
```

Output:

```text
This vehicle is priced approximately 18% below comparable
2022 SUVs in the dataset.

The listing also has below-median mileage.

However, the model/year combination shows higher-than-peer
recall activity, so recall details should be reviewed before
treating the lower price as a pure value opportunity.
```

The LLM must not invent numerical values.

---

# 15. Demo Use Cases

## Use Case 1 — Find unusual value

Question:

> "Which used SUVs appear unusually cheap?"

Pipeline:

```text
LLM -> candidate analyses
CLM -> select peer-price + mileage analysis
Python -> calculate metrics
JEV -> verify evidence
LLM -> explain
```

---

## Use Case 2 — Why is this vehicle cheap?

Question:

> "Why is this 2022 SUV priced at $24,800?"

Possible analysis:

```text
peer price
mileage
model year
fuel economy
recalls
complaints
safety rating
```

JEV decides which evidence is sufficiently strong to surface.

---

## Use Case 3 — Value + safety investigation

Question:

> "Find vehicles under $30K that have good value but don't show unusual safety signals."

Pipeline:

```text
price
   +
mileage
   +
peer comparison
   +
recalls
   +
complaints
   +
NHTSA safety rating
```

The system should present the underlying evidence instead of hiding the decision behind a single AI score.

---

## Use Case 4 — Anomaly detection

Question:

> "Find suspiciously low-priced listings."

Detect:

```text
price < peer expected price
AND
sufficient peer sample
```

Then investigate:

```text
mileage
year
trim
fuel type
recalls
complaints
```

JEV decides whether the anomaly is strong enough to surface.

---

## Use Case 5 — Vehicle comparison

Question:

> "Compare three SUVs for value."

Generate candidate comparisons:

```text
price
mileage
fuel economy
depreciation
recalls
complaints
safety
```

CLM selects the relevant dimensions.

JEV evaluates whether each comparison is supported by sufficient evidence.

---

# 16. Recommended First Demo

Start with only:

```text
Used-car listings
        +
EPA fuel economy
        +
NHTSA recalls
```

Do not start with every dataset.

### Version 1

```text
CSV
 |
Pandas
 |
LLM
 |
CLM
 |
Python analysis
 |
JEV
 |
LLM explanation
 |
Streamlit
```

### Version 2

Add:

```text
NHTSA complaints
NHTSA safety ratings
```

### Version 3

Add:

```text
live NHTSA API
additional market datasets
vector search
historical trend analysis
```

---

# 17. Technology Stack

## Backend

```text
Python 3.12+
FastAPI
Pandas
DuckDB
Pydantic
```

## AI

```text
LLM
CLM
JEV
```

## UI

```text
Streamlit
```

Optional later:

```text
React
```

## Data

```text
CSV
Parquet
DuckDB
```

Avoid introducing a large database for the first demo.

---

# 18. Suggested Repository

```text
autoinsight-ai/
│
├── README.md
├── requirements.md
├── pyproject.toml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── src/
│   ├── ingestion/
│   │   ├── used_cars.py
│   │   ├── epa.py
│   │   └── nhtsa.py
│   │
│   ├── profiling/
│   │   └── profiler.py
│   │
│   ├── planning/
│   │   ├── llm_planner.py
│   │   └── schemas.py
│   │
│   ├── decision/
│   │   ├── clm.py
│   │   └── jev.py
│   │
│   ├── analysis/
│   │   ├── price.py
│   │   ├── mileage.py
│   │   ├── fuel_economy.py
│   │   ├── recalls.py
│   │   └── complaints.py
│   │
│   ├── insights/
│   │   ├── generator.py
│   │   └── verifier.py
│   │
│   └── orchestration/
│       └── pipeline.py
│
├── app/
│   └── streamlit_app.py
│
└── tests/
    ├── test_analysis.py
    ├── test_clm.py
    ├── test_jev.py
    └── test_pipeline.py
```

---

# 19. Observability

Every decision should be logged.

Example:

```json
{
  "request_id": "abc-123",
  "question": "Which SUVs are unusually cheap?",
  "candidate_actions": 8,
  "clm_selected": 4,
  "analysis_executed": 4,
  "candidate_insights": 12,
  "jev_verified": 7,
  "insights_surface": 4
}
```

Record:

```text
LLM latency
CLM latency
JEV latency
analysis latency
token usage
candidate count
selection rate
verification rate
final insight count
```

This is important for evaluating whether the System One layer actually improves the pipeline.

---

# 20. Evaluation Plan

Create a small golden test set:

```text
20 user questions
```

For each question define expected:

```text
relevant analysis actions
irrelevant actions
important insights
insufficient-evidence cases
```

Measure:

### Planning

```text
candidate-action precision
candidate-action recall
```

### CLM

```text
selected-action precision
unnecessary-analysis rate
latency
```

### JEV

```text
supported-insight rate
unsupported-insight rejection rate
escalation rate
```

### Final system

```text
factual accuracy
numerical accuracy
insight relevance
latency
cost
```

The most important experiment is:

```text
Baseline:
LLM -> Python -> Answer

Experiment:
LLM -> CLM -> Python -> JEV -> Answer
```

Compare the two pipelines on the same questions.

---

# 21. Success Criteria

The demo is successful if it can demonstrate:

1. The LLM generates multiple possible analysis actions.
2. CLM reduces unnecessary analysis.
3. Python performs the actual numerical calculations.
4. JEV independently evaluates whether evidence supports surfacing an insight.
5. The final LLM cannot invent unsupported numerical facts.
6. Every surfaced insight has traceable evidence.
7. Low-confidence decisions can trigger additional analysis.
8. The entire decision path is observable.

---

# 22. Important Design Principle

Do not build:

```text
User
  ↓
LLM
  ↓
JEV
  ↓
Answer
```

Build:

```text
User
  ↓
LLM
  ↓
Candidate Plans
  ↓
CLM
  ↓
Selected Plans
  ↓
Deterministic Analytics
  ↓
Candidate Insights
  ↓
JEV
  ↓
Verified Insights
  ↓
LLM
  ↓
Answer
```

This makes the demo useful as an **AI analytics architecture experiment**, rather than simply another chatbot.

---

# 23. Future Extension

Once the vehicle demo works, the same architecture can be applied to enterprise analytics:

```text
Vehicle Intelligence
       ↓
Retail Analytics
       ↓
Discover / Consumer Analytics
       ↓
Enterprise AI Analyst
```

The domain-specific analytics functions can change while the decision architecture remains:

```text
LLM
 ↓
Candidate Actions
 ↓
CLM
 ↓
Analytics Engine
 ↓
JEV
 ↓
Verified Insights
 ↓
LLM
```

That separation should be treated as the primary architectural objective of the PoC.
