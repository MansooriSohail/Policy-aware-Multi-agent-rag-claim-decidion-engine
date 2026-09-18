# Policy Retrieval and Claim Analysis

This project implements the policy ingestion pipeline, hybrid retrieval layer, multi-agent claim reasoning workflow, and a FastAPI analysis API for the Aptino candidate assignment.

## Quick start

```bash
pip install -r requirements.txt
python -m policy_indexer.cli --pdf policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf --output-dir data/indexes
```

The pipeline writes chunk metadata and indexes to the output directory.

## API usage

Start the API with:

```bash
.
.\.venv\Scripts\Activate.ps1
uvicorn policy_indexer.api:app --reload
```

Then call the endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "API-001",
    "policy_id": "UNIHLIP18004V011718",
    "policy_start_date": "2018-01-01",
    "claim_date": "2019-01-15",
    "sum_insured_inr": 250000,
    "patient": {"name": "Asha", "age": 42, "gender": "female"},
    "hospital": {"name": "Sunrise Hospital"},
    "treatment": {
      "type": "inpatient",
      "diagnosis": "Fracture",
      "procedure": "Surgery",
      "admission_hours": 48,
      "pre_existing": false
    },
    "expenses": {"total": 120000},
    "task": "Assess whether this treatment is admissible under the policy."
  }'
```

The response follows the structured `DecisionOutput` contract with fields such as `case_id`, `decision`, `confidence`, `key_findings`, `applicable_limits`, `missing_evidence`, `citations`, and `validation`.

> This API is intentionally conservative: when the policy evidence is insufficient or the claim facts are incomplete, it returns a `NEEDS_REVIEW` outcome instead of inventing unsupported policy logic.
