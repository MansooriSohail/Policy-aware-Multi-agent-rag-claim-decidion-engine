from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from policy_indexer.orchestrator import PolicyDecisionWorkflow
from policy_indexer.schemas import DecisionOutput

# Use the supplied policy source as the default evidence base for the API.
_POLICY_PATH = Path(__file__).resolve().parents[2] / 'policy' / 'USGIC-CSCIndividualHealthInsurance_2017-2018.pdf'
_WORKFLOW = PolicyDecisionWorkflow(_POLICY_PATH)

app = FastAPI(
    title='Policy Claim Analysis API',
    version='1.0.0',
    description='Evidence-grounded claim analysis powered by the policy indexer and hybrid retrieval stack.',
)


class ClaimRequest(BaseModel):
    """Incoming claim payload to analyze."""

    case_id: str
    policy_id: str
    policy_start_date: str | None = None
    claim_date: str | None = None
    sum_insured_inr: float | int | None = None
    patient: dict | None = None
    hospital: dict | None = None
    treatment: dict | None = None
    expenses: dict | None = None
    documents: list | None = None
    task: str | None = None

    model_config = ConfigDict(
        extra='allow',
        json_schema_extra={
            'example': {
                'case_id': 'API-001',
                'policy_id': 'UNIHLIP18004V011718',
                'policy_start_date': '2018-01-01',
                'claim_date': '2019-01-15',
                'sum_insured_inr': 250000,
                'patient': {'name': 'Asha', 'age': 42, 'gender': 'female'},
                'hospital': {'name': 'Sunrise Hospital'},
                'treatment': {
                    'type': 'inpatient',
                    'diagnosis': 'Fracture',
                    'procedure': 'Surgery',
                    'admission_hours': 48,
                    'pre_existing': False,
                },
                'expenses': {'total': 120000},
                'task': 'Assess whether this treatment is admissible under the policy.'
            }
        },
    )


@app.get('/health')
def health() -> dict[str, str]:
    """A lightweight readiness endpoint for service checks."""
    return {'status': 'ok'}


@app.post('/analyze', response_model=DecisionOutput, summary='Analyze a claim against the policy')
def analyze_claim(payload: ClaimRequest) -> DecisionOutput:
    """Compute a policy-grounded decision using the multi-agent workflow.

    The API intentionally returns the exact structured decision contract used by the
    workflow so external consumers can rely on a stable schema.

    Caution: the decision is only as reliable as the policy evidence retrieved and the
    claim facts provided. When the system lacks sufficient evidence, it returns a
    NEEDS_REVIEW outcome rather than guessing.
    """
    case = payload.model_dump(exclude_none=True)
    return _WORKFLOW.build_decision(case)


@app.post('/analyze-detailed', summary='Analyze a claim and return execution trace details')
def analyze_claim_detailed(payload: ClaimRequest) -> dict[str, Any]:
    """Return the full agent workflow state for UI trace rendering and debugging."""
    case = payload.model_dump(exclude_none=True)
    return _WORKFLOW.analyze_case(case)
