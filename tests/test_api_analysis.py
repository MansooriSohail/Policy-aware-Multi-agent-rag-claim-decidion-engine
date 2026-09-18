import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from policy_indexer.api import app


def test_analyze_endpoint_returns_structured_decision_contract():
    client = TestClient(app)

    payload = {
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

    response = client.post('/analyze', json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    expected_keys = {
        'case_id',
        'decision',
        'confidence',
        'key_findings',
        'applicable_limits',
        'missing_evidence',
        'citations',
        'validation',
    }
    assert set(body.keys()) == expected_keys
    assert body['case_id'] == 'API-001'
    assert body['decision'] in {'ADMISSIBLE', 'ADMISSIBLE_WITH_LIMITS', 'PARTIALLY_ADMISSIBLE', 'NOT_ADMISSIBLE', 'NEEDS_REVIEW'}
    assert set(body['validation'].keys()) == {'status', 'unsupported_claims', 'revision_needed'}
    assert isinstance(body['citations'], list)
    assert isinstance(body['key_findings'], list)
    assert isinstance(body['applicable_limits'], list)
    assert isinstance(body['missing_evidence'], list)
