import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_indexer.orchestrator import PolicyDecisionWorkflow


WORKFLOW = PolicyDecisionWorkflow(ROOT / 'policy' / 'USGIC-CSCIndividualHealthInsurance_2017-2018.pdf')


with (ROOT / 'candidate_data' / 'public_test_cases.json').open('r', encoding='utf-8') as handle:
    CASES = json.load(handle)


def get_case(case_id: str):
    for case in CASES:
        if case['case_id'] == case_id:
            return case
    raise KeyError(case_id)


def test_waiting_period_case_is_not_admissible():
    case = get_case('PUB-002')
    result = WORKFLOW.analyze_case(case)
    assert result['decision'] not in {'ADMISSIBLE', 'ADMISSIBLE_WITH_LIMITS', 'PARTIALLY_ADMISSIBLE'}


def test_pre_existing_case_is_not_admissible():
    case = get_case('PUB-003')
    result = WORKFLOW.analyze_case(case)
    assert result['decision'] not in {'ADMISSIBLE', 'ADMISSIBLE_WITH_LIMITS', 'PARTIALLY_ADMISSIBLE'}


def test_missing_hospital_evidence_requires_review():
    case = get_case('PUB-011')
    result = WORKFLOW.analyze_case(case)
    assert result['validation']['status'] == 'NEEDS_REVIEW'


def test_experimental_treatment_is_excluded():
    case = get_case('PUB-012')
    result = WORKFLOW.analyze_case(case)
    assert result['decision'] in {'NOT_ADMISSIBLE', 'NEEDS_REVIEW'}
