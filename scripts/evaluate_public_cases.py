from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from policy_indexer.orchestrator import PolicyDecisionWorkflow

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / 'candidate_data' / 'public_test_cases.json'
POLICY_PATH = ROOT / 'policy' / 'USGIC-CSCIndividualHealthInsurance_2017-2018.pdf'


def load_cases() -> List[Dict[str, Any]]:
    with CASES_PATH.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def main() -> None:
    workflow = PolicyDecisionWorkflow(POLICY_PATH)
    cases = load_cases()
    results: List[Dict[str, Any]] = []

    for case in cases:
        case_id = case.get('case_id', 'UNKNOWN')
        response = workflow.analyze_case(case)
        result = {
            'case_id': case_id,
            'decision': response.get('decision'),
            'confidence': response.get('confidence'),
            'validation_status': response.get('validation', {}).get('status'),
            'unsupported_claims': response.get('validation', {}).get('unsupported_claims', []),
            'missing_evidence': response.get('missing_evidence', []),
            'key_findings': response.get('key_findings', []),
            'applicable_limits': response.get('applicable_limits', []),
            'citations_count': len(response.get('citations', [])),
        }
        results.append(result)

    summary_path = ROOT / 'data' / 'public_case_evaluation.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print(f'Evaluated {len(results)} public cases.')
    print(f'Summary written to {summary_path}')

    for item in results:
        print(f"{item['case_id']}: {item['decision']} | validation={item['validation_status']} | confidence={item['confidence']}")


if __name__ == '__main__':
    main()
