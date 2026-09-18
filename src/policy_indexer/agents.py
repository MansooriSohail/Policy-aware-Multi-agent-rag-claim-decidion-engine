from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from policy_indexer.retrieval import HybridRetriever
from policy_indexer.schemas import (
    AnalysisState,
    EvidenceCitation,
    RetrievalQuery,
    RetrievalResult,
    ReasoningState,
    ValidationState,
)


class AnalysisAgent:
    """Extract the core facts and plan retrieval queries for a claim."""

    def analyze(self, case: Dict[str, Any]) -> AnalysisState:
        case_id = str(case.get('case_id', 'UNKNOWN'))

        patient = case.get('patient', {})
        treatment = case.get('treatment', {})
        expenses = case.get('expenses', {})
        hospital = case.get('hospital', {})

        patient_summary = {
            'name': patient.get('name'),
            'age': patient.get('age'),
            'gender': patient.get('gender'),
        }
        claim_summary = {
            'policy_id': case.get('policy_id'),
            'claim_date': case.get('claim_date'),
            'sum_insured_inr': case.get('sum_insured_inr'),
            'diagnosis': treatment.get('diagnosis'),
            'procedure': treatment.get('procedure'),
            'type': treatment.get('type'),
            'admission_hours': treatment.get('admission_hours'),
            'hospital': hospital.get('name'),
            'total_expense': expenses.get('total'),
        }

        missing_fields: List[str] = []
        required_keys = [
            'case_id',
            'policy_id',
            'policy_start_date',
            'claim_date',
            'sum_insured_inr',
            'patient',
            'hospital',
            'treatment',
            'expenses',
            'task',
        ]
        for key in required_keys:
            if key not in case:
                missing_fields.append(key)

        retrieval_queries = [
            RetrievalQuery(query='waiting period policy coverage', intent='waiting_period', priority=1),
            RetrievalQuery(query='pre-existing disease exclusion', intent='pre_existing', priority=2),
            RetrievalQuery(query='hospitalization coverage scope and limits', intent='coverage_scope', priority=3),
            RetrievalQuery(query='category limits sub-limits deductions', intent='limits', priority=4),
        ]

        return AnalysisState(
            case_id=case_id,
            patient_summary=patient_summary,
            claim_summary=claim_summary,
            missing_fields=missing_fields,
            retrieval_queries=retrieval_queries,
        )


class RetrievalAgent:
    """Execute hybrid search and return ranked, metadata-rich evidence."""

    def __init__(self, index_data: Dict[str, Any]):
        self.retriever = HybridRetriever(index_data)

    def retrieve(self, analysis: AnalysisState, top_k: int = 5) -> List[RetrievalResult]:
        evidence: List[RetrievalResult] = []

        for query in analysis.retrieval_queries:
            hits = self.retriever.search(query.query, top_k=top_k)
            for hit in hits:
                evidence.append(
                    RetrievalResult(
                        chunk_id=hit['chunk_id'],
                        text=hit['text'],
                        section=hit['section'],
                        heading=hit.get('heading'),
                        page_start=hit['page_start'],
                        page_end=hit['page_end'],
                        dense_score=hit.get('dense_score', 0.0),
                        sparse_score=hit.get('sparse_score', 0.0),
                        fused_score=hit.get('fused_score', 0.0),
                        rerank_score=hit.get('rerank_score', 0.0),
                    )
                )

        deduped: Dict[str, RetrievalResult] = {}
        for item in evidence:
            if item.chunk_id not in deduped:
                deduped[item.chunk_id] = item
            else:
                current = deduped[item.chunk_id]
                current.fused_score = max(current.fused_score, item.fused_score)
                current.rerank_score = max(current.rerank_score, item.rerank_score)

        ranked = sorted(deduped.values(), key=lambda item: item.rerank_score, reverse=True)
        return ranked[:top_k]


class ReasoningAgent:
    """Reason over policy evidence and produce structured findings."""

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'true', 'yes', 'y', '1'}
        return bool(value)

    def reason(self, case: Dict[str, Any], retrieval_results: List[RetrievalResult]) -> ReasoningState:
        case_id = str(case.get('case_id', 'UNKNOWN'))
        coverage_findings: List[str] = []
        waiting_period_findings: List[str] = []
        limit_findings: List[str] = []
        exclusion_findings: List[str] = []
        unsupported_claims: List[str] = []

        patient = case.get('patient', {})
        treatment = case.get('treatment', {})
        hospital = case.get('hospital', {})
        exp_timing = case.get('expense_timing', {})
        prior_policy = case.get('prior_policy', {})

        continuous_months = case.get('continuous_coverage_months')
        treatment_type = str(treatment.get('type', '')).lower()
        diagnosis = str(treatment.get('diagnosis', '')).lower()
        pre_existing = self._as_bool(treatment.get('pre_existing'))
        is_experimental = self._as_bool(treatment.get('experimental'))
        hospital_name = str(hospital.get('name', '')).lower()
        network_provider = self._as_bool(hospital.get('network_provider'))

        # Waiting period is a core policy gate and must not be assumed away.
        if continuous_months is not None and continuous_months < 12:
            waiting_period_findings.append('The claim falls within the initial waiting-period window and is not eligible for full coverage.')
            unsupported_claims.append('Initial waiting period is not satisfied.')

        # Pre-existing disease cover is a known exclusion condition in the policy.
        if pre_existing:
            exclusion_findings.append('The treatment relates to a pre-existing condition, which is subject to the policy waiting-period rules.')
            if continuous_months is not None and continuous_months < 48:
                unsupported_claims.append('Pre-existing condition waiting period has not been completed.')

        # Domiciliary treatment is only covered under specific conditions.
        if treatment_type == 'domiciliary':
            coverage_findings.append('The treatment is domiciliary and should be checked against the domiciliary-coverage conditions.')
            room_unavailable = self._as_bool(treatment.get('hospital_room_unavailable'))
            patient_cannot_be_moved = self._as_bool(treatment.get('patient_cannot_be_moved'))
            if not (room_unavailable or patient_cannot_be_moved):
                unsupported_claims.append('Domiciliary treatment is not supported by the required policy conditions.')

        # Day-care treatment must meet the less-than-24-hour and medically necessary criteria.
        if treatment_type in {'day_care', 'day-care'}:
            coverage_findings.append('The treatment is day-care and should satisfy the less-than-24-hour hospitalization criteria.')
            admission_hours = treatment.get('admission_hours')
            if isinstance(admission_hours, (int, float)) and admission_hours > 24:
                unsupported_claims.append('Day-care treatment exceeds the less-than-24-hour policy definition.')

        # Experimental or unproven treatment is explicitly excluded.
        if is_experimental:
            exclusion_findings.append('The treatment is marked as experimental or unproven, which is excluded under the policy.')
            unsupported_claims.append('Experimental treatment is excluded.')

        # Missing hospital evidence should trigger review instead of auto-admission.
        if treatment_type == 'inpatient' and not hospital_name:
            unsupported_claims.append('Hospital evidence is missing for an inpatient claim.')

        # If the facility is not clearly a hospital or hospital criteria are missing, require review.
        if hospital_name and 'unknown' in hospital_name:
            unsupported_claims.append('Hospital identity is not established enough for final admissibility determination.')

        hospital_registered = case.get('evidence_context', {}).get('hospital_registered')
        hospital_criteria = case.get('evidence_context', {}).get('hospital_minimum_criteria_documented')
        if hospital_registered is False or hospital_criteria is False:
            unsupported_claims.append('Hospital minimum-criteria evidence is insufficient for a final decision.')

        # Claim windows: pre/post hospitalization are only payable when the underlying claim is admissible.
        if exp_timing:
            pre_days = exp_timing.get('pre_hospitalization_days_before_admission')
            post_days = exp_timing.get('post_hospitalization_days_after_discharge')
            if pre_days is not None and pre_days > 30:
                unsupported_claims.append('Pre-hospitalization expenses exceed the policy window.')
            if post_days is not None and post_days > 60:
                unsupported_claims.append('Post-hospitalization expenses exceed the policy window.')

        # Prior coverage may reduce waiting periods if policy conditions are met.
        if prior_policy:
            prior_years = prior_policy.get('continuous_years', 0)
            if prior_years and continuous_months is not None and continuous_months < 12:
                waiting_period_findings.append('Prior insurance continuity may reduce the waiting period, but the claim still needs policy evidence.')

        if not retrieval_results:
            unsupported_claims.append('No policy evidence was retrieved for the claim assessment.')

        citations = [
            EvidenceCitation(
                chunk_id=result.chunk_id,
                section=result.section,
                heading=result.heading,
                page_start=result.page_start,
                page_end=result.page_end,
                source='policy.pdf',
                snippet=result.text[:400],
            )
            for result in retrieval_results
        ]

        decision = 'ADMISSIBLE'
        if unsupported_claims:
            decision = 'NEEDS_REVIEW'

        if 'experimental' in diagnosis or is_experimental:
            decision = 'NOT_ADMISSIBLE'

        if pre_existing and continuous_months is not None and continuous_months < 48:
            decision = 'NOT_ADMISSIBLE'

        if continuous_months is not None and continuous_months < 12:
            decision = 'NOT_ADMISSIBLE'

        if treatment_type == 'domiciliary' and not (self._as_bool(treatment.get('hospital_room_unavailable')) or self._as_bool(treatment.get('patient_cannot_be_moved'))):
            decision = 'NOT_ADMISSIBLE'

        return ReasoningState(
            case_id=case_id,
            coverage_findings=coverage_findings,
            waiting_period_findings=waiting_period_findings,
            limit_findings=limit_findings,
            exclusion_findings=exclusion_findings,
            unsupported_claims=unsupported_claims,
            decision=decision,
            citations=citations,
        )


class ValidationAgent:
    """Audit whether the decision is actually grounded in the policy evidence."""

    def validate(self, reasoning: ReasoningState) -> ValidationState:
        unsupported = reasoning.unsupported_claims[:]
        if not reasoning.citations:
            unsupported.append('No policy citations were produced for the decision.')

        if reasoning.decision == 'NOT_ADMISSIBLE':
            return ValidationState(status='PASS', unsupported_claims=unsupported, revision_needed=False)

        if unsupported:
            return ValidationState(status='NEEDS_REVIEW', unsupported_claims=unsupported, revision_needed=True)

        return ValidationState(status='PASS', unsupported_claims=[], revision_needed=False)
