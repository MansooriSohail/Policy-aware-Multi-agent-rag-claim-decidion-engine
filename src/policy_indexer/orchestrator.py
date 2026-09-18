from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from policy_indexer.agents import AnalysisAgent, RetrievalAgent, ReasoningAgent, ValidationAgent
from policy_indexer.pipeline import build_policy_index
from policy_indexer.schemas import DecisionOutput


class PolicyDecisionWorkflow:
    """Coordinate the multi-agent policy decision pipeline."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self.index_data = build_policy_index(self.pdf_path)
        self.analysis_agent = AnalysisAgent()
        self.retrieval_agent = RetrievalAgent(self.index_data)
        self.reasoning_agent = ReasoningAgent()
        self.validation_agent = ValidationAgent()

    def build_decision(self, case: Dict[str, Any]) -> DecisionOutput:
        """Return the final policy decision payload expected by the API contract."""
        analysis = self.analysis_agent.analyze(case)
        retrieval = self.retrieval_agent.retrieve(analysis)
        reasoning = self.reasoning_agent.reason(case, retrieval)
        validation = self.validation_agent.validate(reasoning)

        decision = reasoning.decision or 'NEEDS_REVIEW'
        if validation.status == 'NEEDS_REVIEW':
            decision = 'NEEDS_REVIEW'

        return DecisionOutput(
            case_id=str(case.get('case_id', 'UNKNOWN')),
            decision=decision,
            confidence=0.75 if validation.status == 'PASS' else 0.35,
            key_findings=reasoning.coverage_findings + reasoning.waiting_period_findings + reasoning.limit_findings + reasoning.exclusion_findings,
            applicable_limits=reasoning.limit_findings,
            missing_evidence=analysis.missing_fields,
            citations=reasoning.citations,
            validation=validation,
        )

    def analyze_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full structured workflow for one claim case."""
        analysis = self.analysis_agent.analyze(case)
        retrieval = self.retrieval_agent.retrieve(analysis)
        reasoning = self.reasoning_agent.reason(case, retrieval)
        validation = self.validation_agent.validate(reasoning)

        decision = self.build_decision(case)

        return {
            'case_id': decision.case_id,
            'decision': decision.decision,
            'analysis': analysis.model_dump(),
            'retrieval': [item.model_dump() for item in retrieval],
            'reasoning': reasoning.model_dump(),
            'validation': validation.model_dump(),
            'citations': decision.citations,
            'confidence': decision.confidence,
            'key_findings': decision.key_findings,
            'applicable_limits': decision.applicable_limits,
            'missing_evidence': decision.missing_evidence,
        }
