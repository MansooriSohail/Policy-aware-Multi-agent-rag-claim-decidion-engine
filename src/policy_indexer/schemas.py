from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """A planned evidence query generated from the claim facts."""

    query: str
    intent: str
    priority: int = 1


class EvidenceCitation(BaseModel):
    """Policy-backed evidence reference for a claim finding."""

    chunk_id: str
    section: str
    heading: Optional[str] = None
    page_start: int
    page_end: int
    source: str = "policy.pdf"
    snippet: str


class AnalysisState(BaseModel):
    """Structured output from the analysis stage."""

    case_id: str
    patient_summary: Dict[str, Any] = Field(default_factory=dict)
    claim_summary: Dict[str, Any] = Field(default_factory=dict)
    missing_fields: List[str] = Field(default_factory=list)
    retrieval_queries: List[RetrievalQuery] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """One item returned from hybrid retrieval, with evidence metadata."""

    chunk_id: str
    text: str
    section: str
    heading: Optional[str] = None
    page_start: int
    page_end: int
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0


class ReasoningState(BaseModel):
    """Structured reasoning output from the policy coverage analysis."""

    case_id: str
    coverage_findings: List[str] = Field(default_factory=list)
    waiting_period_findings: List[str] = Field(default_factory=list)
    limit_findings: List[str] = Field(default_factory=list)
    exclusion_findings: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    decision: Optional[str] = None
    citations: List[EvidenceCitation] = Field(default_factory=list)


class ValidationState(BaseModel):
    """Validation result confirming whether the decision is grounded in policy evidence."""

    status: Literal["PASS", "FAIL", "NEEDS_REVIEW"]
    unsupported_claims: List[str] = Field(default_factory=list)
    revision_needed: bool = False


class DecisionOutput(BaseModel):
    """Final machine-readable decision contract returned by the workflow."""

    case_id: str
    decision: Literal[
        "ADMISSIBLE",
        "ADMISSIBLE_WITH_LIMITS",
        "PARTIALLY_ADMISSIBLE",
        "NOT_ADMISSIBLE",
        "NEEDS_REVIEW",
    ]
    confidence: float = 0.0
    key_findings: List[str] = Field(default_factory=list)
    applicable_limits: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    citations: List[EvidenceCitation] = Field(default_factory=list)
    validation: ValidationState
