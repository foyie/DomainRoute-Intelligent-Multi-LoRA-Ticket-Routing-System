"""
routing/models.py
──────────────────
Pydantic v2 schemas used across the routing pipeline and serving layer.

Schemas
-------
TicketRequest       – Incoming API request
DomainScore         – Per-domain confidence score from the router
RoutingDecision     – Full router output (domain + all scores + metadata)
LoRASelection       – Which adapter to load + why
GenerationRequest   – Model inference payload
GenerationResponse  – Raw model output
TicketResponse      – Final API response to the caller
EscalationEvent     – Audit log entry for escalated tickets
LatencyBreakdown    – Per-stage latency measurements
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ───────────────────────────────────────────────────────────────

class Domain(str, Enum):
    TECHNICAL  = "technical"
    BILLING    = "billing"
    RETURNS    = "returns"
    ESCALATION = "escalation"
    UNKNOWN    = "unknown"


class ResolutionStatus(str, Enum):
    RESOLVED   = "resolved"
    ESCALATED  = "escalated"
    PENDING    = "pending"
    FAILED     = "failed"


class RoutingMethod(str, Enum):
    SEMANTIC    = "semantic"      # SBERT cosine similarity (primary)
    KEYWORD     = "keyword"       # Fallback: escalation keyword scan
    ZERO_SHOT   = "zero_shot"     # Fallback: zero-shot classification
    DEFAULT     = "default"       # Hard fallback: use configured default domain


# ── Request schemas ────────────────────────────────────────────────────────────

class TicketRequest(BaseModel):
    """Incoming API request payload."""
    ticket_text: str = Field(
        ...,
        min_length=5,
        max_length=4000,
        description="The customer support ticket text",
        examples=["My wireless headphones stopped charging after the firmware update."],
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Prior conversation turns as list of {role, content} dicts",
    )
    customer_id: Optional[str] = Field(
        None,
        description="Optional customer ID for context / repeated-contact detection",
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for request tracing",
    )
    force_domain: Optional[Domain] = Field(
        None,
        description="Override router and force a specific domain (for testing)",
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata (e.g. channel, locale)",
    )

    @field_validator("ticket_text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    @field_validator("conversation_history")
    @classmethod
    def validate_history_roles(cls, v: List[Dict]) -> List[Dict]:
        valid_roles = {"user", "assistant", "system"}
        for turn in v:
            if "role" not in turn or "content" not in turn:
                raise ValueError("Each history turn must have 'role' and 'content'")
            if turn["role"] not in valid_roles:
                raise ValueError(f"Invalid role '{turn['role']}'. Must be one of {valid_roles}")
        return v


# ── Router output schemas ──────────────────────────────────────────────────────

class DomainScore(BaseModel):
    """Confidence score for a single domain."""
    domain: Domain
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence in [0, 1]")
    rank: int    = Field(..., ge=1, description="Rank among all domains (1 = highest)")


class RoutingDecision(BaseModel):
    """Full output from the intent router."""
    primary_domain: Domain
    primary_score: float      = Field(..., ge=0.0, le=1.0)
    all_scores: List[DomainScore]
    routing_method: RoutingMethod = RoutingMethod.SEMANTIC
    escalation_detected: bool     = False
    escalation_score: float       = Field(0.0, ge=0.0, le=1.0)
    is_confident: bool            = True    # True if primary_score ≥ threshold
    fallback_triggered: bool      = False
    router_latency_ms: float      = 0.0
    ticket_length: int            = 0
    timestamp: datetime           = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def scores_sum_to_one(self) -> "RoutingDecision":
        total = sum(s.score for s in self.all_scores)
        if self.all_scores and abs(total - 1.0) > 0.05:
            # Soft warning — don't reject, just normalise internally
            pass
        return self

    @property
    def runner_up_domain(self) -> Optional[Domain]:
        sorted_scores = sorted(self.all_scores, key=lambda s: s.score, reverse=True)
        return sorted_scores[1].domain if len(sorted_scores) > 1 else None

    @property
    def confidence_gap(self) -> float:
        """Gap between top-2 domain scores (higher = more certain routing)."""
        sorted_scores = sorted(self.all_scores, key=lambda s: s.score, reverse=True)
        if len(sorted_scores) >= 2:
            return sorted_scores[0].score - sorted_scores[1].score
        return self.primary_score


class LoRASelection(BaseModel):
    """Which LoRA adapter was selected and why."""
    domain: Domain
    adapter_path: str
    lora_rank: int
    selection_reason: str       # human-readable rationale
    confidence: float           = Field(..., ge=0.0, le=1.0)
    fallback_used: bool         = False
    composition_domains: List[Domain] = Field(
        default_factory=list,
        description="If composed, which LoRAs were blended",
    )
    composition_weights: List[float] = Field(
        default_factory=list,
        description="Weights for each LoRA in the composition",
    )

    @model_validator(mode="after")
    def validate_composition(self) -> "LoRASelection":
        if self.composition_domains and self.composition_weights:
            if len(self.composition_domains) != len(self.composition_weights):
                raise ValueError(
                    "composition_domains and composition_weights must have equal length"
                )
        return self


# ── Generation schemas ─────────────────────────────────────────────────────────

class GenerationRequest(BaseModel):
    """Payload sent to the LLM inference layer."""
    prompt: str
    domain: Domain
    max_new_tokens: int   = Field(256, ge=1, le=1024)
    temperature: float    = Field(0.3, ge=0.0, le=2.0)
    top_p: float          = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    do_sample: bool       = True
    adapter_path: Optional[str] = None


class GenerationResponse(BaseModel):
    """Raw output from the LLM layer."""
    text: str
    domain: Domain
    tokens_generated: int  = 0
    generation_latency_ms: float = 0.0
    finish_reason: str     = "stop"   # "stop" | "length" | "error"


# ── Final response schema ──────────────────────────────────────────────────────

class LatencyBreakdown(BaseModel):
    """Per-stage latency for a single request (all in milliseconds)."""
    router_ms:      float = 0.0
    lora_load_ms:   float = 0.0
    generation_ms:  float = 0.0
    safety_ms:      float = 0.0
    total_ms:       float = 0.0

    def compute_total(self) -> "LatencyBreakdown":
        self.total_ms = self.router_ms + self.lora_load_ms + self.generation_ms + self.safety_ms
        return self


class TicketResponse(BaseModel):
    """Final API response returned to the caller."""
    response_text: str
    domain: Domain
    resolution_status: ResolutionStatus
    routing_decision: RoutingDecision
    lora_selection: LoRASelection
    latency: LatencyBreakdown
    escalation_detected: bool    = False
    escalation_reason: Optional[str] = None
    case_id: Optional[str]       = None    # Set if escalated, for audit trail
    session_id: Optional[str]    = None
    hallucination_score: float   = Field(0.0, ge=0.0, le=1.0)
    quality_passed: bool         = True
    timestamp: datetime          = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Escalation audit log ───────────────────────────────────────────────────────

class EscalationEvent(BaseModel):
    """Audit log entry written for every escalated ticket."""
    case_id: str
    customer_id: Optional[str]
    session_id: Optional[str]
    ticket_text: str
    escalation_score: float
    escalation_reason: str
    routing_domain: Domain
    routing_confidence: float
    triggered_signals: List[str] = Field(
        default_factory=list,
        description="Which escalation signals fired (anger, threat, repeated_contact, etc.)",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    resolved_by: Optional[str] = None    # Human agent ID once resolved

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Router calibration metrics ─────────────────────────────────────────────────

class CalibrationResult(BaseModel):
    """Expected Calibration Error (ECE) measurement for the router."""
    ece: float          = Field(..., ge=0.0, le=1.0, description="Expected Calibration Error")
    n_bins: int         = 10
    domain: Optional[Domain] = None
    n_samples: int      = 0
    accuracy: float     = 0.0
    avg_confidence: float = 0.0
    bin_accuracies: List[float] = Field(default_factory=list)
    bin_confidences: List[float] = Field(default_factory=list)
    bin_counts: List[int] = Field(default_factory=list)
