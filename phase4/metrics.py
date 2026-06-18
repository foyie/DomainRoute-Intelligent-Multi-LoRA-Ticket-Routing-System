"""
evaluation/metrics.py
──────────────────────
Domain-aware evaluation metrics for VeriTune.

Goes beyond generic BLEU/ROUGE — everything is business-aligned:
  auto_resolution_rate    – Did the ticket resolve without human escalation?
  escalation_metrics      – Sensitivity, specificity, FNR, FPR (safety critical)
  latency_metrics         – p50/p95/p99 against <200ms SLA target
  cost_metrics            – Per-domain and aggregate cost per ticket
  routing_accuracy        – Domain classification accuracy + per-class breakdown
  composite_score         – Weighted aggregate for checkpoint selection
  domain_evaluation       – Full per-domain evaluation from a batch of responses

Portfolio note: "Designed 8 custom evaluation metrics replacing generic BLEU
with business-aligned measures — auto-resolution, escalation FNR, latency SLA"

Public API
----------
EvaluationResult        – Complete evaluation output dataclass
DomainMetrics           – Per-domain metric bundle
auto_resolution_rate(predictions, labels)          → float
escalation_metrics(esc_preds, esc_labels)          → EscalationMetrics
latency_metrics(latencies_ms)                      → LatencyMetrics
cost_metrics(domain_counts, cost_map)              → CostMetrics
routing_accuracy(routing_decisions, true_labels)   → RoutingMetrics
composite_score(eval_result, weights)              → float
domain_evaluation(responses, ground_truth, cfg)   → EvaluationResult
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from routing.models import Domain

logger = logging.getLogger(__name__)

DOMAINS = [Domain.TECHNICAL, Domain.BILLING, Domain.RETURNS, Domain.ESCALATION]

# Cost model: estimated $ per ticket by domain (LoRA rank-proportional)
DEFAULT_COST_MAP: Dict[Domain, float] = {
    Domain.TECHNICAL:  0.05,
    Domain.BILLING:    0.12,
    Domain.RETURNS:    0.09,
    Domain.ESCALATION: 0.15,
}

BASELINE_COST_PER_TICKET = 0.50     # full model, no routing
LATENCY_SLA_MS           = 200.0    # p95 target from spec
ESCALATION_FN_BUDGET     = 0.005    # max 0.5% false negative rate


# ── Sub-metric dataclasses ─────────────────────────────────────────────────────

@dataclass
class EscalationMetrics:
    """Safety-critical escalation detection metrics."""
    sensitivity: float       # true positive rate  (recall on escalation class)
    specificity: float       # true negative rate
    false_negative_rate: float  # 1 - sensitivity  (MUST be near 0)
    false_positive_rate: float  # 1 - specificity
    precision: float
    f1: float
    n_true_escalations: int
    n_predicted_escalations: int
    n_total: int
    meets_safety_budget: bool  # FNR < ESCALATION_FN_BUDGET

    def summary(self) -> str:
        status = "✓ PASS" if self.meets_safety_budget else "✗ FAIL"
        return (
            f"Escalation  sensitivity={self.sensitivity:.4f}  "
            f"specificity={self.specificity:.4f}  "
            f"FNR={self.false_negative_rate:.4f}  "
            f"[{status}]"
        )


@dataclass
class LatencyMetrics:
    """Inference latency distribution metrics."""
    p50_ms:   float
    p95_ms:   float
    p99_ms:   float
    mean_ms:  float
    max_ms:   float
    n_requests: int
    meets_sla: bool      # p95 < LATENCY_SLA_MS

    def summary(self) -> str:
        status = "✓ PASS" if self.meets_sla else "✗ FAIL"
        return (
            f"Latency  p50={self.p50_ms:.0f}ms  "
            f"p95={self.p95_ms:.0f}ms  "
            f"p99={self.p99_ms:.0f}ms  [{status}]"
        )


@dataclass
class CostMetrics:
    """Cost per ticket analysis."""
    avg_cost_per_ticket: float
    total_cost: float
    cost_by_domain: Dict[str, float]
    baseline_cost_per_ticket: float = BASELINE_COST_PER_TICKET
    cost_reduction_pct: float = 0.0
    n_tickets: int = 0

    def __post_init__(self) -> None:
        if self.baseline_cost_per_ticket > 0:
            self.cost_reduction_pct = (
                1.0 - self.avg_cost_per_ticket / self.baseline_cost_per_ticket
            ) * 100


@dataclass
class RoutingMetrics:
    """Intent router accuracy metrics."""
    overall_accuracy: float
    per_domain_accuracy: Dict[str, float]
    per_domain_support: Dict[str, int]
    macro_f1: float
    weighted_f1: float
    n_samples: int
    confusion_matrix: Optional[List[List[int]]] = None


@dataclass
class DomainMetrics:
    """All metrics for a single domain."""
    domain: str
    auto_resolution_rate: float
    n_samples: int
    n_resolved: int
    n_escalated: int
    avg_response_length: float = 0.0
    avg_latency_ms: float = 0.0
    cost_per_ticket: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation output — one per checkpoint or run."""
    run_name: str
    auto_resolution_rate: float          # overall
    escalation: EscalationMetrics
    latency: LatencyMetrics
    cost: CostMetrics
    routing: Optional[RoutingMetrics]
    per_domain: Dict[str, DomainMetrics]
    hallucination_rate: float = 0.0      # from hallucination_detector
    composite_score: float    = 0.0      # from composite_score()
    n_total: int              = 0
    timestamp: str            = ""

    def passes_all_gates(self) -> bool:
        """Return True if all hard safety + SLA requirements are met."""
        return (
            self.escalation.meets_safety_budget
            and self.latency.meets_sla
            and self.hallucination_rate < 0.05   # < 5% hallucination
        )

    def summary(self) -> str:
        lines = [
            f"Run: {self.run_name}",
            f"  Auto-resolution rate : {self.auto_resolution_rate:.3f}",
            f"  {self.escalation.summary()}",
            f"  {self.latency.summary()}",
            f"  Cost/ticket          : ${self.cost.avg_cost_per_ticket:.3f} "
            f"(-{self.cost.cost_reduction_pct:.0f}% vs baseline)",
            f"  Hallucination rate   : {self.hallucination_rate:.3f}",
            f"  Composite score      : {self.composite_score:.4f}",
            f"  Gates passed         : {'YES ✓' if self.passes_all_gates() else 'NO ✗'}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert Domain enum keys to strings
        d["per_domain"] = {k: asdict(v) for k, v in self.per_domain.items()}
        return d

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("EvaluationResult saved → %s", path)


# ── Core metric functions ──────────────────────────────────────────────────────

def auto_resolution_rate(
    predictions: List[str],
    labels: List[str],
    resolved_label: str = "resolved",
) -> float:
    """
    Fraction of tickets predicted as 'resolved' that are truly resolved.
    This is the primary business metric (target: >94%).
    """
    if not predictions:
        return 0.0
    correctly_resolved = sum(
        1 for p, l in zip(predictions, labels)
        if p == resolved_label and l == resolved_label
    )
    total_truly_resolved = sum(1 for l in labels if l == resolved_label)
    return correctly_resolved / max(total_truly_resolved, 1)


def escalation_metrics(
    esc_predictions: List[bool],
    esc_labels: List[bool],
) -> EscalationMetrics:
    """
    Compute safety-critical escalation detection metrics.

    Parameters
    ----------
    esc_predictions : List[bool] — True if model predicted escalation
    esc_labels      : List[bool] — True if ticket truly requires escalation

    Returns EscalationMetrics with sensitivity, specificity, FNR, etc.
    """
    n = len(esc_predictions)
    if n == 0:
        return EscalationMetrics(0, 0, 1, 1, 0, 0, 0, 0, 0, False)

    tp = sum(1 for p, l in zip(esc_predictions, esc_labels) if p and l)
    tn = sum(1 for p, l in zip(esc_predictions, esc_labels) if not p and not l)
    fp = sum(1 for p, l in zip(esc_predictions, esc_labels) if p and not l)
    fn = sum(1 for p, l in zip(esc_predictions, esc_labels) if not p and l)

    n_true_esc = tp + fn
    n_pred_esc = tp + fp

    sensitivity = tp / max(n_true_esc, 1)      # recall
    specificity = tn / max(tn + fp, 1)
    precision   = tp / max(n_pred_esc, 1)
    fnr         = fn / max(n_true_esc, 1)
    fpr         = fp / max(tn + fp, 1)
    f1          = (2 * precision * sensitivity / max(precision + sensitivity, 1e-9))

    return EscalationMetrics(
        sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4),
        false_negative_rate=round(fnr, 4),
        false_positive_rate=round(fpr, 4),
        precision=round(precision, 4),
        f1=round(f1, 4),
        n_true_escalations=n_true_esc,
        n_predicted_escalations=n_pred_esc,
        n_total=n,
        meets_safety_budget=fnr < ESCALATION_FN_BUDGET,
    )


def latency_metrics(latencies_ms: List[float]) -> LatencyMetrics:
    """
    Compute latency distribution metrics from a list of per-request latencies.
    """
    if not latencies_ms:
        return LatencyMetrics(0, 0, 0, 0, 0, 0, False)

    arr = np.array(latencies_ms, dtype=float)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    return LatencyMetrics(
        p50_ms=round(p50, 1),
        p95_ms=round(p95, 1),
        p99_ms=round(p99, 1),
        mean_ms=round(float(arr.mean()), 1),
        max_ms=round(float(arr.max()), 1),
        n_requests=len(latencies_ms),
        meets_sla=p95 < LATENCY_SLA_MS,
    )


def cost_metrics(
    domain_counts: Dict[str, int],
    cost_map: Optional[Dict[Domain, float]] = None,
) -> CostMetrics:
    """
    Compute cost metrics given per-domain request counts.

    Parameters
    ----------
    domain_counts : {domain_name: n_requests}
    cost_map      : {Domain: cost_per_ticket} — defaults to DEFAULT_COST_MAP
    """
    cost_map = cost_map or DEFAULT_COST_MAP
    cost_by_domain: Dict[str, float] = {}
    total_cost = 0.0
    total_tickets = 0

    for domain_str, count in domain_counts.items():
        try:
            domain = Domain(domain_str)
        except ValueError:
            continue
        unit_cost = cost_map.get(domain, 0.10)
        domain_total = unit_cost * count
        cost_by_domain[domain_str] = round(unit_cost, 4)
        total_cost += domain_total
        total_tickets += count

    avg_cost = total_cost / max(total_tickets, 1)

    return CostMetrics(
        avg_cost_per_ticket=round(avg_cost, 4),
        total_cost=round(total_cost, 4),
        cost_by_domain=cost_by_domain,
        n_tickets=total_tickets,
    )


def routing_accuracy(
    routing_decisions: List,              # List[RoutingDecision]
    true_labels: List[Domain],
) -> RoutingMetrics:
    """
    Compute routing accuracy metrics from a list of RoutingDecisions.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix as sk_cm,
        precision_recall_fscore_support,
    )

    predicted = [d.primary_domain for d in routing_decisions]
    domain_list = [d.value for d in DOMAINS]

    pred_strs  = [d.value for d in predicted]
    label_strs = [d.value for d in true_labels]

    overall_acc = accuracy_score(label_strs, pred_strs)
    macro_f1    = f1_score(label_strs, pred_strs, average="macro",  zero_division=0)
    weighted_f1 = f1_score(label_strs, pred_strs, average="weighted", zero_division=0)

    _, _, _, support = precision_recall_fscore_support(
        label_strs, pred_strs, labels=domain_list, zero_division=0
    )
    per_domain_f1s = f1_score(
        label_strs, pred_strs, labels=domain_list, average=None, zero_division=0
    )

    per_domain_acc: Dict[str, float] = {}
    per_domain_sup: Dict[str, int]   = {}
    for domain, f1, sup in zip(domain_list, per_domain_f1s, support):
        domain_mask = [l == domain for l in label_strs]
        domain_preds = [p for p, m in zip(pred_strs, domain_mask) if m]
        domain_true  = [l for l, m in zip(label_strs, domain_mask) if m]
        per_domain_acc[domain] = (
            accuracy_score(domain_true, domain_preds) if domain_true else 0.0
        )
        per_domain_sup[domain] = int(sup)

    cm = sk_cm(label_strs, pred_strs, labels=domain_list).tolist()

    return RoutingMetrics(
        overall_accuracy=round(overall_acc, 4),
        per_domain_accuracy={k: round(v, 4) for k, v in per_domain_acc.items()},
        per_domain_support=per_domain_sup,
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        n_samples=len(routing_decisions),
        confusion_matrix=cm,
    )


def composite_score(
    result: EvaluationResult,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute a single composite score for checkpoint selection.

    Default weights (tuned for business priorities):
      auto_resolution_rate : 0.35
      escalation_sensitivity: 0.30  (safety critical — high weight)
      latency_sla_pass     : 0.15
      cost_efficiency      : 0.10
      low_hallucination    : 0.10
    """
    weights = weights or {
        "auto_resolution":    0.35,
        "escalation_sens":    0.30,
        "latency_sla":        0.15,
        "cost_efficiency":    0.10,
        "low_hallucination":  0.10,
    }

    arr_score = result.auto_resolution_rate
    esc_score = result.escalation.sensitivity
    lat_score = 1.0 if result.latency.meets_sla else max(
        0, 1.0 - (result.latency.p95_ms - LATENCY_SLA_MS) / LATENCY_SLA_MS
    )
    # Cost: higher cost reduction = higher score
    cost_score = min(result.cost.cost_reduction_pct / 100.0, 1.0)
    # Hallucination: 0% = 1.0, 10%+ = 0.0
    hall_score = max(0.0, 1.0 - result.hallucination_rate * 10)

    score = (
        weights["auto_resolution"]   * arr_score +
        weights["escalation_sens"]   * esc_score +
        weights["latency_sla"]       * lat_score +
        weights["cost_efficiency"]   * cost_score +
        weights["low_hallucination"] * hall_score
    )
    return round(float(score), 4)


# ── Full domain evaluation ─────────────────────────────────────────────────────

def domain_evaluation(
    responses: List[Dict],
    ground_truth: List[Dict],
    latencies_ms: Optional[List[float]] = None,
    routing_decisions: Optional[List] = None,
    run_name: str = "evaluation",
    cfg: Optional[dict] = None,
) -> EvaluationResult:
    """
    Run full evaluation over a batch of model responses vs ground truth.

    Parameters
    ----------
    responses      : [{text, domain, label, response}] — model outputs
    ground_truth   : [{text, domain, label, response}] — reference outputs
    latencies_ms   : Per-request latency measurements
    routing_decisions : RoutingDecision objects (for routing accuracy)
    run_name       : Label for this evaluation run

    Returns
    -------
    EvaluationResult with all metrics populated
    """
    from datetime import datetime

    if len(responses) != len(ground_truth):
        raise ValueError(
            f"responses ({len(responses)}) and ground_truth ({len(ground_truth)}) "
            f"must have the same length"
        )

    n = len(responses)
    logger.info("Running domain evaluation on %d examples (run='%s')", n, run_name)

    # ── Resolution predictions ─────────────────────────────────────────────────
    pred_labels  = [r.get("label", "resolved") for r in responses]
    true_labels  = [g.get("label", "resolved") for g in ground_truth]
    arr          = auto_resolution_rate(pred_labels, true_labels)

    # ── Escalation metrics ─────────────────────────────────────────────────────
    esc_preds  = [r.get("label") == "escalate" for r in responses]
    esc_labels = [g.get("label") == "escalate" for g in ground_truth]
    esc        = escalation_metrics(esc_preds, esc_labels)

    # ── Latency ────────────────────────────────────────────────────────────────
    lat = latency_metrics(latencies_ms or [])

    # ── Cost ───────────────────────────────────────────────────────────────────
    domain_counts: Dict[str, int] = {}
    for r in responses:
        d = r.get("domain", "technical")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    cost = cost_metrics(domain_counts)

    # ── Routing accuracy ───────────────────────────────────────────────────────
    routing = None
    if routing_decisions:
        true_domains = []
        for g in ground_truth:
            try:
                true_domains.append(Domain(g.get("domain", "technical")))
            except ValueError:
                true_domains.append(Domain.TECHNICAL)
        routing = routing_accuracy(routing_decisions, true_domains)

    # ── Per-domain metrics ─────────────────────────────────────────────────────
    per_domain: Dict[str, DomainMetrics] = {}
    domain_groups: Dict[str, List] = {}
    for r, g in zip(responses, ground_truth):
        d = g.get("domain", "technical")
        domain_groups.setdefault(d, []).append((r, g))

    for domain_str, pairs in domain_groups.items():
        dom_preds  = [p.get("label", "resolved") for p, _ in pairs]
        dom_labels = [g.get("label", "resolved") for _, g in pairs]
        dom_arr    = auto_resolution_rate(dom_preds, dom_labels)
        n_dom      = len(pairs)
        n_resolved = sum(1 for l in dom_labels if l == "resolved")
        n_escalated = n_dom - n_resolved
        avg_len = (
            np.mean([len(p.get("response", "")) for p, _ in pairs])
            if pairs else 0.0
        )
        per_domain[domain_str] = DomainMetrics(
            domain=domain_str,
            auto_resolution_rate=round(dom_arr, 4),
            n_samples=n_dom,
            n_resolved=n_resolved,
            n_escalated=n_escalated,
            avg_response_length=round(float(avg_len), 1),
            cost_per_ticket=DEFAULT_COST_MAP.get(
                Domain(domain_str) if domain_str in [d.value for d in Domain]
                else Domain.TECHNICAL, 0.10
            ),
        )

    result = EvaluationResult(
        run_name=run_name,
        auto_resolution_rate=round(arr, 4),
        escalation=esc,
        latency=lat,
        cost=cost,
        routing=routing,
        per_domain=per_domain,
        n_total=n,
        timestamp=datetime.utcnow().isoformat(),
    )
    result.composite_score = composite_score(result)

    logger.info(result.summary())
    return result


# ── Tone / compliance heuristics ───────────────────────────────────────────────

def tone_score(response_text: str) -> float:
    """
    Heuristic empathy / tone score in [0, 1].
    Used as a soft signal alongside LLM-as-judge.
    """
    text_lower = response_text.lower()
    empathy_phrases = [
        "sorry", "apologise", "apologies", "understand", "frustrat",
        "thank you", "appreciate", "help you", "i can see",
    ]
    action_phrases = [
        "i will", "i've", "processing", "arranging", "refunding",
        "replacing", "escalating", "looking into",
    ]
    empathy_count = sum(1 for p in empathy_phrases if p in text_lower)
    action_count  = sum(1 for p in action_phrases  if p in text_lower)
    return min((empathy_count * 0.15 + action_count * 0.20), 1.0)


def compliance_check(response_text: str, domain: str) -> Dict[str, bool]:
    """
    Heuristic domain-specific compliance check.
    Returns {rule_name: passed} dict.
    """
    text_lower = response_text.lower()
    rules: Dict[str, bool] = {"non_empty": bool(response_text.strip())}

    if domain == "billing":
        rules["mentions_account"]   = any(w in text_lower for w in ["account", "charge", "payment", "refund"])
        rules["no_pii_exposure"]    = "password" not in text_lower and "ssn" not in text_lower
        rules["action_stated"]      = any(w in text_lower for w in ["i will", "i've", "processed", "refund"])

    elif domain == "escalation":
        rules["contains_apology"]   = any(w in text_lower for w in ["sorry", "apologise", "apologies"])
        rules["provides_case_id"]   = any(w in text_lower for w in ["case", "ticket", "reference", "id"])
        rules["escalates"]          = any(w in text_lower for w in ["specialist", "manager", "escalat", "priority"])

    elif domain == "technical":
        rules["actionable_steps"]   = any(w in text_lower for w in ["step", "try", "restart", "reset", "check"])

    elif domain == "returns":
        rules["return_guidance"]    = any(w in text_lower for w in ["return", "label", "ship", "exchange"])

    return rules
