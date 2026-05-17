"""
routing/lora_selector.py
─────────────────────────
Translates a RoutingDecision into a LoRASelection: which adapter to load,
whether to escalate immediately, and whether to compose multiple LoRAs.

Selection logic
---------------
1. If escalation_detected → always select escalation LoRA (safety override)
2. If primary_score ≥ domain threshold → select that domain's LoRA
3. If primary_score < threshold (low confidence):
     a. If runner-up score is close → consider composition
     b. Otherwise → fall back to best domain with escalation safety check
4. Log every selection decision for audit trail

Composition heuristic
---------------------
When top-2 domain scores are within `composition_threshold` of each other,
blend both LoRAs weighted by their normalised confidence scores. This handles
ambiguous tickets (e.g. "billing issue with a returned item" → billing+returns).

Public API
----------
LoRASelector
  .select(routing_decision)            → LoRASelection
  .select_with_escalation_check(...)   → (LoRASelection, bool)  # (selection, should_escalate)
  .update_adapter_registry(registry)   → None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from routing.models import Domain, LoRASelection, RoutingDecision

logger = logging.getLogger(__name__)

# Default adapter paths — overridden by registry at runtime
DEFAULT_ADAPTER_PATHS: Dict[Domain, str] = {
    Domain.TECHNICAL:  "outputs/checkpoints/technical_best",
    Domain.BILLING:    "outputs/checkpoints/billing_best",
    Domain.RETURNS:    "outputs/checkpoints/returns_best",
    Domain.ESCALATION: "outputs/checkpoints/escalation_best",
}

DEFAULT_LORA_RANKS: Dict[Domain, int] = {
    Domain.TECHNICAL:  32,
    Domain.BILLING:    24,
    Domain.RETURNS:    28,
    Domain.ESCALATION: 8,
}

# Domain-specific confidence thresholds for auto-selection
SELECTION_THRESHOLDS: Dict[Domain, float] = {
    Domain.TECHNICAL:  0.70,
    Domain.BILLING:    0.70,
    Domain.RETURNS:    0.70,
    Domain.ESCALATION: 0.50,   # safety critical — lower threshold
}

# Below this threshold → consider immediate escalation regardless of domain
GLOBAL_LOW_CONFIDENCE_ESCALATION_THRESHOLD = 0.40


@dataclass
class AdapterRegistry:
    """Maps domain → adapter path and rank. Updated when new checkpoints are promoted."""
    paths: Dict[Domain, str]
    ranks: Dict[Domain, int]

    @classmethod
    def default(cls) -> "AdapterRegistry":
        return cls(
            paths=dict(DEFAULT_ADAPTER_PATHS),
            ranks=dict(DEFAULT_LORA_RANKS),
        )

    @classmethod
    def from_checkpoint_manager(cls, manager) -> "AdapterRegistry":
        """Build registry from a fitted CheckpointManager."""
        paths: Dict[Domain, str] = {}
        ranks: Dict[Domain, int] = {}
        for domain_str, record in manager.best_across_all_domains().items():
            try:
                domain = Domain(domain_str)
                paths[domain] = record.adapter_path
                ranks[domain] = DEFAULT_LORA_RANKS.get(domain, 32)
            except (ValueError, AttributeError):
                pass
        # Fill missing domains with defaults
        for d in Domain:
            if d not in paths and d != Domain.UNKNOWN:
                paths[d] = DEFAULT_ADAPTER_PATHS.get(d, "")
                ranks[d] = DEFAULT_LORA_RANKS.get(d, 32)
        return cls(paths=paths, ranks=ranks)


class LoRASelector:
    """
    Translates a RoutingDecision into a concrete LoRASelection.

    Parameters
    ----------
    registry              : AdapterRegistry mapping domain → path + rank
    composition_threshold : If top-2 scores differ by < this, compose both adapters
    escalation_override   : Always check escalation LoRA even for non-escalation domains
    """

    def __init__(
        self,
        registry: Optional[AdapterRegistry] = None,
        composition_threshold: float = 0.15,
        escalation_override: bool = True,
    ) -> None:
        self.registry              = registry or AdapterRegistry.default()
        self.composition_threshold = composition_threshold
        self.escalation_override   = escalation_override

    def update_adapter_registry(self, registry: AdapterRegistry) -> None:
        """Hot-reload adapter paths without restarting the server."""
        self.registry = registry
        logger.info(
            "AdapterRegistry updated: %d domains",
            len(registry.paths),
        )

    def select(self, routing_decision: RoutingDecision) -> LoRASelection:
        """
        Select the appropriate LoRA adapter for a routing decision.

        Returns LoRASelection with the chosen adapter path, rank, and rationale.
        """
        # ── Safety override: escalation always wins ────────────────────────────
        if routing_decision.escalation_detected:
            return self._build_selection(
                domain=Domain.ESCALATION,
                routing_decision=routing_decision,
                reason=(
                    f"Escalation detected (score={routing_decision.escalation_score:.3f}). "
                    f"Safety override — routing to escalation adapter regardless of primary domain."
                ),
            )

        primary = routing_decision.primary_domain
        primary_score = routing_decision.primary_score
        threshold = SELECTION_THRESHOLDS.get(primary, 0.70)

        # ── Very low confidence → escalate to human ───────────────────────────
        if primary_score < GLOBAL_LOW_CONFIDENCE_ESCALATION_THRESHOLD:
            logger.warning(
                "Very low routing confidence (%.3f) — escalation safety fallback",
                primary_score,
            )
            return self._build_selection(
                domain=Domain.ESCALATION,
                routing_decision=routing_decision,
                reason=(
                    f"Primary confidence too low ({primary_score:.3f} < "
                    f"{GLOBAL_LOW_CONFIDENCE_ESCALATION_THRESHOLD}). "
                    f"Routing to escalation for human review."
                ),
                fallback=True,
            )

        # ── Confident single-domain selection ─────────────────────────────────
        if primary_score >= threshold:
            # Check if we should compose with runner-up
            runner_up = routing_decision.runner_up_domain
            if (
                runner_up
                and runner_up != Domain.ESCALATION
                and routing_decision.confidence_gap < self.composition_threshold
            ):
                return self._build_composition(
                    primary=primary,
                    secondary=runner_up,
                    routing_decision=routing_decision,
                )

            return self._build_selection(
                domain=primary,
                routing_decision=routing_decision,
                reason=(
                    f"Confident routing to {primary.value} "
                    f"(score={primary_score:.3f} ≥ threshold={threshold:.2f})."
                ),
            )

        # ── Below threshold but not critical low ───────────────────────────────
        # Use primary domain but flag as uncertain
        logger.info(
            "Below-threshold routing: domain=%s score=%.3f threshold=%.2f — using primary with fallback flag",
            primary, primary_score, threshold,
        )
        return self._build_selection(
            domain=primary,
            routing_decision=routing_decision,
            reason=(
                f"Below-threshold routing to {primary.value} "
                f"(score={primary_score:.3f} < threshold={threshold:.2f}). "
                f"Proceeding with fallback flag set."
            ),
            fallback=True,
        )

    def select_with_escalation_check(
        self,
        routing_decision: RoutingDecision,
    ) -> Tuple[LoRASelection, bool]:
        """
        Select LoRA and return a tuple of (selection, should_escalate_to_human).

        should_escalate_to_human is True when:
        - Escalation LoRA is selected AND escalation_score > 0.70
        - Very low confidence on any domain
        """
        selection = self.select(routing_decision)
        should_escalate = (
            selection.domain == Domain.ESCALATION
            and routing_decision.escalation_score >= 0.70
        ) or selection.fallback_used and routing_decision.primary_score < GLOBAL_LOW_CONFIDENCE_ESCALATION_THRESHOLD

        return selection, should_escalate

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_selection(
        self,
        domain: Domain,
        routing_decision: RoutingDecision,
        reason: str,
        fallback: bool = False,
    ) -> LoRASelection:
        path = self.registry.paths.get(domain, DEFAULT_ADAPTER_PATHS.get(domain, ""))
        rank = self.registry.ranks.get(domain, DEFAULT_LORA_RANKS.get(domain, 32))

        return LoRASelection(
            domain=domain,
            adapter_path=str(path),
            lora_rank=rank,
            selection_reason=reason,
            confidence=routing_decision.primary_score,
            fallback_used=fallback,
        )

    def _build_composition(
        self,
        primary: Domain,
        secondary: Domain,
        routing_decision: RoutingDecision,
    ) -> LoRASelection:
        """
        Blend two domain LoRAs weighted by their normalised confidence scores.
        """
        scores = {s.domain: s.score for s in routing_decision.all_scores}
        p_score = scores.get(primary,   0.0)
        s_score = scores.get(secondary, 0.0)
        total   = p_score + s_score + 1e-9
        w_p     = p_score / total
        w_s     = s_score / total

        logger.info(
            "Composing LoRAs: %s (%.2f) + %s (%.2f) — gap=%.3f",
            primary, w_p, secondary, w_s, routing_decision.confidence_gap,
        )

        return LoRASelection(
            domain=primary,
            adapter_path=self.registry.paths.get(primary, ""),
            lora_rank=self.registry.ranks.get(primary, 32),
            selection_reason=(
                f"Composing {primary.value} ({w_p:.1%}) + {secondary.value} ({w_s:.1%}): "
                f"confidence gap {routing_decision.confidence_gap:.3f} < threshold {self.composition_threshold:.2f}."
            ),
            confidence=routing_decision.primary_score,
            fallback_used=False,
            composition_domains=[primary, secondary],
            composition_weights=[round(w_p, 4), round(w_s, 4)],
        )
