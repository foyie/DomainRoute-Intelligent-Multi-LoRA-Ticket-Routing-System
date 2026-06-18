"""
serving/safety_filters.py
──────────────────────────
Production safety filters applied to every VeriTune response before it is
returned to the caller.

Pipeline (all filters run in order):
  1. EscalationGuard    – Re-check for escalation signals in the ticket
                          (belt-and-suspenders after the LoRA prediction)
  2. HallucinationGuard – Fast heuristic hallucination check
  3. PIIScrubber        – Remove / mask any PII that leaked into the response
  4. ToneChecker        – Flag responses that are too short, robotic, or rude
  5. ComplianceChecker  – Domain-specific rule checks

Each filter returns a FilterResult with a pass/fail flag and optional
override_response for hard failures.

Public API
----------
FilterResult        – Single filter outcome
SafetyPipeline
  .run(ticket, response, domain, escalation_score) → SafetyReport
SafetyReport        – Full pipeline result
EscalationGuard.check(ticket, response, score)     → FilterResult
PIIScrubber.scrub(text)                            → str
ToneChecker.check(response, domain)                → FilterResult
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── PII patterns ───────────────────────────────────────────────────────────────
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),        "[SSN REDACTED]"),
    (re.compile(r"\b4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[CARD REDACTED]"),
    (re.compile(r"\b5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[CARD REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL REDACTED]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE REDACTED]"),
    (re.compile(r"\bpassword\s*[:=]\s*\S+", re.I),             "[PASSWORD REDACTED]"),
]

# ── Escalation keywords for guard ─────────────────────────────────────────────
_ESC_ANGER   = re.compile(r"\b(furious|unacceptable|outraged|disgusted|useless|incompetent)\b", re.I)
_ESC_THREAT  = re.compile(r"\b(chargeback|lawsuit|legal action|sue|BBB|FTC|fraud)\b", re.I)
_ESC_URGENCY = re.compile(r"\b(immediately|right now|NOW)\b", re.I)

# ── Tone patterns ──────────────────────────────────────────────────────────────
_RUDE_PATTERNS = re.compile(
    r"\b(stupid|idiot|dumb|shut up|don't bother|not my problem)\b", re.I
)
_DEFLECTION_PATTERNS = re.compile(
    r"\b(cannot help|unable to assist|please contact|refer to our website)\b", re.I
)

# Escalation canned response
ESCALATION_RESPONSE = (
    "I sincerely apologise for the frustration this has caused. "
    "I'm escalating your case to a senior specialist who will contact you "
    "within the hour. Your case reference is {case_id}. "
    "We take your concerns very seriously."
)


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    filter_name:       str
    passed:            bool
    confidence:        float = 1.0
    reason:            str   = ""
    override_response: Optional[str] = None  # if set, replaces the model output


@dataclass
class SafetyReport:
    """Full safety pipeline result for a single request."""
    ticket:            str
    original_response: str
    final_response:    str
    domain:            str
    filter_results:    List[FilterResult] = field(default_factory=list)
    escalation_triggered: bool = False
    pii_scrubbed:      bool = False
    hallucination_flagged: bool = False
    quality_passed:    bool = True
    case_id:           Optional[str] = None
    safety_latency_ms: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(f.passed for f in self.filter_results)

    def summary(self) -> str:
        results = " | ".join(
            f"{f.filter_name}={'✓' if f.passed else '✗'}" for f in self.filter_results
        )
        return f"SafetyReport[{self.domain}]: {results}"


# ── Individual filters ─────────────────────────────────────────────────────────

class EscalationGuard:
    """
    Belt-and-suspenders escalation check applied AFTER the LoRA prediction.
    Catches any escalation signals that the model may have missed.

    Uses both the upstream escalation_score and a fresh keyword scan.
    """

    ESCALATION_SCORE_THRESHOLD = 0.60

    def check(
        self,
        ticket: str,
        response: str,
        escalation_score: float = 0.0,
        case_id: Optional[str] = None,
    ) -> FilterResult:
        """
        Returns a failed FilterResult if escalation is needed,
        with override_response set to the canned escalation message.
        """
        # Direct score threshold
        if escalation_score >= self.ESCALATION_SCORE_THRESHOLD:
            return self._escalate(
                f"escalation_score={escalation_score:.3f} >= {self.ESCALATION_SCORE_THRESHOLD}",
                case_id,
            )

        # Fresh keyword scan on the ticket
        score = 0.0
        if _ESC_ANGER.search(ticket):   score += 0.35
        if _ESC_THREAT.search(ticket):  score += 0.50
        if _ESC_URGENCY.search(ticket): score += 0.20

        if score >= self.ESCALATION_SCORE_THRESHOLD:
            return self._escalate(
                f"keyword_scan_score={score:.3f}", case_id
            )

        return FilterResult(
            filter_name="EscalationGuard",
            passed=True,
            confidence=1.0 - max(escalation_score, score),
            reason="No escalation signals detected",
        )

    def _escalate(self, reason: str, case_id: Optional[str]) -> FilterResult:
        import uuid
        cid = case_id or f"ESC-{uuid.uuid4().hex[:6].upper()}"
        return FilterResult(
            filter_name="EscalationGuard",
            passed=False,
            confidence=0.95,
            reason=reason,
            override_response=ESCALATION_RESPONSE.format(case_id=cid),
        )


class PIIScrubber:
    """
    Remove / mask PII patterns from model responses.
    Runs on every response to prevent accidental PII leakage.
    """

    def scrub(self, text: str) -> Tuple[str, bool]:
        """
        Scrub PII from text.
        Returns (scrubbed_text, was_modified).
        """
        modified = False
        result   = text
        for pattern, replacement in _PII_PATTERNS:
            new_result = pattern.sub(replacement, result)
            if new_result != result:
                modified = True
                result   = new_result
                logger.warning(
                    "PII detected and scrubbed: pattern=%s", pattern.pattern[:40]
                )
        return result, modified

    def check(self, response: str) -> FilterResult:
        scrubbed, modified = self.scrub(response)
        return FilterResult(
            filter_name="PIIScrubber",
            passed=True,   # scrubbing doesn't block — it modifies
            confidence=1.0,
            reason="PII scrubbed" if modified else "No PII detected",
            override_response=scrubbed if modified else None,
        )


class ToneChecker:
    """
    Check that the response meets minimum quality and tone standards.
    Blocks responses that are:
      - Too short (< 20 words — likely incomplete)
      - Contains rude / inappropriate language
      - Pure deflection with no action
    """

    MIN_WORDS      = 20
    MIN_WORDS_ESC  = 15   # escalation responses can be shorter

    def check(self, response: str, domain: str = "technical") -> FilterResult:
        words = response.split()
        min_w = self.MIN_WORDS_ESC if domain == "escalation" else self.MIN_WORDS

        # Too short
        if len(words) < min_w:
            return FilterResult(
                filter_name="ToneChecker",
                passed=False,
                confidence=0.90,
                reason=f"Response too short ({len(words)} words < {min_w})",
            )

        # Rude language
        if _RUDE_PATTERNS.search(response):
            return FilterResult(
                filter_name="ToneChecker",
                passed=False,
                confidence=0.95,
                reason="Inappropriate language detected",
            )

        # Pure deflection
        if _DEFLECTION_PATTERNS.search(response) and len(words) < 30:
            return FilterResult(
                filter_name="ToneChecker",
                passed=False,
                confidence=0.80,
                reason="Response appears to be a deflection with no action",
            )

        return FilterResult(
            filter_name="ToneChecker",
            passed=True,
            confidence=0.95,
            reason="Tone check passed",
        )


class HallucinationGuard:
    """
    Fast heuristic hallucination check (no API cost).
    Flags responses with fabricated specifics not present in the ticket.
    """

    HALLUCINATION_THRESHOLD = 0.50

    def check(self, ticket: str, response: str) -> FilterResult:
        from evaluation.hallucination_detector import HallucinationDetector
        detector = HallucinationDetector(use_llm=False, threshold=self.HALLUCINATION_THRESHOLD)
        score    = detector.heuristic_check(ticket, response)
        flagged  = score >= self.HALLUCINATION_THRESHOLD

        if flagged:
            logger.warning(
                "HallucinationGuard: flagged response (score=%.3f)", score
            )

        return FilterResult(
            filter_name="HallucinationGuard",
            passed=not flagged,
            confidence=round(score, 3),
            reason=f"Heuristic score={score:.3f}",
        )


class ComplianceChecker:
    """
    Domain-specific compliance rules — ensures the response addresses
    what the domain actually requires.
    """

    def check(self, response: str, domain: str) -> FilterResult:
        from evaluation.metrics import compliance_check
        rules  = compliance_check(response, domain)
        failed = [k for k, v in rules.items() if not v]

        if failed:
            logger.info(
                "ComplianceChecker: domain=%s failed rules=%s", domain, failed
            )

        return FilterResult(
            filter_name="ComplianceChecker",
            passed=len(failed) == 0,
            confidence=1.0 - len(failed) / max(len(rules), 1),
            reason=(
                "All compliance rules passed"
                if not failed
                else f"Failed rules: {', '.join(failed)}"
            ),
        )


# ── Safety pipeline ────────────────────────────────────────────────────────────

class SafetyPipeline:
    """
    Runs all safety filters in sequence on every response.

    Short-circuits on hard failures (EscalationGuard, ToneChecker).
    PII scrubbing and compliance are soft — they modify or annotate
    but don't block the response.

    Parameters
    ----------
    run_hallucination_check : Enable fast heuristic hallucination filter
    run_compliance_check    : Enable domain compliance rules
    """

    def __init__(
        self,
        run_hallucination_check: bool = True,
        run_compliance_check:    bool = True,
    ) -> None:
        self.escalation_guard    = EscalationGuard()
        self.pii_scrubber        = PIIScrubber()
        self.tone_checker        = ToneChecker()
        self.hallucination_guard = HallucinationGuard() if run_hallucination_check else None
        self.compliance_checker  = ComplianceChecker()  if run_compliance_check    else None

    def run(
        self,
        ticket:           str,
        response:         str,
        domain:           str   = "technical",
        escalation_score: float = 0.0,
        case_id:          Optional[str] = None,
    ) -> SafetyReport:
        """
        Run all safety filters. Returns a SafetyReport with the
        (possibly modified) final response and all filter outcomes.
        """
        import time as _time
        t0 = _time.perf_counter()

        final_response = response
        filter_results: List[FilterResult] = []
        escalation_triggered = False
        pii_scrubbed         = False
        hallucination_flagged = False

        # ── 1. Escalation guard ────────────────────────────────────────────────
        esc_result = self.escalation_guard.check(
            ticket, final_response, escalation_score, case_id
        )
        filter_results.append(esc_result)
        if not esc_result.passed:
            escalation_triggered = True
            if esc_result.override_response:
                final_response = esc_result.override_response
            # Don't return early — still run PII scrub on the override
            # but skip tone/compliance since it's a canned response

        # ── 2. PII scrub (always runs) ─────────────────────────────────────────
        pii_result = self.pii_scrubber.check(final_response)
        filter_results.append(pii_result)
        if pii_result.override_response:
            final_response = pii_result.override_response
            pii_scrubbed   = True

        # Skip remaining filters for escalated / canned responses
        if not escalation_triggered:

            # ── 3. Hallucination guard ─────────────────────────────────────────
            if self.hallucination_guard:
                hall_result = self.hallucination_guard.check(ticket, final_response)
                filter_results.append(hall_result)
                if not hall_result.passed:
                    hallucination_flagged = True
                    # Don't override — just flag; response is still returned
                    logger.warning(
                        "HallucinationGuard flagged response for domain=%s", domain
                    )

            # ── 4. Tone check ──────────────────────────────────────────────────
            tone_result = self.tone_checker.check(final_response, domain)
            filter_results.append(tone_result)

            # ── 5. Compliance check ────────────────────────────────────────────
            if self.compliance_checker:
                comp_result = self.compliance_checker.check(final_response, domain)
                filter_results.append(comp_result)

        safety_ms = (_time.perf_counter() - t0) * 1000

        report = SafetyReport(
            ticket=ticket,
            original_response=response,
            final_response=final_response,
            domain=domain,
            filter_results=filter_results,
            escalation_triggered=escalation_triggered,
            pii_scrubbed=pii_scrubbed,
            hallucination_flagged=hallucination_flagged,
            quality_passed=all(f.passed for f in filter_results),
            case_id=case_id,
            safety_latency_ms=round(safety_ms, 2),
        )
        logger.debug(report.summary())
        return report
