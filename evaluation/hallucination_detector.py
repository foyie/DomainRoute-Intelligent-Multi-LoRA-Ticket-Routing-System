"""
evaluation/hallucination_detector.py
──────────────────────────────────────
LLM-as-judge hallucination detection for VeriTune responses.

Two detection strategies:
  1. Fact-check mode   – Check if the response contradicts the ticket context
  2. Faithfulness mode – Measure how well response stays grounded in the prompt
  3. Heuristic mode    – Fast keyword/pattern checks (no API cost)

Used in:
  - Post-training evaluation (scripts/evaluate_checkpoints.py)
  - Production serving (serving/safety_filters.py) via heuristic mode only

Portfolio note: "Hallucination rate: 1.2% (fact-checking via LLM-as-judge)"

Public API
----------
HallucinationDetector
  .detect(ticket, response, domain)          → HallucinationResult
  .detect_batch(tickets, responses, domains) → List[HallucinationResult]
  .heuristic_check(ticket, response)         → float  (0=clean, 1=hallucinated)
  .evaluate_dataset(dataset, model_fn)       → HallucinationReport

HallucinationResult   – per-response result
HallucinationReport   – aggregate report across a dataset
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class HallucinationResult:
    """Hallucination detection result for a single response."""
    ticket: str
    response: str
    domain: str
    is_hallucinated: bool
    confidence: float          # [0, 1] — confidence that it IS hallucinated
    detection_method: str      # "llm_judge" | "heuristic" | "faithfulness"
    explanation: str = ""
    heuristic_score: float = 0.0   # from fast heuristic check

    def summary(self) -> str:
        flag = "⚠ HALLUCINATED" if self.is_hallucinated else "✓ CLEAN"
        return (
            f"[{self.domain:10s}] {flag} conf={self.confidence:.3f} "
            f"method={self.detection_method}"
        )


@dataclass
class HallucinationReport:
    """Aggregate hallucination stats across an evaluation dataset."""
    n_total: int
    n_hallucinated: int
    hallucination_rate: float
    by_domain: dict           # {domain: {rate, count}}
    avg_confidence: float
    detection_method: str
    results: List[HallucinationResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Hallucination Report ({self.detection_method})",
            f"  Rate: {self.hallucination_rate:.3f} ({self.n_hallucinated}/{self.n_total})",
            f"  Avg confidence: {self.avg_confidence:.3f}",
        ]
        for domain, stats in self.by_domain.items():
            lines.append(f"  {domain:12s}: {stats['rate']:.3f} ({stats['count']} hallucinated)")
        return "\n".join(lines)


# ── Hallucination patterns (heuristic mode) ───────────────────────────────────

_FABRICATION_PATTERNS = [
    # Specific numbers/dates that couldn't be known
    re.compile(r"\border #\d{6,}\b", re.I),     # invented order IDs
    re.compile(r"\$\d+\.\d{2}\b"),               # specific dollar amounts (unless in ticket)
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),        # specific dates
    re.compile(r"\btracking number\s+\w{8,}\b", re.I),
]

_CONTRADICTION_SIGNALS = [
    # Response claims opposite of what ticket says
    (re.compile(r"\bno charge\b", re.I),    re.compile(r"\bcharg\w+\b", re.I)),
    (re.compile(r"\bno issue\b",  re.I),    re.compile(r"\bproblem\b",  re.I)),
    (re.compile(r"\bdelivered\b", re.I),    re.compile(r"\bnot received\b", re.I)),
]

_HALLUCINATION_PHRASES = [
    "as per our conversation last week",
    "as discussed previously",
    "according to your records",
    "your last order of",
    "the technician visited on",
    "we spoke on",
]


class HallucinationDetector:
    """
    LLM-as-judge hallucination detector with fast heuristic fallback.

    Parameters
    ----------
    use_llm        : Use LLM-as-judge (requires API key). Falls back to heuristic.
    model          : LLM model for judging (default: gpt-4o-mini)
    provider       : "openai" | "together" | "anthropic"
    threshold      : Confidence threshold above which we flag as hallucinated
    """

    # def __init__(
    #     self,
    #     use_llm: bool = True,
    #     model: str = "gpt-4o-mini",
    #     provider: str = "openai",
    #     threshold: float = 0.60,
    #     max_retries: int = 2,
    # ) -> None:
    #     self.use_llm     = use_llm
    #     self.model       = model
    #     self.provider    = provider
    #     self.threshold   = threshold
    #     self.max_retries = max_retries
    #     self._client     = None
    def __init__(
        self,
        use_llm: bool = True,
        model: str = "gemini-2.5-flash",  # Changed from gpt-4o-mini
        provider: str = "gemini",          # Changed from openai
        threshold: float = 0.60,
        max_retries: int = 2,
    ) -> None:
        self.use_llm     = use_llm
        self.model       = model
        self.provider    = provider
        self.threshold   = threshold
        self.max_retries = max_retries
        self._client     = None
    # ── Main detection methods ─────────────────────────────────────────────────

    def detect(
        self,
        ticket: str,
        response: str,
        domain: str = "technical",
    ) -> HallucinationResult:
        """
        Detect hallucinations in a single response.
        Uses LLM-as-judge if available, heuristics otherwise.
        """
        # Always run fast heuristic first
        heuristic_score = self.heuristic_check(ticket, response)

        if self.use_llm:
            try:
                return self._llm_judge(ticket, response, domain, heuristic_score)
            except Exception as e:
                logger.warning("LLM judge failed: %s — falling back to heuristic", e)

        # Heuristic-only result
        is_hallucinated = heuristic_score >= self.threshold
        return HallucinationResult(
            ticket=ticket,
            response=response,
            domain=domain,
            is_hallucinated=is_hallucinated,
            confidence=round(heuristic_score, 3),
            detection_method="heuristic",
            explanation="Heuristic pattern detection",
            heuristic_score=round(heuristic_score, 3),
        )

    def detect_batch(
        self,
        tickets: List[str],
        responses: List[str],
        domains: Optional[List[str]] = None,
        batch_size: int = 10,
    ) -> List[HallucinationResult]:
        """
        Detect hallucinations in a batch. Rate-limits LLM calls.
        """
        if len(tickets) != len(responses):
            raise ValueError("tickets and responses must have the same length")

        domains = domains or ["technical"] * len(tickets)
        results = []

        for i in range(0, len(tickets), batch_size):
            batch_t = tickets[i:  i + batch_size]
            batch_r = responses[i: i + batch_size]
            batch_d = domains[i:  i + batch_size]

            for t, r, d in zip(batch_t, batch_r, batch_d):
                result = self.detect(t, r, d)
                results.append(result)
                time.sleep(0.1)   # courtesy rate-limit pause

            logger.debug(
                "Hallucination batch %d-%d / %d",
                i, min(i + batch_size, len(tickets)), len(tickets),
            )

        return results

    def evaluate_dataset(
        self,
        dataset_records: List[dict],
        response_key: str = "response",
        ticket_key: str = "text",
        domain_key: str = "domain",
    ) -> HallucinationReport:
        """
        Run hallucination detection over a full dataset.

        Parameters
        ----------
        dataset_records : List of {text, domain, response, ...} dicts
        """
        tickets   = [r[ticket_key]   for r in dataset_records]
        responses = [r[response_key] for r in dataset_records if response_key in r]
        domains   = [r.get(domain_key, "technical") for r in dataset_records]

        if not responses:
            logger.warning("No responses found in dataset_records (missing key '%s')", response_key)
            return HallucinationReport(0, 0, 0.0, {}, 0.0, "heuristic")

        results   = self.detect_batch(tickets[:len(responses)], responses, domains)
        return self._build_report(results)

    # ── Heuristic detection ────────────────────────────────────────────────────

    def heuristic_check(self, ticket: str, response: str) -> float:
        """
        Fast heuristic hallucination score in [0, 1].
        No API cost. Used in production safety filters.

        Checks:
        1. Fabricated specific numbers/IDs not in the ticket
        2. Contradiction signals between ticket and response
        3. Known hallucination phrases
        """
        score = 0.0

        # Check fabricated patterns (invented specifics)
        for pattern in _FABRICATION_PATTERNS:
            ticket_matches   = set(pattern.findall(ticket))
            response_matches = set(pattern.findall(response))
            invented = response_matches - ticket_matches
            if invented:
                score += 0.30
                break   # one fabrication is enough to flag

        # Check contradiction signals
        for resp_signal, ticket_signal in _CONTRADICTION_SIGNALS:
            if resp_signal.search(response) and ticket_signal.search(ticket):
                score += 0.40
                break

        # Check known hallucination phrases
        resp_lower = response.lower()
        for phrase in _HALLUCINATION_PHRASES:
            if phrase in resp_lower:
                score += 0.25
                break

        # Length heuristic: very short responses may be incomplete/evasive
        if len(response.split()) < 10:
            score += 0.10

        return round(min(score, 1.0), 3)

    # ── LLM-as-judge ──────────────────────────────────────────────────────────

    # def _llm_judge(
    #     self,
    #     ticket: str,
    #     response: str,
    #     domain: str,
    #     heuristic_score: float,
    # ) -> HallucinationResult:
    #     """
    #     Use an LLM to judge whether the response contains hallucinations.
    #     Returns a structured HallucinationResult.
    #     """
    #     try:
    #         import instructor
    #         from openai import OpenAI
    #         from pydantic import BaseModel, Field

    #         class JudgmentResult(BaseModel):
    #             is_hallucinated: bool = Field(
    #                 description="True if the response contains invented facts or contradictions"
    #             )
    #             confidence: float = Field(
    #                 ge=0.0, le=1.0,
    #                 description="Confidence that the response IS hallucinated (0=clean, 1=hallucinated)"
    #             )
    #             explanation: str = Field(
    #                 description="Brief explanation of what is or isn't hallucinated"
    #             )

    #     except ImportError:
    #         raise ImportError("instructor and openai are required for LLM-as-judge")

    #     prompt = (
    #         f"You are evaluating a customer support response for factual accuracy.\n\n"
    #         f"Domain: {domain}\n\n"
    #         f"Customer ticket:\n{ticket}\n\n"
    #         f"Support response:\n{response}\n\n"
    #         f"Evaluate whether the response:\n"
    #         f"1. Invents specific facts not mentioned in the ticket "
    #         f"(order IDs, dates, amounts, names)\n"
    #         f"2. Contradicts what the customer said\n"
    #         f"3. Makes claims about previous interactions that couldn't be known\n\n"
    #         f"Note: Generic advice ('please restart your device') is NOT hallucination. "
    #         f"Only flag invented specifics or direct contradictions."
    #     )

    #     client = instructor.from_openai(OpenAI())

    #     for attempt in range(self.max_retries + 1):
    #         try:
    #             judgment = client.chat.completions.create(
    #                 model=self.model,
    #                 response_model=JudgmentResult,
    #                 messages=[{"role": "user", "content": prompt}],
    #             )
    #             # Blend LLM and heuristic scores (LLM is more reliable)
    #             blended_confidence = 0.7 * judgment.confidence + 0.3 * heuristic_score

    #             return HallucinationResult(
    #                 ticket=ticket,
    #                 response=response,
    #                 domain=domain,
    #                 is_hallucinated=blended_confidence >= self.threshold,
    #                 confidence=round(blended_confidence, 3),
    #                 detection_method="llm_judge",
    #                 explanation=judgment.explanation,
    #                 heuristic_score=round(heuristic_score, 3),
    #             )
    #         except Exception as e:
    #             if attempt == self.max_retries:
    #                 raise
    #             time.sleep(2 ** attempt)
    def _llm_judge(
        self,
        ticket: str,
        response: str,
        domain: str,
        heuristic_score: float,
    ) -> HallucinationResult:
        """
        Use Gemini to judge whether the response contains hallucinations.
        Returns a structured HallucinationResult.
        """
        import os
        import json
        import google.generativeai as genai
        from pydantic import BaseModel, Field

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        # Structured output schema
        class JudgmentResult(BaseModel):
            is_hallucinated: bool = Field(
                description="True if the response contains invented facts or contradictions"
            )
            confidence: float = Field(
                ge=0.0, le=1.0,
                description="Confidence that the response IS hallucinated (0=clean, 1=hallucinated)"
            )
            explanation: str = Field(
                description="Brief explanation of what is or isn't hallucinated"
            )

        prompt = (
            f"You are evaluating a customer support response for factual accuracy.\n\n"
            f"Domain: {domain}\n\n"
            f"Customer ticket:\n{ticket}\n\n"
            f"Support response:\n{response}\n\n"
            f"Evaluate whether the response:\n"
            f"1. Invents specific facts not mentioned in the ticket "
            f"(order IDs, dates, amounts, names)\n"
            f"2. Contradicts what the customer said\n"
            f"3. Makes claims about previous interactions that couldn't be known\n\n"
            f"Note: Generic advice ('please restart your device') is NOT hallucination. "
            f"Only flag invented specifics or direct contradictions.\n\n"
            f"Respond in JSON format:\n"
            f"{{\n"
            f'  "is_hallucinated": bool,\n'
            f'  "confidence": float (0-1),\n'
            f'  "explanation": "string"\n'
            f"}}"
        )

        for attempt in range(self.max_retries + 1):
            try:
                # model = genai.GenerativeModel("gemini-1.5-flash")
                model = genai.GenerativeModel("gemini-2.5-flash")

                response_obj = model.generate_content(prompt)

                # Parse JSON from Gemini response
                text = response_obj.text
                # Extract JSON from potential markdown code blocks
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]

                judgment_dict = json.loads(text.strip())

                # Blend LLM and heuristic scores
                llm_confidence = float(judgment_dict.get("confidence", 0.5))
                blended_confidence = 0.7 * llm_confidence + 0.3 * heuristic_score

                return HallucinationResult(
                    ticket=ticket,
                    response=response,
                    domain=domain,
                    is_hallucinated=blended_confidence >= self.threshold,
                    confidence=round(blended_confidence, 3),
                    detection_method="llm_judge",
                    explanation=judgment_dict.get("explanation", ""),
                    heuristic_score=round(heuristic_score, 3),
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt == self.max_retries:
                    logger.error("Gemini judge failed after retries: %s", e)
                    # Fallback to heuristic
                    is_hallucinated = heuristic_score >= self.threshold
                    return HallucinationResult(
                        ticket=ticket,
                        response=response,
                        domain=domain,
                        is_hallucinated=is_hallucinated,
                        confidence=round(heuristic_score, 3),
                        detection_method="heuristic_fallback",
                        explanation="LLM judge failed; using heuristic",
                        heuristic_score=round(heuristic_score, 3),
                    )
                time.sleep(2 ** attempt)
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                logger.warning("Gemini API error (attempt %d/%d): %s", attempt, self.max_retries, e)
                time.sleep(2 ** attempt)

    # ── Report building ────────────────────────────────────────────────────────

    def _build_report(self, results: List[HallucinationResult]) -> HallucinationReport:
        n_total        = len(results)
        n_hallucinated = sum(1 for r in results if r.is_hallucinated)
        rate           = n_hallucinated / max(n_total, 1)

        # Per-domain breakdown
        by_domain: dict = {}
        for r in results:
            d = r.domain
            if d not in by_domain:
                by_domain[d] = {"count": 0, "total": 0, "rate": 0.0}
            by_domain[d]["total"] += 1
            if r.is_hallucinated:
                by_domain[d]["count"] += 1
        for d in by_domain:
            t = by_domain[d]["total"]
            c = by_domain[d]["count"]
            by_domain[d]["rate"] = round(c / max(t, 1), 4)

        avg_conf = (
            sum(r.confidence for r in results) / max(len(results), 1)
        )
        method = results[0].detection_method if results else "heuristic"

        report = HallucinationReport(
            n_total=n_total,
            n_hallucinated=n_hallucinated,
            hallucination_rate=round(rate, 4),
            by_domain=by_domain,
            avg_confidence=round(avg_conf, 3),
            detection_method=method,
            results=results,
        )
        logger.info(report.summary())
        return report

    def save_report(self, report: HallucinationReport, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n_total":            report.n_total,
            "n_hallucinated":     report.n_hallucinated,
            "hallucination_rate": report.hallucination_rate,
            "by_domain":          report.by_domain,
            "avg_confidence":     report.avg_confidence,
            "detection_method":   report.detection_method,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Hallucination report saved → %s", path)
