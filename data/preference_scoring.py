"""
data/preference_scoring.py
───────────────────────────
Converts raw support-ticket responses into preference-ranked pairs suitable
for DPO (Direct Preference Optimisation) fine-tuning.

Two approaches:
  1. Rule-based scoring  – fast, deterministic, no API cost
  2. LLM-as-judge        – GPT-4o-mini scores pairs (higher quality, costs ~$0.001/pair)

Bradley-Terry model fits pairwise win/loss counts into a global ranking,
giving a real-valued "quality score" per response.

Public API
----------
score_response_rule_based(response, domain)     → float
score_responses_llm(ticket, responses, domain)  → List[float]
build_preference_pairs(dataset, cfg)            → List[PreferencePair]
bradley_terry_ranking(win_matrix)               → np.ndarray
save_preference_pairs(pairs, path)              → None
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PreferencePair:
    """A single preference training example."""
    ticket: str                    # input ticket text
    chosen: str                    # preferred response
    rejected: str                  # dispreferred response
    domain: str
    chosen_score: float
    rejected_score: float
    scoring_method: str            # "rule_based" | "llm"

    def to_dpo_format(self) -> dict:
        """Return dict compatible with TRL's DPOTrainer."""
        return {
            "prompt":   self.ticket,
            "chosen":   self.chosen,
            "rejected": self.rejected,
        }


# ── Rule-based scoring ─────────────────────────────────────────────────────────

RULE_WEIGHTS = {
    "empathy": 0.20,
    "action":  0.25,
    "clarity": 0.20,
    "length":  0.15,
    "policy":  0.20,
}

EMPATHY_PHRASES = [
    "i'm sorry", "i apologise", "i understand", "that must be",
    "thank you for", "i can see", "i appreciate",
]

ACTION_PHRASES = [
    "i will", "i've", "i have", "we will", "we've", "processing",
    "arranging", "escalating", "refunding", "replacing",
]

VAGUE_PHRASES = [
    "please contact", "please visit", "unfortunately we cannot",
    "as per policy", "i cannot help with that",
]

DOMAIN_REQUIRED_ACTIONS = {
    "technical": ["step", "try", "restart", "reset", "update", "check"],
    "billing":   ["refund", "charge", "invoice", "account", "payment"],
    "returns":   ["return", "label", "ship", "exchange", "replacement"],
    "escalation": ["escalat", "specialist", "manager", "priority", "case id"],
}


def score_response_rule_based(response: str, domain: str) -> float:
    """
    Score a single response on a 0-1 scale using heuristic rules.

    Dimensions
    ----------
    empathy  – contains empathetic language
    action   – contains concrete action (not vague deflection)
    clarity  – appropriate length and structure
    length   – not too short (< 20 words) or too long (> 150 words)
    policy   – contains domain-required action keywords
    """
    response_lower = response.lower()
    words = response_lower.split()
    n_words = len(words)

    # Empathy score
    empathy = sum(1 for p in EMPATHY_PHRASES if p in response_lower)
    empathy_score = min(empathy / 2, 1.0)

    # Action score (positive actions minus vague deflections)
    actions = sum(1 for p in ACTION_PHRASES if p in response_lower)
    vague   = sum(1 for p in VAGUE_PHRASES  if p in response_lower)
    action_score = min(max((actions - vague) / 2, 0), 1.0)

    # Clarity — penalise single run-on sentence or bullet-point dump
    sentences = re.split(r"[.!?]", response)
    n_sentences = sum(1 for s in sentences if len(s.strip()) > 5)
    clarity_score = min(n_sentences / 3, 1.0) if n_sentences >= 2 else 0.4

    # Length score
    if n_words < 20:
        length_score = 0.3
    elif n_words > 150:
        length_score = 0.5
    else:
        # Peak at 50-100 words
        length_score = 1.0 - abs(n_words - 75) / 75 * 0.4

    # Policy / domain compliance
    domain_keywords = DOMAIN_REQUIRED_ACTIONS.get(domain, [])
    domain_hits = sum(1 for kw in domain_keywords if kw in response_lower)
    policy_score = min(domain_hits / max(len(domain_keywords), 1), 1.0)

    total = (
        RULE_WEIGHTS["empathy"] * empathy_score +
        RULE_WEIGHTS["action"]  * action_score  +
        RULE_WEIGHTS["clarity"] * clarity_score +
        RULE_WEIGHTS["length"]  * length_score  +
        RULE_WEIGHTS["policy"]  * policy_score
    )
    return round(float(total), 4)


# ── LLM-as-judge scoring ───────────────────────────────────────────────────────

def score_responses_llm(
    ticket: str,
    responses: List[str],
    domain: str,
    model: str = "gpt-4o-mini",
) -> List[float]:
    """
    Use GPT-4o-mini to score each response on a 1-10 scale.
    Returns a list of normalised [0, 1] scores.
    Falls back to rule-based scoring if API unavailable.
    """
    try:
        import instructor
        from openai import OpenAI
        from pydantic import BaseModel, Field

        class ScoreList(BaseModel):
            scores: List[float] = Field(
                description="Score for each response from 1 (worst) to 10 (best)"
            )

    except ImportError:
        logger.warning("instructor/openai not installed. Using rule-based scoring.")
        return [score_response_rule_based(r, domain) for r in responses]

    prompt = (
        f"You are evaluating customer support responses for quality.\n\n"
        f"Domain: {domain}\n"
        f"Customer ticket: \"{ticket}\"\n\n"
        f"Rate each response from 1 (very poor) to 10 (excellent) based on:\n"
        f"- Empathy and tone\n"
        f"- Concrete action taken or offered\n"
        f"- Clarity and appropriate length\n"
        f"- Domain-appropriate content (e.g. billing responses should address the charge)\n\n"
        + "\n\n".join(f"Response {i+1}:\n{r}" for i, r in enumerate(responses))
        + f"\n\nReturn exactly {len(responses)} scores."
    )

    client = instructor.from_openai(OpenAI())
    try:
        result = client.chat.completions.create(
            model=model,
            response_model=ScoreList,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_scores = result.scores[:len(responses)]
        # Normalise to [0, 1]
        return [round(s / 10.0, 4) for s in raw_scores]
    except Exception as e:
        logger.warning("LLM scoring failed: %s. Falling back to rule-based.", e)
        return [score_response_rule_based(r, domain) for r in responses]


# ── Preference pair construction ───────────────────────────────────────────────

def build_preference_pairs(
    records: List[dict],
    cfg: Optional[dict] = None,
    use_llm: bool = False,
    min_score_gap: float = 0.15,
) -> List[PreferencePair]:
    """
    Build DPO-format preference pairs from records containing multiple responses.

    Each record should have:
      - text     : the ticket
      - domain   : domain label
      - responses: list of candidate response strings

    Parameters
    ----------
    records       : List of ticket records with candidate responses
    min_score_gap : Minimum score difference to create a preference pair
                    (pairs too close in quality are ambiguous)

    Returns
    -------
    List of PreferencePair (chosen > rejected by at least min_score_gap)
    """
    pairs: List[PreferencePair] = []

    for record in records:
        ticket    = record.get("text", "")
        domain    = record.get("domain", "technical")
        responses = record.get("responses", [])

        if len(responses) < 2:
            continue

        # Score all responses
        if use_llm:
            scores = score_responses_llm(ticket, responses, domain)
            method = "llm"
        else:
            scores = [score_response_rule_based(r, domain) for r in responses]
            method = "rule_based"

        # Build pairs from all combinations
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                s_i, s_j = scores[i], scores[j]
                gap = abs(s_i - s_j)

                if gap < min_score_gap:
                    continue   # too ambiguous

                if s_i > s_j:
                    chosen, rejected = responses[i], responses[j]
                    chosen_score, rejected_score = s_i, s_j
                else:
                    chosen, rejected = responses[j], responses[i]
                    chosen_score, rejected_score = s_j, s_i

                pairs.append(PreferencePair(
                    ticket=ticket,
                    chosen=chosen,
                    rejected=rejected,
                    domain=domain,
                    chosen_score=chosen_score,
                    rejected_score=rejected_score,
                    scoring_method=method,
                ))

    logger.info(
        "Built %d preference pairs from %d records (min_gap=%.2f, method=%s)",
        len(pairs), len(records), min_score_gap, "llm" if use_llm else "rule_based",
    )
    return pairs


# ── Bradley-Terry model ────────────────────────────────────────────────────────

def bradley_terry_ranking(
    win_matrix: np.ndarray,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Fit the Bradley-Terry model to a pairwise win/loss matrix.
    Returns a normalised quality score vector (sums to 1).

    Parameters
    ----------
    win_matrix : (N, N) array where win_matrix[i, j] = number of times i beat j
    n_iter     : Maximum iterations
    tol        : Convergence tolerance

    Returns
    -------
    scores : (N,) array of quality scores (higher = better)
    """
    n = win_matrix.shape[0]
    scores = np.ones(n)

    for _ in range(n_iter):
        new_scores = np.zeros(n)
        for i in range(n):
            wins = win_matrix[i].sum()
            if wins == 0:
                new_scores[i] = 1e-8
                continue
            denom = sum(
                (win_matrix[i, j] + win_matrix[j, i]) / (scores[i] + scores[j])
                for j in range(n) if i != j and (win_matrix[i, j] + win_matrix[j, i]) > 0
            )
            new_scores[i] = wins / denom if denom > 0 else 1e-8

        # Normalise
        new_scores /= new_scores.sum()

        if np.abs(new_scores - scores).max() < tol:
            break
        scores = new_scores

    return scores


def pairs_to_win_matrix(
    pairs: List[PreferencePair],
    response_index: dict,
) -> np.ndarray:
    """
    Convert a list of PreferencePairs into an (N, N) win matrix.
    `response_index` maps response text → integer index.
    """
    n = len(response_index)
    win_matrix = np.zeros((n, n), dtype=float)
    for pair in pairs:
        i = response_index.get(pair.chosen,  -1)
        j = response_index.get(pair.rejected, -1)
        if i >= 0 and j >= 0:
            win_matrix[i, j] += 1
    return win_matrix


# ── Persistence ────────────────────────────────────────────────────────────────

def save_preference_pairs(pairs: List[PreferencePair], path: str | Path) -> None:
    """Save preference pairs as JSONL for DPO training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")

    logger.info("Saved %d preference pairs to %s", len(pairs), path)


def load_preference_pairs(path: str | Path) -> List[PreferencePair]:
    """Load preference pairs from a JSONL file."""
    path = Path(path)
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                pairs.append(PreferencePair(**data))
    return pairs
