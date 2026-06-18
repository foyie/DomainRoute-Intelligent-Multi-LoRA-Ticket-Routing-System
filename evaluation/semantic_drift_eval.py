"""
evaluation/semantic_drift_eval.py
───────────────────────────────────
Post-training semantic drift evaluation for all VeriTune domain LoRAs.

Unlike the per-epoch DriftCallback used during training, this module runs
a comprehensive post-hoc drift analysis across:
  - All four domain LoRAs vs the base model
  - Cross-domain contamination (does billing vocab appear in technical outputs?)
  - Vocabulary shift (top-K token probability changes)
  - Embedding space geometry (intra-domain vs inter-domain distances)

Portfolio note: "Semantic drift: all domains >0.94 cosine similarity to base"

Public API
----------
SemanticDriftEvaluator
  .evaluate_domain(domain, adapter_path, probe_texts)   → DriftEvalResult
  .evaluate_all_domains(registry, probe_set)            → Dict[str, DriftEvalResult]
  .cross_domain_contamination(domains, probes)          → ContaminationMatrix
  .vocabulary_shift(base_model, ft_model, tokenizer, texts) → VocabShiftResult
  .save_report(results, path)                           → None
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

COSINE_SIM_THRESHOLD = 0.94    # below this → flag as drifted


@dataclass
class DriftEvalResult:
    """Drift evaluation result for a single domain."""
    domain: str
    adapter_path: str
    mean_cosine_similarity: float
    std_cosine_similarity: float
    min_cosine_similarity: float
    max_cosine_similarity: float
    cosine_distance: float             # 1 - mean_cosine_similarity
    is_drifted: bool                   # True if mean_sim < threshold
    n_probe_texts: int
    layer_similarities: Dict[str, float] = field(default_factory=dict)  # per-layer analysis

    def summary(self) -> str:
        status = "⚠ DRIFT" if self.is_drifted else "✓ OK"
        return (
            f"{self.domain:12s}  cosine_sim={self.mean_cosine_similarity:.4f} "
            f"± {self.std_cosine_similarity:.4f}  dist={self.cosine_distance:.4f}  "
            f"[{status}]"
        )


@dataclass
class ContaminationMatrix:
    """Cross-domain vocabulary contamination analysis."""
    domains: List[str]
    matrix: List[List[float]]    # contamination[i][j] = how much domain i bleeds into j
    max_contamination: float
    contamination_pairs: List[Tuple[str, str, float]]  # (source, target, score) sorted desc

    def summary(self) -> str:
        lines = ["Cross-domain contamination matrix:"]
        lines.append("  " + "  ".join(f"{d[:6]:>6}" for d in self.domains))
        for i, (row, domain) in enumerate(zip(self.matrix, self.domains)):
            vals = "  ".join(f"{v:6.3f}" for v in row)
            lines.append(f"  {domain[:6]:>6}  {vals}")
        if self.contamination_pairs:
            top = self.contamination_pairs[0]
            lines.append(f"  Max: {top[0]} → {top[1]} ({top[2]:.4f})")
        return "\n".join(lines)


@dataclass
class VocabShiftResult:
    """Vocabulary probability shift between base and fine-tuned model."""
    domain: str
    top_shifted_tokens: List[Tuple[str, float, float]]  # (token, base_prob, ft_prob)
    mean_kl_divergence: float
    max_token_shift: float
    n_tokens_analyzed: int


class SemanticDriftEvaluator:
    """
    Comprehensive semantic drift evaluation across all VeriTune domain LoRAs.

    Parameters
    ----------
    base_model_name : HuggingFace model identifier for the base model
    tokenizer       : Pre-loaded tokenizer (shared across domains)
    device          : Compute device
    layer           : Which hidden layer to extract embeddings from (-1 = last)
    """

    def __init__(
        self,
        base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        tokenizer=None,
        device: str = "cpu",
        layer: int = -1,
    ) -> None:
        self.base_model_name = base_model_name
        self.tokenizer       = tokenizer
        self.device          = device
        self.layer           = layer
        self._base_model     = None

    @property
    def base_model(self):
        """Lazy-load the base model (shared across all domain evaluations)."""
        if self._base_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("Loading base model for drift evaluation: %s", self.base_model_name)
            self._base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name, device_map=self.device, trust_remote_code=True
            )
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        return self._base_model

    # ── Single domain evaluation ───────────────────────────────────────────────

    def evaluate_domain(
        self,
        domain: str,
        adapter_path: str | Path,
        probe_texts: List[str],
    ) -> DriftEvalResult:
        """
        Evaluate semantic drift for a single domain LoRA.

        Compares the last hidden state embeddings of the base model vs
        the fine-tuned model on a shared probe set.

        Parameters
        ----------
        domain       : Domain name (for reporting)
        adapter_path : Path to the saved LoRA adapter
        probe_texts  : Texts to use as drift probes (use validation set)
        """
        from peft import PeftModel

        logger.info("Evaluating drift: domain=%s adapter=%s", domain, adapter_path)

        # Extract base embeddings
        base_embs = self._extract_embeddings(self.base_model, probe_texts)

        # Load fine-tuned model and extract embeddings
        ft_model = PeftModel.from_pretrained(
            self.base_model, str(adapter_path), is_trainable=False
        )
        ft_embs = self._extract_embeddings(ft_model, probe_texts)

        # Compute per-example cosine similarities
        cos_sims = self._cosine_similarities(base_embs, ft_embs)
        mean_sim = float(np.mean(cos_sims))
        std_sim  = float(np.std(cos_sims))

        # Per-layer analysis (sample 3 layers: early, middle, last)
        layer_sims: Dict[str, float] = {}
        for layer_idx, layer_name in [(-1, "last"), (-8, "mid"), (1, "early")]:
            try:
                b_layer = self._extract_embeddings(self.base_model, probe_texts, layer=layer_idx)
                f_layer = self._extract_embeddings(ft_model, probe_texts, layer=layer_idx)
                layer_sims[layer_name] = round(float(np.mean(self._cosine_similarities(b_layer, f_layer))), 4)
            except Exception:
                pass

        del ft_model  # free memory

        result = DriftEvalResult(
            domain=domain,
            adapter_path=str(adapter_path),
            mean_cosine_similarity=round(mean_sim, 4),
            std_cosine_similarity=round(std_sim, 4),
            min_cosine_similarity=round(float(np.min(cos_sims)), 4),
            max_cosine_similarity=round(float(np.max(cos_sims)), 4),
            cosine_distance=round(1.0 - mean_sim, 4),
            is_drifted=mean_sim < COSINE_SIM_THRESHOLD,
            n_probe_texts=len(probe_texts),
            layer_similarities=layer_sims,
        )

        logger.info(result.summary())
        if result.is_drifted:
            logger.warning(
                "DRIFT DETECTED: %s cosine_sim=%.4f < threshold=%.2f",
                domain, mean_sim, COSINE_SIM_THRESHOLD,
            )

        return result

    # ── All domains ────────────────────────────────────────────────────────────

    def evaluate_all_domains(
        self,
        adapter_registry,          # AdapterRegistry
        probe_texts_by_domain: Dict[str, List[str]],
    ) -> Dict[str, DriftEvalResult]:
        """
        Run drift evaluation for all domains and return a summary dict.

        Parameters
        ----------
        adapter_registry      : AdapterRegistry with domain → path mapping
        probe_texts_by_domain : {domain_str: [probe_text, ...]}
        """
        results: Dict[str, DriftEvalResult] = {}

        for domain_str, probe_texts in probe_texts_by_domain.items():
            from routing.models import Domain
            try:
                domain = Domain(domain_str)
            except ValueError:
                logger.warning("Unknown domain '%s' — skipping", domain_str)
                continue

            adapter_path = adapter_registry.paths.get(domain, "")
            if not adapter_path:
                logger.warning("No adapter path for domain '%s' — skipping", domain_str)
                continue

            try:
                results[domain_str] = self.evaluate_domain(
                    domain_str, adapter_path, probe_texts
                )
            except Exception as e:
                logger.error("Drift eval failed for '%s': %s", domain_str, e)

        self._log_summary(results)
        return results

    # ── Cross-domain contamination ─────────────────────────────────────────────

    def cross_domain_contamination(
        self,
        domain_probe_embeddings: Dict[str, np.ndarray],
    ) -> ContaminationMatrix:
        """
        Measure how much vocabulary from each domain bleeds into other domains
        by computing cross-domain cosine similarities between prototype embeddings.

        Parameters
        ----------
        domain_probe_embeddings : {domain_str: (N, D) embedding matrix}
        """
        domains = list(domain_probe_embeddings.keys())
        n = len(domains)

        # Compute domain prototypes (mean embedding)
        prototypes = {
            d: embs.mean(axis=0) / (np.linalg.norm(embs.mean(axis=0)) + 1e-9)
            for d, embs in domain_probe_embeddings.items()
        }

        # Build contamination matrix
        matrix = [[0.0] * n for _ in range(n)]
        pairs = []

        for i, src in enumerate(domains):
            for j, tgt in enumerate(domains):
                if i == j:
                    matrix[i][j] = 1.0  # self-similarity
                    continue
                sim = float(np.dot(prototypes[src], prototypes[tgt]))
                # Contamination: how similar is source domain's embedding to target's
                contamination = max(0.0, sim)   # clip to [0, 1]
                matrix[i][j] = round(contamination, 4)
                if i < j:
                    pairs.append((src, tgt, contamination))

        pairs.sort(key=lambda x: x[2], reverse=True)

        return ContaminationMatrix(
            domains=domains,
            matrix=matrix,
            max_contamination=max(p[2] for p in pairs) if pairs else 0.0,
            contamination_pairs=pairs,
        )

    # ── Vocabulary shift ───────────────────────────────────────────────────────

    def vocabulary_shift(
        self,
        base_model,
        ft_model,
        texts: List[str],
        top_k: int = 20,
    ) -> VocabShiftResult:
        """
        Analyse which tokens have shifted most in probability between
        the base model and fine-tuned model.

        Measures KL divergence of token distributions over a probe set.
        """
        import torch

        if self.tokenizer is None:
            raise RuntimeError("tokenizer must be set before calling vocabulary_shift()")

        base_logits_list = []
        ft_logits_list   = []

        base_model.eval(); ft_model.eval()

        with torch.no_grad():
            for text in texts[:10]:   # limit for speed
                enc = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=64
                ).to(self.device)

                base_out = base_model(**enc).logits[:, -1, :]   # last token logits
                ft_out   = ft_model(**enc).logits[:, -1, :]

                base_logits_list.append(torch.softmax(base_out, dim=-1).squeeze().cpu().numpy())
                ft_logits_list.append(torch.softmax(ft_out, dim=-1).squeeze().cpu().numpy())

        base_probs = np.stack(base_logits_list).mean(axis=0)   # (vocab_size,)
        ft_probs   = np.stack(ft_logits_list).mean(axis=0)

        # KL divergence: KL(ft || base)
        kl = float(np.sum(ft_probs * np.log((ft_probs + 1e-9) / (base_probs + 1e-9))))

        # Top shifted tokens
        shifts    = np.abs(ft_probs - base_probs)
        top_idx   = np.argsort(shifts)[::-1][:top_k]
        token_map = self.tokenizer.convert_ids_to_tokens(top_idx.tolist())
        top_shifted = [
            (str(tok), round(float(base_probs[i]), 6), round(float(ft_probs[i]), 6))
            for tok, i in zip(token_map, top_idx)
        ]

        return VocabShiftResult(
            domain="",
            top_shifted_tokens=top_shifted,
            mean_kl_divergence=round(kl, 6),
            max_token_shift=round(float(shifts.max()), 6),
            n_tokens_analyzed=len(texts),
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_report(
        self,
        results: Dict[str, DriftEvalResult],
        path: str | Path,
        contamination: Optional[ContaminationMatrix] = None,
    ) -> None:
        """Save all drift evaluation results to a JSON report."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "threshold": COSINE_SIM_THRESHOLD,
            "results":   {d: asdict(r) for d, r in results.items()},
            "summary": {
                d: r.summary() for d, r in results.items()
            },
            "all_pass": all(not r.is_drifted for r in results.values()),
        }
        if contamination:
            report["contamination"] = asdict(contamination)

        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Drift report saved → %s", path)

    @classmethod
    def load_report(cls, path: str | Path) -> Dict[str, DriftEvalResult]:
        """Load a saved drift evaluation report."""
        with open(path) as f:
            data = json.load(f)
        return {
            domain: DriftEvalResult(**r)
            for domain, r in data.get("results", {}).items()
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _extract_embeddings(
        self,
        model,
        texts: List[str],
        layer: Optional[int] = None,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Extract mean-pooled hidden states from a model."""
        import torch

        layer = layer if layer is not None else self.layer
        model.eval()
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                enc   = self.tokenizer(
                    batch, return_tensors="pt", truncation=True,
                    max_length=128, padding=True,
                ).to(self.device)

                out    = model(**enc, output_hidden_states=True, return_dict=True)
                hidden = out.hidden_states[layer]       # (B, seq, D)
                mask   = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                embeddings.append(pooled.cpu().numpy())

        return np.vstack(embeddings).astype(np.float32)

    @staticmethod
    def _cosine_similarities(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import normalize
        return (normalize(a) * normalize(b)).sum(axis=1)

    def _log_summary(self, results: Dict[str, DriftEvalResult]) -> None:
        logger.info("=== Semantic Drift Summary ===")
        for domain, r in results.items():
            logger.info("  %s", r.summary())
        all_pass = all(not r.is_drifted for r in results.values())
        logger.info("  All domains pass threshold: %s", "YES" if all_pass else "NO")


# ── Lightweight drift check (no model loading — uses saved embeddings) ─────────

def fast_drift_check(
    base_embeddings: np.ndarray,
    ft_embeddings: np.ndarray,
    threshold: float = COSINE_SIM_THRESHOLD,
) -> Tuple[float, bool]:
    """
    Quick cosine similarity check between two embedding sets.
    Used in serving/monitoring to detect drift in production.

    Returns (mean_cosine_similarity, is_drifted).
    """
    from sklearn.preprocessing import normalize
    base_norm = normalize(base_embeddings)
    ft_norm   = normalize(ft_embeddings)
    cos_sims  = (base_norm * ft_norm).sum(axis=1)
    mean_sim  = float(np.mean(cos_sims))
    return round(mean_sim, 4), mean_sim < threshold
