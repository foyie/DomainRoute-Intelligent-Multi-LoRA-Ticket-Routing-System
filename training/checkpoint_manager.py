"""
training/checkpoint_manager.py
────────────────────────────────
Manages LoRA adapter checkpoints across all domains:

- Tracks top-K checkpoints per domain by a configurable metric
- Promotes the best checkpoint to a canonical "best" path
- Loads and compares checkpoints across domains
- Generates a checkpoint registry (JSON) for the serving layer

Public API
----------
CheckpointRecord         – metadata for a single saved checkpoint
CheckpointManager
  .register(domain, step, metrics, path)   → CheckpointRecord
  .get_best(domain)                        → CheckpointRecord | None
  .get_top_k(domain, k)                   → List[CheckpointRecord]
  .promote_best(domain)                   → Path  (copies to outputs/checkpoints/<domain>_best/)
  .prune(domain, keep_k)                  → int   (number pruned)
  .save_registry(path)                    → None
  .load_registry(path)                    → None
  .summary()                              → str
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class CheckpointRecord:
    """Metadata for a single saved checkpoint."""
    domain: str
    step: int
    epoch: float
    metrics: Dict[str, float]       # e.g. {"eval_loss": 0.312, "auto_res_rate": 0.941}
    adapter_path: str               # path to the saved LoRA adapter directory
    is_best: bool = False

    @property
    def primary_metric(self) -> float:
        """Return the primary metric value used for ranking (eval_loss by default)."""
        return self.metrics.get("eval_loss", float("inf"))

    def better_than(self, other: "CheckpointRecord", metric: str = "eval_loss",
                    greater_is_better: bool = False) -> bool:
        v_self  = self.metrics.get(metric, float("inf") if not greater_is_better else float("-inf"))
        v_other = other.metrics.get(metric, float("inf") if not greater_is_better else float("-inf"))
        return v_self < v_other if not greater_is_better else v_self > v_other


# ── Manager ────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Tracks and manages LoRA adapter checkpoints for all domains.

    Parameters
    ----------
    output_dir         : Root directory for checkpoints (e.g. outputs/checkpoints/)
    top_k              : Maximum number of checkpoints to keep per domain
    primary_metric     : Metric to rank checkpoints by
    greater_is_better  : True if higher metric = better (e.g. accuracy)
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs/checkpoints",
        top_k: int = 3,
        primary_metric: str = "eval_loss",
        greater_is_better: bool = False,
    ) -> None:
        self.output_dir       = Path(output_dir)
        self.top_k            = top_k
        self.primary_metric   = primary_metric
        self.greater_is_better = greater_is_better
        self._registry: Dict[str, List[CheckpointRecord]] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Registration ───────────────────────────────────────────────────────────

    def register(
        self,
        domain: str,
        step: int,
        epoch: float,
        metrics: Dict[str, float],
        adapter_path: str | Path,
    ) -> CheckpointRecord:
        """
        Register a new checkpoint. Automatically prunes if > top_k checkpoints.

        Returns the registered CheckpointRecord.
        """
        record = CheckpointRecord(
            domain=domain,
            step=step,
            epoch=epoch,
            metrics=metrics,
            adapter_path=str(adapter_path),
        )

        if domain not in self._registry:
            self._registry[domain] = []

        self._registry[domain].append(record)
        self._registry[domain] = self._sort(self._registry[domain])

        # Update is_best flag
        for i, ckpt in enumerate(self._registry[domain]):
            ckpt.is_best = (i == 0)

        # Prune if over limit
        pruned = self.prune(domain, keep_k=self.top_k)
        if pruned:
            logger.debug("Pruned %d checkpoints for domain '%s'", pruned, domain)

        best = self._registry[domain][0]
        logger.info(
            "Checkpoint registered: domain=%s step=%d %s=%.4f %s",
            domain, step, self.primary_metric,
            metrics.get(self.primary_metric, float("nan")),
            "(new best!)" if record is best else "",
        )
        return record

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def get_best(self, domain: str) -> Optional[CheckpointRecord]:
        """Return the best checkpoint for a domain, or None if none registered."""
        records = self._registry.get(domain, [])
        return records[0] if records else None

    def get_top_k(self, domain: str, k: Optional[int] = None) -> List[CheckpointRecord]:
        """Return top-K checkpoints for a domain (sorted best first)."""
        records = self._registry.get(domain, [])
        k = k or self.top_k
        return records[:k]

    def all_domains(self) -> List[str]:
        return list(self._registry.keys())

    def best_across_all_domains(self) -> Dict[str, CheckpointRecord]:
        """Return {domain: best_checkpoint} for all registered domains."""
        return {
            domain: self.get_best(domain)
            for domain in self._registry
            if self.get_best(domain) is not None
        }

    # ── Promotion ──────────────────────────────────────────────────────────────

    def promote_best(self, domain: str) -> Path:
        """
        Copy the best checkpoint to a canonical path:
          outputs/checkpoints/<domain>_best/

        This is the path the serving layer loads from.
        Returns the destination path.
        """
        best = self.get_best(domain)
        if best is None:
            raise RuntimeError(f"No checkpoints registered for domain '{domain}'")

        src  = Path(best.adapter_path)
        dest = self.output_dir / f"{domain}_best"

        if dest.exists():
            shutil.rmtree(dest)

        if src.exists():
            shutil.copytree(src, dest)
            logger.info("Promoted best checkpoint: %s → %s", src, dest)
        else:
            # If src path doesn't exist (dry run / test), create placeholder
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "adapter_config.json").write_text(
                json.dumps({
                    "domain": domain,
                    "step": best.step,
                    "metrics": best.metrics,
                    "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "lora_r": 32,
                }, indent=2)
            )
            logger.warning(
                "Source checkpoint path does not exist (%s). "
                "Created placeholder at %s.", src, dest,
            )

        return dest

    def promote_all(self) -> Dict[str, Path]:
        """Promote best checkpoint for every registered domain."""
        return {domain: self.promote_best(domain) for domain in self._registry}

    # ── Pruning ────────────────────────────────────────────────────────────────

    def prune(self, domain: str, keep_k: Optional[int] = None) -> int:
        """
        Remove checkpoints beyond keep_k for a domain (deletes files on disk).
        Returns number pruned.
        """
        keep_k  = keep_k or self.top_k
        records = self._registry.get(domain, [])
        to_prune = records[keep_k:]

        for record in to_prune:
            p = Path(record.adapter_path)
            if p.exists() and p.is_dir():
                shutil.rmtree(p)
                logger.debug("Pruned checkpoint: %s", p)

        self._registry[domain] = records[:keep_k]
        return len(to_prune)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_registry(self, path: Optional[str | Path] = None) -> Path:
        """Save checkpoint registry to JSON."""
        path = Path(path) if path else self.output_dir / "checkpoint_registry.json"
        data = {
            "config": {
                "primary_metric":   self.primary_metric,
                "greater_is_better": self.greater_is_better,
                "top_k":            self.top_k,
            },
            "checkpoints": {
                domain: [asdict(r) for r in records]
                for domain, records in self._registry.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Registry saved → %s (%d domains)", path, len(self._registry))
        return path

    def load_registry(self, path: Optional[str | Path] = None) -> None:
        """Load checkpoint registry from JSON."""
        path = Path(path) if path else self.output_dir / "checkpoint_registry.json"
        if not path.exists():
            logger.warning("Registry file not found: %s", path)
            return

        with open(path) as f:
            data = json.load(f)

        self._registry = {}
        for domain, records in data.get("checkpoints", {}).items():
            self._registry[domain] = [CheckpointRecord(**r) for r in records]

        logger.info("Registry loaded from %s (%d domains)", path, len(self._registry))

    # ── Reporting ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        if not self._registry:
            return "CheckpointManager: no checkpoints registered."

        lines = ["CheckpointManager Summary", "=" * 40]
        for domain in sorted(self._registry):
            records = self._registry[domain]
            lines.append(f"\nDomain: {domain} ({len(records)} checkpoints)")
            for i, r in enumerate(records):
                tag = " ← best" if r.is_best else ""
                metric_val = r.metrics.get(self.primary_metric, "N/A")
                metric_str = f"{metric_val:.4f}" if isinstance(metric_val, float) else str(metric_val)
                lines.append(
                    f"  [{i+1}] step={r.step:>5d} epoch={r.epoch:.1f} "
                    f"{self.primary_metric}={metric_str}{tag}"
                )
        return "\n".join(lines)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _sort(self, records: List[CheckpointRecord]) -> List[CheckpointRecord]:
        """Sort records best-first by primary metric."""
        return sorted(
            records,
            key=lambda r: r.metrics.get(
                self.primary_metric,
                float("-inf") if self.greater_is_better else float("inf"),
            ),
            reverse=self.greater_is_better,
        )


# ── HuggingFace TrainerCallback ────────────────────────────────────────────────

class CheckpointCallback:
    """
    HuggingFace TrainerCallback that registers each saved checkpoint
    with the CheckpointManager.

    Usage
    -----
    manager  = CheckpointManager(output_dir="outputs/checkpoints")
    callback = CheckpointCallback(manager, domain="technical")
    trainer  = Trainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        manager: CheckpointManager,
        domain: str,
        extra_metrics_fn=None,    # optional callable(model, eval_dataset) → dict
    ) -> None:
        self.manager          = manager
        self.domain           = domain
        self.extra_metrics_fn = extra_metrics_fn

    def on_save(self, args, state, control, **kwargs):
        """Called by HuggingFace Trainer after each checkpoint save."""
        if not state.best_metric:
            return

        step = state.global_step
        epoch = state.epoch or 0.0

        # Build metrics dict from trainer state logs
        metrics = {"eval_loss": state.best_metric}
        if state.log_history:
            for log in reversed(state.log_history):
                if "eval_loss" in log:
                    metrics["eval_loss"] = log["eval_loss"]
                    break

        # Add extra domain-specific metrics if provided
        if self.extra_metrics_fn is not None:
            try:
                extra = self.extra_metrics_fn()
                metrics.update(extra)
            except Exception as e:
                logger.warning("extra_metrics_fn failed: %s", e)

        adapter_path = Path(args.output_dir) / f"checkpoint-{step}"
        self.manager.register(
            domain=self.domain,
            step=step,
            epoch=epoch,
            metrics=metrics,
            adapter_path=adapter_path,
        )
        self.manager.save_registry()
