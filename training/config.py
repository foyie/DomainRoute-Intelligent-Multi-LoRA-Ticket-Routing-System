"""
training/config.py
───────────────────
LoRA hyperparameter configuration for all four VeriTune domains.

Design rationale
----------------
Each domain has a different complexity profile:

  technical   – nuanced multi-step reasoning, needs higher rank (r=32)
  billing     – moderate complexity, balance of precision and cost (r=24)
  returns     – structured, policy-driven (r=28)
  escalation  – binary classification, keep adapter tiny (r=8)

Sweep grids define ranges tested during Phase 2 hyperparameter search.
The best checkpoint per domain is selected by validation loss + custom metric.

Public API
----------
DomainLoRAConfig          – Pydantic config for a single domain
SweepGrid                 – Defines the hyperparameter search space
TrainingConfig            – Top-level config holding all domains + shared params
get_domain_config(domain) → DomainLoRAConfig
get_sweep_grid(domain)    → SweepGrid
load_training_config(path)→ TrainingConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ── Per-domain LoRA config ─────────────────────────────────────────────────────

@dataclass
class DomainLoRAConfig:
    """LoRA adapter configuration for a single domain."""

    domain: str

    # ── Base model ─────────────────────────────────────────────────────────────
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_qlora: bool = True           # 8-bit quantisation (QLoRA)
    load_in_4bit: bool = False       # 4-bit NF4 (more aggressive compression)

    # ── LoRA adapter parameters ────────────────────────────────────────────────
    lora_r: int = 32                 # rank — controls adapter expressiveness
    lora_alpha: int = 64             # scaling: alpha/r controls effective LR
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"               # "none" | "all" | "lora_only"
    task_type: str = "CAUSAL_LM"

    # ── Training hyperparameters ───────────────────────────────────────────────
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4    # effective batch = 4 × 4 = 16
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ── Sequence length ────────────────────────────────────────────────────────
    max_seq_length: int = 512

    # ── Evaluation & checkpointing ─────────────────────────────────────────────
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3          # keep top-3 checkpoints
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 5   # stop after N evals without improvement

    # ── Logging ───────────────────────────────────────────────────────────────
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    run_name: Optional[str] = None     # auto-set to f"{domain}_lora_r{lora_r}"

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "outputs/checkpoints"
    fp16: bool = True
    bf16: bool = False                 # prefer fp16 for broader GPU compat
    dataloader_num_workers: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.run_name is None:
            self.run_name = f"{self.domain}_lora_r{self.lora_r}"
        self.output_dir = str(
            Path(self.output_dir) / self.domain
        )

    @property
    def effective_batch_size(self) -> int:
        return (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
        )

    def to_peft_config(self):
        """Return a peft.LoraConfig instance."""
        from peft import LoraConfig, TaskType
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
        )

    def to_training_arguments(self):
        """Return a transformers.TrainingArguments instance."""
        from transformers import TrainingArguments
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            fp16=self.fp16,
            bf16=self.bf16,
            evaluation_strategy=self.evaluation_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            logging_steps=self.logging_steps,
            report_to=self.report_to,
            run_name=self.run_name,
            dataloader_num_workers=self.dataloader_num_workers,
            seed=self.seed,
            remove_unused_columns=False,
        )


# ── Sweep grid ─────────────────────────────────────────────────────────────────

@dataclass
class SweepGrid:
    """Defines the hyperparameter search space for a domain sweep."""
    domain: str
    lora_r_values: List[int]           = field(default_factory=lambda: [8, 16, 32, 64])
    learning_rates: List[float]        = field(default_factory=lambda: [1e-4, 2e-4, 5e-4])
    lora_alpha_multipliers: List[int]  = field(default_factory=lambda: [1, 2])  # alpha = r × mult
    dropout_values: List[float]        = field(default_factory=lambda: [0.0, 0.05, 0.10])
    warmup_ratios: List[float]         = field(default_factory=lambda: [0.03, 0.05])

    @property
    def total_combinations(self) -> int:
        return (
            len(self.lora_r_values)
            * len(self.learning_rates)
            * len(self.lora_alpha_multipliers)
            * len(self.dropout_values)
            * len(self.warmup_ratios)
        )

    def iter_configs(self, base_cfg: DomainLoRAConfig):
        """Yield DomainLoRAConfig instances for each sweep combination."""
        import itertools
        for r, lr, alpha_mult, dropout, warmup in itertools.product(
            self.lora_r_values,
            self.learning_rates,
            self.lora_alpha_multipliers,
            self.dropout_values,
            self.warmup_ratios,
        ):
            import copy
            cfg = copy.deepcopy(base_cfg)
            cfg.lora_r         = r
            cfg.lora_alpha     = r * alpha_mult
            cfg.learning_rate  = lr
            cfg.lora_dropout   = dropout
            cfg.warmup_ratio   = warmup
            cfg.run_name       = (
                f"{self.domain}_r{r}_lr{lr:.0e}_a{alpha_mult}_d{dropout}_w{warmup}"
            )
            yield cfg


# ── Domain defaults ────────────────────────────────────────────────────────────

# Pareto-optimal configs found during sweep (r=32 selected for most domains)
DOMAIN_DEFAULTS: Dict[str, dict] = {
    "technical": {
        "lora_r": 32,
        "lora_alpha": 64,
        "learning_rate": 2e-4,
        "lora_dropout": 0.05,
        "num_train_epochs": 3,
        "warmup_ratio": 0.05,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
    },
    "billing": {
        "lora_r": 24,
        "lora_alpha": 48,
        "learning_rate": 2e-4,
        "lora_dropout": 0.05,
        "num_train_epochs": 3,
        "warmup_ratio": 0.05,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
    },
    "returns": {
        "lora_r": 28,
        "lora_alpha": 56,
        "learning_rate": 2e-4,
        "lora_dropout": 0.05,
        "num_train_epochs": 3,
        "warmup_ratio": 0.05,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
    },
    "escalation": {
        "lora_r": 8,
        "lora_alpha": 16,
        "learning_rate": 1e-4,
        "lora_dropout": 0.0,
        "num_train_epochs": 5,           # more epochs for small adapter
        "warmup_ratio": 0.10,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
    },
}

SWEEP_GRIDS: Dict[str, SweepGrid] = {
    "technical": SweepGrid(
        domain="technical",
        lora_r_values=[16, 32, 64],
        learning_rates=[1e-4, 2e-4],
        lora_alpha_multipliers=[2],
        dropout_values=[0.0, 0.05],
        warmup_ratios=[0.05],
    ),
    "billing": SweepGrid(
        domain="billing",
        lora_r_values=[16, 24, 32],
        learning_rates=[1e-4, 2e-4],
        lora_alpha_multipliers=[2],
        dropout_values=[0.0, 0.05],
        warmup_ratios=[0.05],
    ),
    "returns": SweepGrid(
        domain="returns",
        lora_r_values=[16, 28, 32],
        learning_rates=[1e-4, 2e-4],
        lora_alpha_multipliers=[2],
        dropout_values=[0.0, 0.05],
        warmup_ratios=[0.05],
    ),
    "escalation": SweepGrid(
        domain="escalation",
        lora_r_values=[4, 8, 16],
        learning_rates=[5e-5, 1e-4, 2e-4],
        lora_alpha_multipliers=[1, 2],
        dropout_values=[0.0],
        warmup_ratios=[0.05, 0.10],
    ),
}


# ── Public API ─────────────────────────────────────────────────────────────────

def get_domain_config(
    domain: str,
    overrides: Optional[dict] = None,
) -> DomainLoRAConfig:
    """
    Get the production-optimal DomainLoRAConfig for a domain.

    Parameters
    ----------
    domain    : One of technical / billing / returns / escalation
    overrides : Optional dict of field overrides (e.g. {"num_train_epochs": 5})
    """
    if domain not in DOMAIN_DEFAULTS:
        raise ValueError(f"Unknown domain '{domain}'. Valid: {list(DOMAIN_DEFAULTS)}")

    defaults = dict(DOMAIN_DEFAULTS[domain])
    if overrides:
        defaults.update(overrides)

    return DomainLoRAConfig(domain=domain, **defaults)


def get_sweep_grid(domain: str) -> SweepGrid:
    """Get the hyperparameter sweep grid for a domain."""
    if domain not in SWEEP_GRIDS:
        raise ValueError(f"No sweep grid for domain '{domain}'")
    return SWEEP_GRIDS[domain]


@dataclass
class TrainingConfig:
    """Top-level training configuration holding all domain configs."""
    domains: Dict[str, DomainLoRAConfig]
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    data_dir: str = "data/datasets/processed"
    output_dir: str = "outputs/checkpoints"
    wandb_project: str = "veritune"
    seed: int = 42
    run_sweep: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load TrainingConfig from hyperparams.yaml."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        domains = {}
        for domain, dcfg in raw.get("domains", {}).items():
            domains[domain] = DomainLoRAConfig(domain=domain, **dcfg)

        return cls(
            domains=domains,
            base_model=raw.get("base_model", cls.base_model),
            data_dir=raw.get("data_dir", cls.data_dir),
            output_dir=raw.get("output_dir", cls.output_dir),
            wandb_project=raw.get("wandb_project", cls.wandb_project),
            seed=raw.get("seed", cls.seed),
            run_sweep=raw.get("run_sweep", cls.run_sweep),
        )

    @classmethod
    def default(cls) -> "TrainingConfig":
        """Create default TrainingConfig with all domain defaults."""
        return cls(
            domains={d: get_domain_config(d) for d in DOMAIN_DEFAULTS},
        )


def load_training_config(path: str | Path = "config/hyperparams.yaml") -> TrainingConfig:
    """Load training config from YAML, falling back to defaults if file missing."""
    path = Path(path)
    if path.exists():
        return TrainingConfig.from_yaml(path)

    import logging
    logging.getLogger(__name__).warning(
        "hyperparams.yaml not found at %s — using built-in defaults", path
    )
    return TrainingConfig.default()
