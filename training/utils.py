"""
training/utils.py
──────────────────
Shared utilities for the VeriTune training pipeline:

- Device detection and memory reporting
- QLoRA (quantised LoRA) model loading
- Reproducibility seeding
- Tokeniser helpers
- Prompt formatting per domain

Public API
----------
detect_device()                          → str  ("cuda" | "mps" | "cpu")
get_gpu_memory_info()                    → dict
set_seed(seed)                           → None
load_base_model(cfg)                     → (model, tokenizer)
load_qlora_model(cfg)                    → (model, tokenizer)
apply_lora(model, lora_cfg)             → PeftModel
format_prompt(ticket, domain, response) → str
count_trainable_params(model)           → (trainable, total, pct)
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "### System:\n"
    "You are a helpful customer support agent for {domain} issues. "
    "Respond clearly, empathetically, and with concrete action.\n\n"
    "### Customer:\n{ticket}\n\n"
    "### Agent:\n{response}"
)

INFERENCE_TEMPLATE = (
    "### System:\n"
    "You are a helpful customer support agent for {domain} issues. "
    "Respond clearly, empathetically, and with concrete action.\n\n"
    "### Customer:\n{ticket}\n\n"
    "### Agent:\n"
)

DOMAIN_DESCRIPTIONS = {
    "technical":  "technical support and device troubleshooting",
    "billing":    "billing, payments, and subscription",
    "returns":    "returns, exchanges, and shipping",
    "escalation": "urgent or sensitive customer escalation",
}


# ── Device detection ───────────────────────────────────────────────────────────

def detect_device() -> str:
    """Return the best available compute device: 'cuda', 'mps', or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info("Device detected: %s", device)
        return device
    except ImportError:
        logger.warning("torch not installed — defaulting to cpu")
        return "cpu"


def get_gpu_memory_info() -> dict:
    """Return GPU memory stats (used/total in GB). Returns empty dict on CPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        mem = torch.cuda.mem_get_info()   # (free, total) in bytes
        return {
            "free_gb":  round(mem[0] / 1e9, 2),
            "total_gb": round(mem[1] / 1e9, 2),
            "used_gb":  round((mem[1] - mem[0]) / 1e9, 2),
        }
    except Exception as e:
        logger.debug("GPU memory query failed: %s", e)
        return {}


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.debug("Seed set to %d", seed)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_base_model(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    device_map: str = "auto",
) -> Tuple:
    """
    Load the base model and tokeniser (no quantisation).
    Use for CPU/MPS environments or when QLoRA is not needed.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model: %s", model_name)

    tokenizer = _load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=True,
    )
    logger.info("Base model loaded. Params: %s", _format_param_count(model))
    return model, tokenizer


def load_qlora_model(
    cfg,                              # DomainLoRAConfig
    device_map: str = "auto",
) -> Tuple:
    """
    Load model in 8-bit QLoRA mode using bitsandbytes.
    Reduces VRAM from ~14GB → ~6GB for Mistral-7B.

    Returns (quantised_base_model, tokenizer) — LoRA not yet applied.
    Call apply_lora() next.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "transformers and bitsandbytes are required for QLoRA. "
            "Run: pip install transformers bitsandbytes"
        ) from e

    logger.info(
        "Loading QLoRA model: %s (load_in_4bit=%s)",
        cfg.base_model, cfg.load_in_4bit,
    )

    if cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        # 8-bit quantisation (QLoRA default)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = _load_tokenizer(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Required for gradient checkpointing with quantised models
    model.enable_input_require_grads()

    trainable, total, pct = count_trainable_params(model)
    logger.info(
        "QLoRA base model loaded. Trainable: %s / %s (%.1f%%)",
        _fmt(trainable), _fmt(total), pct,
    )
    return model, tokenizer


def apply_lora(model, cfg) -> "PeftModel":
    """
    Wrap the base model with a LoRA adapter from DomainLoRAConfig.
    Call after load_qlora_model() or load_base_model().
    """
    from peft import get_peft_model, prepare_model_for_kbit_training

    # Prepare for k-bit training (handles gradient checkpointing setup)
    if cfg.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    lora_cfg = cfg.to_peft_config()
    model = get_peft_model(model, lora_cfg)

    trainable, total, pct = count_trainable_params(model)
    logger.info(
        "LoRA applied (r=%d, alpha=%d). Trainable: %s / %s (%.2f%%)",
        cfg.lora_r, cfg.lora_alpha,
        _fmt(trainable), _fmt(total), pct,
    )
    return model


# ── Tokeniser helpers ──────────────────────────────────────────────────────────

def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        trust_remote_code=True,
    )
    # Mistral uses eos as pad token if pad not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def tokenize_example(
    example: dict,
    tokenizer,
    max_length: int = 512,
    domain: str = "technical",
) -> dict:
    """
    Tokenise a single training example.
    Masks the prompt tokens so loss is only computed on the response.
    """
    prompt = format_prompt(
        ticket=example["text"],
        domain=domain,
        response="",   # empty — will be filled below for full sequence
    )
    full_text = format_prompt(
        ticket=example["text"],
        domain=domain,
        response=example.get("response", ""),
    )

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
    )

    input_ids      = full_ids["input_ids"]
    attention_mask = full_ids["attention_mask"]

    # Build labels: mask prompt tokens with -100 (ignored in loss)
    labels = list(input_ids)
    prompt_len = len(prompt_ids)
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


def build_tokenize_fn(tokenizer, max_length: int = 512, domain: str = "technical"):
    """Return a batched tokenisation function for use with dataset.map()."""
    def tokenize_batch(examples):
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        for text, response in zip(examples["text"], examples.get("response", [""] * len(examples["text"]))):
            tok = tokenize_example(
                {"text": text, "response": response},
                tokenizer,
                max_length=max_length,
                domain=domain,
            )
            results["input_ids"].append(tok["input_ids"])
            results["attention_mask"].append(tok["attention_mask"])
            results["labels"].append(tok["labels"])
        return results
    return tokenize_batch


# ── Prompt formatting ──────────────────────────────────────────────────────────

def format_prompt(ticket: str, domain: str, response: str = "") -> str:
    """Format a ticket + response into the Mistral instruction template."""
    domain_desc = DOMAIN_DESCRIPTIONS.get(domain, domain)
    template = PROMPT_TEMPLATE if response else INFERENCE_TEMPLATE
    return template.format(
        domain=domain_desc,
        ticket=ticket.strip(),
        response=response.strip(),
    )


def format_inference_prompt(ticket: str, domain: str) -> str:
    """Format a ticket for inference (no response field)."""
    return format_prompt(ticket, domain, response="")


# ── Parameter counting ─────────────────────────────────────────────────────────

def count_trainable_params(model) -> Tuple[int, int, float]:
    """
    Count trainable vs total parameters.

    Returns
    -------
    (trainable_params, total_params, pct_trainable)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100 * trainable / total if total > 0 else 0.0
    return trainable, total, pct


def _fmt(n: int) -> str:
    """Format large numbers: 7_241_748 → '7.24M'"""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def _format_param_count(model) -> str:
    trainable, total, pct = count_trainable_params(model)
    return f"{_fmt(trainable)} / {_fmt(total)} trainable ({pct:.2f}%)"
