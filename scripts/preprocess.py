"""
scripts/preprocess.py
──────────────────────
End-to-end data pipeline for VeriTune Phase 1.

Steps
-----
1. Load raw seed data from data/datasets/raw/
2. Generate synthetic examples per domain (template + optional LLM)
3. Run quality gates (outlier detection, dedup, label noise filtering)
4. Split into train / val / test per domain
5. Save processed JSONL to data/datasets/processed/
6. Print a dataset report

Usage
-----
  # Minimal (template augmentation only, no LLM calls):
  python scripts/preprocess.py

  # With LLM augmentation (requires OPENAI_API_KEY):
  python scripts/preprocess.py --use-llm

  # Custom target size and output dir:
  python scripts/preprocess.py --target 800 --out data/datasets/processed

  # Single domain:
  python scripts/preprocess.py --domain technical
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ── Make sure project root is on the path ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import typer
import yaml
from datasets import Dataset
from rich.console import Console
from rich.table import Table

from data.loaders import load_config, load_jsonl, dataset_stats, _stratified_split
from data.quality_gates import run_quality_gates
from data.synthetic_gen import augment_domain, save_synthetic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess")
console = Console()

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

DOMAINS = ["technical", "billing", "returns", "escalation"]
RAW_DIR = ROOT / "data" / "datasets" / "raw"
PROCESSED_DIR = ROOT / "data" / "datasets" / "processed"


@app.command()
def main(
    domain: str = typer.Option(None, help="Process a single domain (default: all)"),
    target: int = typer.Option(600, help="Target examples per domain before splitting"),
    use_llm: bool = typer.Option(False, help="Use LLM augmentation (requires OPENAI_API_KEY)"),
    llm_fraction: float = typer.Option(0.4, help="Fraction of synthetic data via LLM"),
    out: Path = typer.Option(PROCESSED_DIR, help="Output directory for processed JSONL"),
    config: Path = typer.Option(ROOT / "config" / "domains.yaml", help="Config file path"),
    seed: int = typer.Option(42, help="Random seed"),
    dry_run: bool = typer.Option(False, help="Print plan without writing files"),
) -> None:
    """VeriTune Phase 1 — Data pipeline orchestrator."""

    cfg = load_config(config)
    ds_cfg = cfg["dataset"]

    domains_to_process = [domain] if domain else DOMAINS

    console.rule("[bold]VeriTune — Data Pipeline[/bold]")
    console.print(
        f"Domains: {domains_to_process} | target={target} | "
        f"use_llm={use_llm} | dry_run={dry_run}"
    )

    all_stats: dict = {}

    for dom in domains_to_process:
        console.rule(f"[cyan]Domain: {dom}[/cyan]")

        # ── Step 1: Load seed examples ─────────────────────────────────────────
        seed_path = RAW_DIR / "domain_intent_examples.json"
        seed_records = _load_seed_examples(seed_path, dom)
        n_seed = len(seed_records)
        console.print(f"  Seed examples loaded : {n_seed}")

        # ── Step 2: Augment to reach target size ───────────────────────────────
        n_needed = max(0, target - n_seed)
        if n_needed > 0:
            console.print(f"  Generating {n_needed} synthetic examples (use_llm={use_llm})…")
            if not dry_run:
                synthetic = augment_domain(
                    dom, n_needed, cfg,
                    use_llm=use_llm,
                    llm_fraction=llm_fraction,
                    seed=seed,
                )
            else:
                synthetic = []
        else:
            synthetic = []
            console.print(f"  No augmentation needed (seed={n_seed} ≥ target={target})")

        all_records = seed_records + synthetic
        console.print(f"  Total before QA gates : {len(all_records)}")

        if dry_run:
            console.print("  [yellow]DRY RUN — skipping quality gates and save[/yellow]")
            continue

        # ── Step 3: Quality gates ──────────────────────────────────────────────
        dataset = Dataset.from_list(all_records)
        clean_dataset, report = run_quality_gates(dataset, cfg, embeddings=None)
        console.print(f"  After quality gates   : {len(clean_dataset)}")
        console.print(f"  Noise rate            : {report.noise_rate:.1%}")

        if len(clean_dataset) < ds_cfg.get("min_examples_per_domain", 500):
            console.print(
                f"  [bold red]⚠ WARNING: Only {len(clean_dataset)} examples — "
                f"below minimum {ds_cfg['min_examples_per_domain']}. "
                f"Increase --target or add more seed data.[/bold red]"
            )

        # ── Step 4: Split ──────────────────────────────────────────────────────
        train_ratio = ds_cfg["train_split"] + ds_cfg["val_split"]
        test_size   = ds_cfg["test_split"]

        train_val, test_ds = _stratified_split(
            clean_dataset, test_size=test_size, seed=seed,
        )
        val_size = ds_cfg["val_split"] / train_ratio
        train_ds, val_ds = _stratified_split(
            train_val, test_size=val_size, seed=seed,
        )

        console.print(
            f"  Split → train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
        )

        # ── Step 5: Save ───────────────────────────────────────────────────────
        _save_split(train_ds, out / "train" / f"{dom}_train.jsonl")
        _save_split(val_ds,   out / "val"   / f"{dom}_val.jsonl")
        _save_split(test_ds,  out / "test"  / f"{dom}_test.jsonl")

        all_stats[dom] = {
            "seed":      n_seed,
            "synthetic": len(synthetic),
            "after_qa":  len(clean_dataset),
            "train":     len(train_ds),
            "val":       len(val_ds),
            "test":      len(test_ds),
            "noise_rate": report.noise_rate,
        }

    if not dry_run and all_stats:
        _print_summary_table(all_stats)
        _save_pipeline_report(all_stats, out)

    console.print("\n[bold green]✓ Pipeline complete.[/bold green]")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_seed_examples(path: Path, domain: str) -> list[dict]:
    """Load domain-specific seed examples from the raw JSON file."""
    if not path.exists():
        logger.warning("Seed file not found: %s", path)
        return []

    with open(path) as f:
        all_examples = json.load(f)

    records = all_examples.get(domain, [])
    for r in records:
        r.setdefault("domain", domain)
        r.setdefault("source", "seed")
        r.setdefault("response", "")

    return records


def _save_split(dataset: Dataset, path: Path) -> None:
    """Save a Dataset split to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = dataset.to_pandas().to_dict(orient="records")
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d rows → %s", len(records), path)


def _print_summary_table(stats: dict) -> None:
    table = Table(title="Dataset Pipeline Summary", show_lines=True)
    table.add_column("Domain",     style="cyan")
    table.add_column("Seed",       justify="right")
    table.add_column("Synthetic",  justify="right")
    table.add_column("After QA",   justify="right")
    table.add_column("Train",      justify="right")
    table.add_column("Val",        justify="right")
    table.add_column("Test",       justify="right")
    table.add_column("Noise %",    justify="right")

    for domain, s in stats.items():
        table.add_row(
            domain,
            str(s["seed"]),
            str(s["synthetic"]),
            str(s["after_qa"]),
            str(s["train"]),
            str(s["val"]),
            str(s["test"]),
            f"{s['noise_rate']:.1%}",
        )

    console.print(table)


def _save_pipeline_report(stats: dict, out: Path) -> None:
    report_path = out / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    console.print(f"\nReport saved → [bold]{report_path}[/bold]")


if __name__ == "__main__":
    app()
