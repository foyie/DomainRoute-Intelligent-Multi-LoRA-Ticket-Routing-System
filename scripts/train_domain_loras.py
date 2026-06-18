"""
scripts/train_domain_loras.py
──────────────────────────────
Orchestrates LoRA fine-tuning across all four VeriTune domains.

Features
--------
- Sequential or parallel domain training (parallel via subprocess)
- Shared CheckpointManager across all domains
- Optional hyperparameter sweep mode (all rank/lr combinations)
- Semantic drift monitoring with configurable probe set
- Rich CLI progress reporting + final summary table
- Promotes best checkpoint for each domain on completion

Usage
-----
  # Train all domains with default configs:
  python scripts/train_domain_loras.py

  # Single domain:
  python scripts/train_domain_loras.py --domain technical

  # Run full hyperparameter sweep:
  python scripts/train_domain_loras.py --sweep

  # Custom config + no W&B:
  python scripts/train_domain_loras.py --config config/hyperparams.yaml --no-wandb

  # Dry run (validate config, print plan, don't train):
  python scripts/train_domain_loras.py --dry-run
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_domain_loras")
console = Console()

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

DOMAINS = ["technical", "billing", "returns", "escalation"]
PROCESSED_DIR = ROOT / "data" / "datasets" / "processed"
OUTPUT_DIR    = ROOT / "outputs" / "checkpoints"


@app.command()
def main(
    domain: Optional[str] = typer.Option(None, help="Single domain to train (default: all)"),
    config: Path = typer.Option(ROOT / "config" / "hyperparams.yaml", help="hyperparams.yaml path"),
    sweep: bool = typer.Option(False, "--sweep", help="Run full hyperparameter sweep"),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="Disable W&B logging"),
    processed_dir: Path = typer.Option(PROCESSED_DIR, help="Processed data directory"),
    output_dir: Path = typer.Option(OUTPUT_DIR, help="Checkpoint output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan without training"),
    n_drift_probes: int = typer.Option(50, help="Number of probe texts for drift tracking"),
) -> None:
    """VeriTune Phase 2 — Domain LoRA training orchestrator."""

    from training.config import load_training_config, get_domain_config, get_sweep_grid
    from training.checkpoint_manager import CheckpointManager

    console.rule("[bold]VeriTune — LoRA Training Orchestrator[/bold]")

    # ── Load config ────────────────────────────────────────────────────────────
    training_cfg = load_training_config(config)
    domains_to_train = [domain] if domain else DOMAINS

    if no_wandb:
        for d in domains_to_train:
            if d in training_cfg.domains:
                training_cfg.domains[d].report_to = []

    # ── Shared checkpoint manager ──────────────────────────────────────────────
    manager = CheckpointManager(
        output_dir=output_dir,
        top_k=3,
        primary_metric="eval_loss",
    )

    # Try loading existing registry
    registry_path = output_dir / "checkpoint_registry.json"
    if registry_path.exists():
        manager.load_registry(registry_path)
        console.print(f"Loaded existing registry: {registry_path}")

    # ── Print training plan ────────────────────────────────────────────────────
    _print_training_plan(domains_to_train, training_cfg, sweep, dry_run)

    if dry_run:
        console.print("\n[yellow]DRY RUN — exiting without training.[/yellow]")
        return

    # ── Training loop ──────────────────────────────────────────────────────────
    results: dict = {}
    start_total = time.time()

    for dom in domains_to_train:
        console.rule(f"[cyan]Training domain: {dom}[/cyan]")

        domain_cfg = training_cfg.domains.get(dom) or get_domain_config(dom)

        # Load drift probe texts from validation set
        drift_probes = _load_drift_probes(dom, processed_dir, n=n_drift_probes)

        if sweep:
            # ── Hyperparameter sweep ───────────────────────────────────────────
            grid = get_sweep_grid(dom)
            console.print(
                f"  Sweep: {grid.total_combinations} combinations "
                f"({len(grid.lora_r_values)} ranks × {len(grid.learning_rates)} LRs × ...)"
            )
            sweep_results = _run_sweep(
                dom, grid, domain_cfg, manager, processed_dir, drift_probes
            )
            results[dom] = {"sweep": sweep_results, "best": manager.get_best(dom)}
        else:
            # ── Single run with optimal config ─────────────────────────────────
            t0 = time.time()
            metrics = _train_single(dom, domain_cfg, manager, processed_dir, drift_probes)
            elapsed = time.time() - t0
            results[dom] = {"metrics": metrics, "elapsed_s": elapsed}

        # Promote best checkpoint
        try:
            best_path = manager.promote_best(dom)
            console.print(f"  ✓ Best adapter promoted → [bold]{best_path}[/bold]")
        except Exception as e:
            console.print(f"  [yellow]Could not promote best: {e}[/yellow]")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - start_total
    _print_results_table(results, manager)
    console.print(f"\nTotal time: {total_elapsed/60:.1f} min")

    # Save final registry
    manager.save_registry(registry_path)
    console.print(f"Registry saved → [bold]{registry_path}[/bold]")
    console.print("\n[bold green]✓ Training complete.[/bold green]")


# ── Training functions ─────────────────────────────────────────────────────────

def _train_single(
    domain: str,
    domain_cfg,
    manager,
    processed_dir: Path,
    drift_probes: list,
) -> dict:
    """Train a single domain with a fixed config. Returns eval metrics."""
    from training.trainer import train_domain

    try:
        _, metrics = train_domain(
            domain=domain,
            cfg=domain_cfg,
            checkpoint_manager=manager,
            drift_probe_texts=drift_probes,
            processed_dir=str(processed_dir),
        )
        console.print(
            f"  ✓ eval_loss={metrics.get('eval_loss', 'N/A'):.4f}"
        )
        return metrics
    except Exception as e:
        logger.error("Training failed for domain '%s': %s", domain, e, exc_info=True)
        console.print(f"  [bold red]✗ Training failed: {e}[/bold red]")
        return {"error": str(e)}
    finally:
        # Free GPU memory between domains — critical on Colab T4 (15 GB VRAM)
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info(
                    "GPU memory freed: %.1f GB available",
                    torch.cuda.mem_get_info()[0] / 1e9,
                )
        except Exception:
            pass


def _run_sweep(
    domain: str,
    grid,
    base_cfg,
    manager,
    processed_dir: Path,
    drift_probes: list,
) -> list:
    """Run hyperparameter sweep for a domain. Returns list of (config, metrics)."""
    from training.trainer import train_domain
    from data.loaders import load_domain_splits

    # Pre-load data once
    data_splits = load_domain_splits(domain, processed_dir=str(processed_dir))

    sweep_results = []
    for i, sweep_cfg in enumerate(grid.iter_configs(base_cfg)):
        console.print(
            f"  Sweep [{i+1}/{grid.total_combinations}] "
            f"r={sweep_cfg.lora_r} lr={sweep_cfg.learning_rate:.0e} "
            f"dropout={sweep_cfg.lora_dropout}"
        )
        try:
            _, metrics = train_domain(
                domain=domain,
                cfg=sweep_cfg,
                data_splits=data_splits,
                checkpoint_manager=manager,
                drift_probe_texts=drift_probes,
            )
            sweep_results.append({
                "run_name": sweep_cfg.run_name,
                "lora_r": sweep_cfg.lora_r,
                "learning_rate": sweep_cfg.learning_rate,
                "lora_dropout": sweep_cfg.lora_dropout,
                "metrics": metrics,
            })
        except Exception as e:
            logger.warning("Sweep run failed: %s — %s", sweep_cfg.run_name, e)
            sweep_results.append({"run_name": sweep_cfg.run_name, "error": str(e)})

    # Save sweep results
    sweep_path = Path("outputs/results") / f"{domain}_sweep_results.json"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    console.print(f"  Sweep results saved → {sweep_path}")

    return sweep_results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_drift_probes(domain: str, processed_dir: Path, n: int = 50) -> list:
    """Load n validation texts to use as semantic drift probe set."""
    val_path = processed_dir / "val" / f"{domain}_val.jsonl"
    if not val_path.exists():
        # Fall back to test set
        val_path = processed_dir / "test" / f"{domain}_test.jsonl"
    if not val_path.exists():
        logger.warning("No probe texts found for domain '%s' — drift tracking disabled", domain)
        return []

    probes = []
    with open(val_path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                probes.append(record.get("text", ""))
                if len(probes) >= n:
                    break

    logger.info("Loaded %d drift probe texts for domain='%s'", len(probes), domain)
    return probes


def _print_training_plan(domains: list, cfg, sweep: bool, dry_run: bool) -> None:
    table = Table(title="Training Plan", show_lines=True)
    table.add_column("Domain",     style="cyan")
    table.add_column("LoRA rank",  justify="right")
    table.add_column("LR",         justify="right")
    table.add_column("Epochs",     justify="right")
    table.add_column("Eff. batch", justify="right")
    table.add_column("QLoRA",      justify="center")
    table.add_column("Mode")

    from training.config import get_domain_config
    for dom in domains:
        domain_cfg = cfg.domains.get(dom) or get_domain_config(dom)
        mode = "SWEEP" if sweep else "SINGLE"
        table.add_row(
            dom,
            str(domain_cfg.lora_r),
            f"{domain_cfg.learning_rate:.0e}",
            str(domain_cfg.num_train_epochs),
            str(domain_cfg.effective_batch_size),
            "✓" if domain_cfg.use_qlora else "✗",
            f"[yellow]{mode}[/yellow]" if sweep else mode,
        )

    console.print(table)
    if dry_run:
        console.print("[bold yellow]Mode: DRY RUN[/bold yellow]")


def _print_results_table(results: dict, manager) -> None:
    table = Table(title="Training Results", show_lines=True)
    table.add_column("Domain",     style="cyan")
    table.add_column("eval_loss",  justify="right")
    table.add_column("Best step",  justify="right")
    table.add_column("Time (min)", justify="right")
    table.add_column("Status")

    for domain, result in results.items():
        best = manager.get_best(domain)
        eval_loss = "N/A"
        best_step = "N/A"

        if "error" in result.get("metrics", {}):
            status = "[red]FAILED[/red]"
        else:
            status = "[green]✓ OK[/green]"
            if best:
                eval_loss = f"{best.metrics.get('eval_loss', 0):.4f}"
                best_step = str(best.step)

        elapsed_min = result.get("elapsed_s", 0) / 60
        table.add_row(
            domain,
            eval_loss,
            best_step,
            f"{elapsed_min:.1f}" if elapsed_min > 0 else "—",
            status,
        )

    console.print(table)


if __name__ == "__main__":
    app()
