"""
data/loaders.py
───────────────
Load raw support-ticket data (CSV / JSONL) into HuggingFace Datasets and
produce stratified train / val / test splits per domain.

Public API
----------
load_raw_csv(path)                  → datasets.Dataset
load_jsonl(path)                    → datasets.Dataset
load_domain_splits(domain, cfg)     → DatasetDict {train, val, test}
load_all_domains(cfg)               → dict[str, DatasetDict]
merge_domains(domain_splits)        → DatasetDict  (combined, shuffled)
dataset_stats(dataset)              → dict
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DOMAINS = ["technical", "billing", "returns", "escalation"]
REQUIRED_COLUMNS = {"text", "domain", "label"}   # label = "resolved" | "escalate"


# ── Config helper ──────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config/domains.yaml") -> dict:
    """Load and return the domains YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Raw loaders ────────────────────────────────────────────────────────────────

def load_raw_csv(path: str | Path, text_col: str = "text") -> Dataset:
    """
    Load a CSV file into a HuggingFace Dataset.

    Expected columns: text, domain, label (optional: response, metadata)
    Missing columns that are required raise ValueError.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Got: {list(df.columns)}")

    df = df.rename(columns={text_col: "text"})
    df = _clean_dataframe(df)

    logger.info("Loaded %d rows from %s", len(df), path)
    return Dataset.from_pandas(df, preserve_index=False)


def load_jsonl(path: str | Path) -> Dataset:
    """
    Load a JSONL file where each line is:
      {"text": "...", "domain": "...", "label": "...", "response": "..."}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

    records: List[dict] = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON at line %d: %s", line_no, e)

    if not records:
        raise ValueError(f"No valid records found in {path}")

    df = pd.DataFrame(records)
    df = _clean_dataframe(df)
    logger.info("Loaded %d records from %s", len(df), path)
    return Dataset.from_pandas(df, preserve_index=False)


def load_domain_splits(
    domain: str,
    cfg: Optional[dict] = None,
    processed_dir: str | Path = "data/datasets/processed",
) -> DatasetDict:
    """
    Load pre-processed train/test JSONL files for a single domain and
    further split the train set into train/val.

    Returns
    -------
    DatasetDict with keys: "train", "val", "test"
    """
    if cfg is None:
        cfg = load_config()

    ds_cfg = cfg["dataset"]
    processed_dir = Path(processed_dir)

    train_path = processed_dir / "train" / f"{domain}_train.jsonl"
    test_path  = processed_dir / "test"  / f"{domain}_test.jsonl"

    train_full = load_jsonl(train_path)
    test_ds    = load_jsonl(test_path)

    # Split train → train + val using stratified shuffle
    val_ratio  = ds_cfg["val_split"] / (ds_cfg["train_split"] + ds_cfg["val_split"])
    train_ds, val_ds = _stratified_split(
        train_full,
        test_size=val_ratio,
        stratify_col="label",
        seed=ds_cfg["random_seed"],
    )

    splits = DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})
    _log_split_stats(domain, splits)
    return splits


def load_all_domains(
    cfg: Optional[dict] = None,
    processed_dir: str | Path = "data/datasets/processed",
    domains: Optional[List[str]] = None,
) -> Dict[str, DatasetDict]:
    """
    Load splits for every domain. Returns dict keyed by domain name.
    Skips domains whose processed files don't exist yet (logs a warning).
    """
    if cfg is None:
        cfg = load_config()
    if domains is None:
        domains = DOMAINS

    result: Dict[str, DatasetDict] = {}
    for domain in domains:
        try:
            result[domain] = load_domain_splits(domain, cfg, processed_dir)
        except FileNotFoundError as e:
            logger.warning("Skipping domain '%s' — files not found: %s", domain, e)

    if not result:
        raise RuntimeError("No domain data loaded. Run scripts/preprocess.py first.")

    return result


def merge_domains(
    domain_splits: Dict[str, DatasetDict],
    seed: int = 42,
) -> DatasetDict:
    """
    Concatenate all domain splits into a single DatasetDict and shuffle.
    Adds a `domain` column if not already present.

    Returns
    -------
    DatasetDict with keys: "train", "val", "test"
    """
    merged: Dict[str, Dataset] = {}
    for split in ("train", "val", "test"):
        parts = []
        for domain, ds_dict in domain_splits.items():
            if split not in ds_dict:
                continue
            ds = ds_dict[split]
            # Ensure domain column is set
            if "domain" not in ds.column_names:
                ds = ds.map(lambda _, d=domain: {"domain": d}, batched=False)
            parts.append(ds)

        if not parts:
            logger.warning("No data for split '%s'", split)
            continue

        combined = concatenate_datasets(parts).shuffle(seed=seed)
        merged[split] = combined
        logger.info(
            "Merged '%s' split: %d total examples across %d domains",
            split, len(combined), len(parts),
        )

    return DatasetDict(merged)


# ── Statistics ─────────────────────────────────────────────────────────────────

def dataset_stats(dataset: Dataset | DatasetDict) -> dict:
    """
    Return a summary dict with row counts, domain distribution, and label balance.
    Works on both Dataset and DatasetDict.
    """
    if isinstance(dataset, DatasetDict):
        return {split: dataset_stats(ds) for split, ds in dataset.items()}

    stats: dict = {"total": len(dataset)}

    if "domain" in dataset.column_names:
        domain_counts = pd.Series(dataset["domain"]).value_counts().to_dict()
        stats["by_domain"] = domain_counts

    if "label" in dataset.column_names:
        label_counts = pd.Series(dataset["label"]).value_counts().to_dict()
        stats["by_label"] = label_counts

    if "text" in dataset.column_names:
        lengths = pd.Series(dataset["text"]).str.len()
        stats["text_length"] = {
            "mean":   round(lengths.mean(), 1),
            "median": round(lengths.median(), 1),
            "p95":    round(lengths.quantile(0.95), 1),
        }

    return stats


# ── Internal helpers ───────────────────────────────────────────────────────────

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a raw DataFrame:
    - Strip whitespace from string columns
    - Drop rows with null text
    - Ensure label column exists (defaults to "resolved")
    - Validate domain values
    """
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Drop empty text
    if "text" in df.columns:
        before = len(df)
        df = df[df["text"].str.len() > 0].copy()
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with empty text", dropped)

    # Default label
    if "label" not in df.columns:
        logger.warning("No 'label' column found; defaulting all labels to 'resolved'")
        df["label"] = "resolved"

    # Normalise label values
    if "label" in df.columns:
        df["label"] = df["label"].str.lower().str.strip()
        valid_labels = {"resolved", "escalate"}
        invalid = df[~df["label"].isin(valid_labels)]
        if len(invalid):
            logger.warning(
                "%d rows have unrecognised labels %s; setting to 'resolved'",
                len(invalid),
                invalid["label"].unique().tolist(),
            )
            df.loc[~df["label"].isin(valid_labels), "label"] = "resolved"

    # Normalise domain values
    if "domain" in df.columns:
        df["domain"] = df["domain"].str.lower().str.strip()
        invalid_domains = df[~df["domain"].isin(DOMAINS + ["unknown"])]
        if len(invalid_domains):
            logger.warning(
                "%d rows have unknown domain values: %s",
                len(invalid_domains),
                invalid_domains["domain"].unique().tolist(),
            )

    return df.reset_index(drop=True)


def _stratified_split(
    dataset: Dataset,
    test_size: float,
    stratify_col: str = "label",
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Perform a stratified split on a Dataset.
    Returns (train_subset, test_subset).
    """
    df = dataset.to_pandas()

    if stratify_col not in df.columns or df[stratify_col].nunique() < 2:
        # Fall back to random split if stratification is not possible
        logger.warning("Cannot stratify on '%s'; using random split", stratify_col)
        split_idx = int(len(df) * (1 - test_size))
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(df, df[stratify_col]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

    return (
        Dataset.from_pandas(train_df, preserve_index=False),
        Dataset.from_pandas(test_df,  preserve_index=False),
    )


def _log_split_stats(domain: str, splits: DatasetDict) -> None:
    for split_name, ds in splits.items():
        label_dist = pd.Series(ds["label"]).value_counts().to_dict() if "label" in ds.column_names else {}
        logger.info(
            "Domain='%s' split='%s' n=%d labels=%s",
            domain, split_name, len(ds), label_dist,
        )
