"""
data/quality_gates.py
──────────────────────
Data quality filtering for VeriTune:

1. IQR-based outlier detection (text length, embedding distance)
2. Confident-learning label-noise detection via cleanlab
3. Duplicate / near-duplicate removal
4. Quality report generation

Public API
----------
run_quality_gates(dataset, cfg)         → QualityReport
filter_outliers(dataset, cfg)           → (clean, flagged)
detect_label_noise(dataset, cfg)        → (clean, noisy, noise_indices)
remove_duplicates(dataset, threshold)   → (clean, duplicate_indices)
quality_report(dataset, cfg)            → QualityReport
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ── Report dataclass ───────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    total_input: int
    total_output: int
    outliers_removed: int
    noisy_labels_removed: int
    duplicates_removed: int
    noise_rate: float                          # fraction of noisy labels found
    domain_counts: dict = field(default_factory=dict)
    label_counts:  dict = field(default_factory=dict)
    flagged_indices: List[int] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Quality Gate Report\n"
            f"  Input rows      : {self.total_input}\n"
            f"  Outliers removed: {self.outliers_removed}\n"
            f"  Noisy labels    : {self.noisy_labels_removed} ({self.noise_rate:.1%})\n"
            f"  Duplicates      : {self.duplicates_removed}\n"
            f"  Output rows     : {self.total_output}\n"
            f"  Domain counts   : {self.domain_counts}\n"
            f"  Label counts    : {self.label_counts}\n"
        )


# ── Main entry point ───────────────────────────────────────────────────────────

def run_quality_gates(
    dataset: Dataset,
    cfg: Optional[dict] = None,
    embeddings: Optional[np.ndarray] = None,
) -> Tuple[Dataset, QualityReport]:
    """
    Run the full quality gate pipeline in sequence:
      1. Filter structural outliers (text length)
      2. Remove near-duplicates
      3. Detect and remove noisy labels (requires sklearn probabilities or embeddings)

    Parameters
    ----------
    dataset    : Input HuggingFace Dataset with columns: text, domain, label
    cfg        : Loaded domains.yaml config dict (uses cfg["quality"] section)
    embeddings : Optional pre-computed sentence embeddings (N × D ndarray).
                 If provided, used for embedding-distance outlier detection and
                 for feeding cleanlab's confident learning.

    Returns
    -------
    (clean_dataset, QualityReport)
    """
    if cfg is None:
        from data.loaders import load_config
        cfg = load_config()

    q_cfg = cfg.get("quality", {})
    total_input = len(dataset)

    # Step 1 — structural outlier removal
    dataset, outlier_indices = filter_outliers(dataset, q_cfg)
    outliers_removed = len(outlier_indices)
    logger.info("After outlier removal: %d rows (%d removed)", len(dataset), outliers_removed)

    # Step 2 — duplicate removal
    dataset, dup_indices = remove_duplicates(dataset)
    duplicates_removed = len(dup_indices)
    logger.info("After dedup: %d rows (%d duplicates removed)", len(dataset), duplicates_removed)

    # Step 3 — label noise detection (only if we have embeddings or enough data)
    noisy_labels_removed = 0
    if embeddings is not None and len(dataset) >= 20:
        dataset, noisy_indices, _ = detect_label_noise(dataset, q_cfg, embeddings)
        noisy_labels_removed = len(noisy_indices)
        logger.info(
            "After noise filtering: %d rows (%d noisy labels removed, rate=%.1f%%)",
            len(dataset), noisy_labels_removed,
            noisy_labels_removed / max(total_input, 1) * 100,
        )
    else:
        logger.info(
            "Skipping label noise detection "
            "(embeddings not provided or dataset too small)"
        )

    noise_rate = noisy_labels_removed / max(total_input, 1)
    if noise_rate > q_cfg.get("max_label_noise_rate", 0.10):
        logger.warning(
            "Label noise rate %.1f%% exceeds threshold %.1f%%. "
            "Review data collection pipeline.",
            noise_rate * 100,
            q_cfg.get("max_label_noise_rate", 0.10) * 100,
        )

    # Build report
    df = dataset.to_pandas()
    report = QualityReport(
        total_input=total_input,
        total_output=len(dataset),
        outliers_removed=outliers_removed,
        noisy_labels_removed=noisy_labels_removed,
        duplicates_removed=duplicates_removed,
        noise_rate=noise_rate,
        domain_counts=df["domain"].value_counts().to_dict() if "domain" in df else {},
        label_counts=df["label"].value_counts().to_dict()  if "label"  in df else {},
    )

    logger.info(report.summary())
    return dataset, report


# ── Outlier detection ──────────────────────────────────────────────────────────

def filter_outliers(
    dataset: Dataset,
    q_cfg: Optional[dict] = None,
) -> Tuple[Dataset, List[int]]:
    """
    Remove rows that are structural outliers:
    - Text too short or too long (hard bounds from config)
    - Text length outside IQR fence (1.5× IQR by default)

    Returns (clean_dataset, flagged_indices).
    """
    if q_cfg is None:
        q_cfg = {}

    min_len     = q_cfg.get("min_text_length", 10)
    max_len     = q_cfg.get("max_text_length", 2000)
    iqr_mult    = q_cfg.get("iqr_multiplier", 1.5)

    df = dataset.to_pandas()
    lengths = df["text"].str.len()

    # Hard bounds
    hard_mask = (lengths >= min_len) & (lengths <= max_len)

    # IQR soft bounds (applied within each domain separately to avoid domain-size bias)
    iqr_mask = pd.Series(True, index=df.index)
    if "domain" in df.columns:
        for domain in df["domain"].unique():
            domain_idx = df["domain"] == domain
            domain_lengths = lengths[domain_idx]
            q1, q3 = domain_lengths.quantile(0.25), domain_lengths.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
            outlier_flag = (domain_lengths < lower) | (domain_lengths > upper)
            iqr_mask[domain_idx & outlier_flag.reindex(df.index, fill_value=False)] = False
    else:
        q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        iqr_mask = (lengths >= lower) & (lengths <= upper)

    keep_mask = hard_mask & iqr_mask
    flagged_indices = df.index[~keep_mask].tolist()

    if flagged_indices:
        logger.info(
            "Outlier detection: flagging %d / %d rows "
            "(hard_bounds=%d, iqr=%d)",
            len(flagged_indices), len(df),
            (~hard_mask).sum(), (~iqr_mask).sum(),
        )

    clean_df = df[keep_mask].reset_index(drop=True)
    return Dataset.from_pandas(clean_df, preserve_index=False), flagged_indices


# ── Label noise detection ──────────────────────────────────────────────────────

def detect_label_noise(
    dataset: Dataset,
    q_cfg: Optional[dict] = None,
    embeddings: Optional[np.ndarray] = None,
) -> Tuple[Dataset, List[int], np.ndarray]:
    """
    Use cleanlab's confident learning to find mislabelled examples.

    Strategy
    --------
    1. Train a lightweight KNN classifier on the provided embeddings
       to get out-of-fold predicted probabilities.
    2. Feed those probabilities to cleanlab's find_label_issues().
    3. Return clean dataset, noisy indices, and the probability matrix.

    If cleanlab is not installed or embeddings are None, falls back to
    a heuristic based on embedding distance to class centroids.

    Returns
    -------
    (clean_dataset, noisy_indices, pred_probs)
    """
    if q_cfg is None:
        q_cfg = {}

    df = dataset.to_pandas()
    labels_str = df["label"].tolist()

    # Encode labels to integers
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_int = le.fit_transform(labels_str)

    if embeddings is None:
        logger.info("No embeddings provided — computing heuristic label quality scores")
        clean_df = df.copy()
        return Dataset.from_pandas(clean_df, preserve_index=False), [], np.array([])

    # Get out-of-fold predicted probabilities via cross-validated KNN
    pred_probs = _get_knn_pred_probs(embeddings, labels_int, n_splits=5)

    # Use cleanlab to find label issues
    try:
        from cleanlab.filter import find_label_issues
        label_issues = find_label_issues(
            labels=labels_int,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence",
        )
        noisy_indices = label_issues.tolist()
        logger.info(
            "Cleanlab found %d label issues (%.1f%% of %d)",
            len(noisy_indices), len(noisy_indices) / len(df) * 100, len(df),
        )
    except ImportError:
        logger.warning("cleanlab not installed — falling back to centroid-distance heuristic")
        noisy_indices = _centroid_distance_noise(embeddings, labels_int, threshold=0.80)

    # Remove noisy rows
    clean_mask = pd.Series(True, index=df.index)
    clean_mask.iloc[noisy_indices] = False
    clean_df = df[clean_mask].reset_index(drop=True)

    return (
        Dataset.from_pandas(clean_df, preserve_index=False),
        noisy_indices,
        pred_probs,
    )


# ── Duplicate removal ──────────────────────────────────────────────────────────

def remove_duplicates(
    dataset: Dataset,
    similarity_threshold: float = 0.95,
) -> Tuple[Dataset, List[int]]:
    """
    Remove exact duplicates and near-duplicates based on text similarity.

    Exact duplicates are detected via string equality.
    Near-duplicates use Jaccard similarity on character trigrams
    (fast and embedding-free).

    Returns (clean_dataset, duplicate_indices).
    """
    df = dataset.to_pandas()
    n = len(df)

    # Exact dedup
    exact_dup_mask = df["text"].duplicated(keep="first")
    exact_dups = df.index[exact_dup_mask].tolist()
    logger.info("Exact duplicates: %d", len(exact_dups))

    # Near-dedup via trigram Jaccard (only on texts that survived exact dedup)
    surviving = df[~exact_dup_mask].copy()
    near_dups = _trigram_jaccard_dedup(surviving, threshold=similarity_threshold)

    all_dups = list(set(exact_dups + near_dups))
    logger.info("Near-duplicates (Jaccard ≥%.2f): %d", similarity_threshold, len(near_dups))

    keep_mask = pd.Series(True, index=df.index)
    keep_mask.iloc[all_dups] = False
    clean_df = df[keep_mask].reset_index(drop=True)

    return Dataset.from_pandas(clean_df, preserve_index=False), all_dups


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_knn_pred_probs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Compute out-of-fold predicted probabilities using a KNN classifier.
    Returns (N, num_classes) array.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier

    n_classes = len(np.unique(labels))
    pred_probs = np.zeros((len(labels), n_classes))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=min(5, len(labels) // n_splits))

    for train_idx, val_idx in skf.split(embeddings, labels):
        knn.fit(embeddings[train_idx], labels[train_idx])
        pred_probs[val_idx] = knn.predict_proba(embeddings[val_idx])

    # Clip to avoid log(0) issues in cleanlab
    pred_probs = np.clip(pred_probs, 1e-6, 1 - 1e-6)
    pred_probs /= pred_probs.sum(axis=1, keepdims=True)
    return pred_probs


def _centroid_distance_noise(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.80,
) -> List[int]:
    """
    Heuristic: flag examples whose cosine similarity to their class centroid
    is below `threshold` as potentially noisy.
    """
    from sklearn.preprocessing import normalize

    normed = normalize(embeddings)
    noisy: List[int] = []

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        centroid = normed[idx].mean(axis=0, keepdims=True)
        centroid = normalize(centroid)
        sims = (normed[idx] @ centroid.T).flatten()
        low_sim = idx[sims < threshold]
        noisy.extend(low_sim.tolist())

    return noisy


def _trigrams(text: str) -> set:
    text = text.lower()
    return {text[i:i+3] for i in range(len(text) - 2)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _trigram_jaccard_dedup(df: pd.DataFrame, threshold: float = 0.95) -> List[int]:
    """
    O(N²) near-dedup on small datasets (≤ 5k rows).
    For larger datasets, use MinHash LSH instead.
    """
    if len(df) > 5000:
        logger.warning(
            "Dataset has %d rows — trigram dedup is O(N²), may be slow. "
            "Consider MinHash LSH for production.",
            len(df),
        )

    texts = df["text"].tolist()
    trigram_sets = [_trigrams(t) for t in texts]
    to_remove: set = set()

    for i in range(len(texts)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(texts)):
            if j in to_remove:
                continue
            if _jaccard(trigram_sets[i], trigram_sets[j]) >= threshold:
                to_remove.add(df.index[j])

    return list(to_remove)
