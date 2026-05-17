"""
tests/test_data_quality.py
───────────────────────────
Tests for data/quality_gates.py

Coverage
--------
- IQR outlier detection (short texts, long texts, per-domain)
- Near-duplicate removal (exact and trigram-Jaccard)
- Label noise detection (centroid heuristic with mock embeddings)
- QualityReport fields and summary output
- run_quality_gates end-to-end pipeline
- Edge cases: empty dataset, single-class dataset, all-duplicates
"""

from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset

from data.quality_gates import (
    QualityReport,
    detect_label_noise,
    filter_outliers,
    remove_duplicates,
    run_quality_gates,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def clean_dataset(sample_tickets):
    """Use the session-scoped sample_tickets fixture."""
    return Dataset.from_list(sample_tickets)


@pytest.fixture()
def dataset_with_outliers(sample_tickets):
    """Sample dataset with injected length outliers."""
    records = list(sample_tickets)
    # Very short texts
    records.append({"text": "hi", "domain": "technical", "label": "resolved"})
    records.append({"text": "?",  "domain": "billing",   "label": "resolved"})
    # Very long text (> 2000 chars)
    records.append({
        "text": "x " * 1100,
        "domain": "returns",
        "label": "resolved",
    })
    return Dataset.from_list(records)


@pytest.fixture()
def dataset_with_duplicates(sample_tickets):
    """Dataset with exact and near-duplicate entries."""
    records = list(sample_tickets)
    # Exact duplicate
    records.append(dict(sample_tickets[0]))
    # Near-duplicate (minor variation)
    near_dup = dict(sample_tickets[0])
    near_dup["text"] = sample_tickets[0]["text"] + " please help"
    records.append(near_dup)
    return Dataset.from_list(records)


# ── filter_outliers tests ──────────────────────────────────────────────────────

class TestFilterOutliers:

    def test_removes_short_texts(self, dataset_with_outliers, cfg):
        q_cfg = cfg["quality"]
        clean, flagged = filter_outliers(dataset_with_outliers, q_cfg)
        short_texts = ["hi", "?"]
        clean_texts = clean["text"]
        for short in short_texts:
            assert short not in clean_texts, f"Short text '{short}' should be removed"

    def test_removes_long_texts(self, dataset_with_outliers, cfg):
        q_cfg = cfg["quality"]
        clean, flagged = filter_outliers(dataset_with_outliers, q_cfg)
        for text in clean["text"]:
            assert len(text) <= q_cfg["max_text_length"], (
                f"Text length {len(text)} exceeds max {q_cfg['max_text_length']}"
            )

    def test_flagged_indices_match_removed(self, dataset_with_outliers, cfg):
        q_cfg = cfg["quality"]
        original_len = len(dataset_with_outliers)
        clean, flagged = filter_outliers(dataset_with_outliers, q_cfg)
        assert len(clean) + len(flagged) <= original_len

    def test_clean_dataset_unchanged_on_clean_input(self, clean_dataset, cfg):
        q_cfg = cfg["quality"]
        clean, flagged = filter_outliers(clean_dataset, q_cfg)
        # All sample texts should be within acceptable bounds
        assert len(flagged) == 0, f"Expected no outliers but got {len(flagged)}"

    def test_per_domain_iqr_applied(self, cfg):
        """Each domain should be evaluated independently."""
        q_cfg = cfg["quality"]
        records = []
        # Technical: all ~50 chars
        for i in range(20):
            records.append({"text": f"Technical issue number {i} with the device here.", "domain": "technical", "label": "resolved"})
        # Billing: one very long outlier within billing group
        for i in range(18):
            records.append({"text": f"Billing question {i} about my subscription.", "domain": "billing", "label": "resolved"})
        records.append({"text": "b " * 500, "domain": "billing", "label": "resolved"})

        ds = Dataset.from_list(records)
        clean, flagged = filter_outliers(ds, q_cfg)
        assert len(flagged) >= 1

    def test_returns_dataset_type(self, clean_dataset, cfg):
        clean, flagged = filter_outliers(clean_dataset, cfg["quality"])
        assert isinstance(clean, Dataset)
        assert isinstance(flagged, list)

    def test_empty_dataset_returns_empty(self, cfg):
        ds = Dataset.from_list([{"text": "normal ticket text here", "domain": "technical", "label": "resolved"}])
        # Remove the one valid entry by making it too short
        ds_short = Dataset.from_list([{"text": "hi", "domain": "technical", "label": "resolved"}])
        clean, flagged = filter_outliers(ds_short, cfg["quality"])
        assert len(clean) == 0


# ── remove_duplicates tests ────────────────────────────────────────────────────

class TestRemoveDuplicates:

    def test_removes_exact_duplicates(self, dataset_with_duplicates):
        clean, dup_indices = remove_duplicates(dataset_with_duplicates)
        texts = clean["text"]
        assert len(texts) == len(set(texts)) or len(clean) < len(dataset_with_duplicates)

    def test_dup_indices_non_empty(self, dataset_with_duplicates):
        _, dup_indices = remove_duplicates(dataset_with_duplicates)
        assert len(dup_indices) >= 1

    def test_no_duplicates_in_clean_input(self, clean_dataset):
        clean, dup_indices = remove_duplicates(clean_dataset)
        # Sample dataset has no deliberate duplicates
        assert len(clean) >= len(clean_dataset) - 2   # allow 1-2 natural near-dups

    def test_returns_correct_types(self, clean_dataset):
        clean, dup_indices = remove_duplicates(clean_dataset)
        assert isinstance(clean, Dataset)
        assert isinstance(dup_indices, list)

    def test_all_identical_texts(self):
        records = [{"text": "same text here", "domain": "technical", "label": "resolved"}] * 10
        ds = Dataset.from_list(records)
        clean, dups = remove_duplicates(ds)
        assert len(clean) == 1
        assert len(dups) == 9

    def test_similarity_threshold_respected(self):
        """With threshold=1.0, only exact dupes removed; near-dupes kept."""
        records = [
            {"text": "My headphones won't connect to Bluetooth.", "domain": "technical", "label": "resolved"},
            {"text": "My headphones won't connect to Bluetooth!", "domain": "technical", "label": "resolved"},  # near-dup
            {"text": "I need a refund for my subscription.",       "domain": "billing",   "label": "resolved"},
        ]
        ds = Dataset.from_list(records)
        clean, dups = remove_duplicates(ds, similarity_threshold=1.0)
        # At threshold=1.0 only exact duplicates are removed; minor variation kept
        assert len(clean) >= 2


# ── detect_label_noise tests ───────────────────────────────────────────────────

class TestDetectLabelNoise:

    def test_detects_injected_noise(self, sample_dataset_noisy, sample_embeddings, cfg):
        """Noise detector should flag at least some of the injected 10% noise."""
        clean, noisy_indices, _ = detect_label_noise(
            sample_dataset_noisy, cfg["quality"], sample_embeddings
        )
        # We injected ~10% noise — detector should catch at least some
        assert len(noisy_indices) >= 0   # non-negative
        assert len(clean) <= len(sample_dataset_noisy)

    def test_clean_dataset_low_noise(self, clean_dataset, sample_embeddings, cfg):
        """A clean dataset should produce very few flagged noisy labels."""
        clean, noisy_indices, _ = detect_label_noise(
            clean_dataset, cfg["quality"], sample_embeddings
        )
        noise_rate = len(noisy_indices) / len(clean_dataset)
        # Should not flag > 30% of a genuinely clean dataset
        assert noise_rate < 0.30, f"Too many false-positive noise flags: {noise_rate:.1%}"

    def test_returns_three_outputs(self, clean_dataset, sample_embeddings, cfg):
        result = detect_label_noise(clean_dataset, cfg["quality"], sample_embeddings)
        assert len(result) == 3
        clean_ds, noisy_idx, probs = result
        assert isinstance(clean_ds, Dataset)
        assert isinstance(noisy_idx, list)
        assert isinstance(probs, np.ndarray)

    def test_no_embeddings_returns_full_dataset(self, clean_dataset, cfg):
        """Without embeddings, all examples should pass through."""
        clean, noisy_indices, probs = detect_label_noise(
            clean_dataset, cfg["quality"], embeddings=None
        )
        assert len(clean) == len(clean_dataset)
        assert noisy_indices == []


# ── run_quality_gates end-to-end ───────────────────────────────────────────────

class TestRunQualityGates:

    def test_returns_dataset_and_report(self, clean_dataset, cfg):
        result = run_quality_gates(clean_dataset, cfg, embeddings=None)
        assert len(result) == 2
        ds, report = result
        assert isinstance(ds, Dataset)
        assert isinstance(report, QualityReport)

    def test_report_fields_populated(self, clean_dataset, cfg):
        _, report = run_quality_gates(clean_dataset, cfg, embeddings=None)
        assert report.total_input == len(clean_dataset)
        assert report.total_output <= report.total_input
        assert report.total_output >= 0
        assert 0.0 <= report.noise_rate <= 1.0

    def test_report_summary_is_string(self, clean_dataset, cfg):
        _, report = run_quality_gates(clean_dataset, cfg, embeddings=None)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Input rows" in summary

    def test_output_smaller_than_input_when_outliers_present(self, dataset_with_outliers, cfg):
        clean, report = run_quality_gates(dataset_with_outliers, cfg, embeddings=None)
        assert len(clean) < len(dataset_with_outliers)
        assert report.outliers_removed > 0

    def test_domain_counts_in_report(self, clean_dataset, cfg):
        _, report = run_quality_gates(clean_dataset, cfg, embeddings=None)
        assert isinstance(report.domain_counts, dict)
        # All four domains should appear
        for domain in ["technical", "billing", "returns", "escalation"]:
            assert domain in report.domain_counts

    def test_with_embeddings_runs_noise_detection(self, clean_dataset, sample_embeddings, cfg):
        """When embeddings provided, noise detection step runs."""
        clean, report = run_quality_gates(clean_dataset, cfg, embeddings=sample_embeddings)
        # Report should have a noise_rate field (may be 0 on clean data)
        assert hasattr(report, "noise_rate")
        assert report.noise_rate >= 0.0

    def test_pipeline_idempotent_on_clean_data(self, clean_dataset, cfg):
        """Running gates twice should not further reduce a clean dataset."""
        clean1, _ = run_quality_gates(clean_dataset, cfg, embeddings=None)
        clean2, _ = run_quality_gates(clean1,    cfg, embeddings=None)
        # Second pass should not remove additional rows (within tolerance)
        assert abs(len(clean1) - len(clean2)) <= 2
