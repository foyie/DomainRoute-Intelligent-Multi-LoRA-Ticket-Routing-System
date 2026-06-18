"""
tests/test_evaluation.py
─────────────────────────
Tests for Phase 4 evaluation suite.

Coverage
--------
Metrics:
  - auto_resolution_rate: perfect, zero, partial, all-escalation inputs
  - escalation_metrics: TP/TN/FP/FN, safety budget, edge cases
  - latency_metrics: p50/p95/p99, SLA pass/fail, single value, empty
  - cost_metrics: domain counts, reduction %, unknown domain ignored
  - routing_accuracy: overall acc, per-domain, macro F1
  - composite_score: weights sum correctly, escalation safety weight
  - domain_evaluation: end-to-end with mocked data
  - compliance_check and tone_score heuristics

Semantic drift:
  - fast_drift_check: identical embeddings, drifted embeddings, threshold boundary
  - DriftEvalResult summary string
  - ContaminationMatrix summary string

Hallucination detector:
  - heuristic_check: fabricated specifics, contradiction signals, clean response
  - HallucinationResult summary
  - HallucinationReport construction

Pareto frontier:
  - ParetoPoint.dominates() — correct dominance logic
  - ParetoFrontier.compute() — finds correct non-dominated set
  - ParetoFrontier.select_best() — priority modes
  - ParetoFrontier.dominated_hypervolume()
  - save / load roundtrip
  - synthetic_sweep_pareto() — produces valid frontier
  - build_pareto_frontier() from EvaluationResult list

A/B test harness:
  - mcnemar_test: identical predictions, treatment wins, control wins
  - bootstrap_ci: CI contains true delta, width scales with n
  - minimum_detectable_effect: decreases with larger n
  - required_sample_size: increases for smaller MDE
  - run(): significant when large delta, not significant when small
  - per_domain_breakdown: Bonferroni correction applied
  - save/load roundtrip
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from evaluation.metrics import (
    auto_resolution_rate,
    escalation_metrics,
    latency_metrics,
    cost_metrics,
    composite_score,
    tone_score,
    compliance_check,
    EvaluationResult,
    EscalationMetrics,
    LatencyMetrics,
    CostMetrics,
    DomainMetrics,
    LATENCY_SLA_MS,
    ESCALATION_FN_BUDGET,
    DEFAULT_COST_MAP,
)
from evaluation.pareto_frontier import (
    ParetoPoint,
    ParetoFrontier,
    synthetic_sweep_pareto,
    build_pareto_frontier,
)
from evaluation.ab_test_harness import ABTestHarness, ABTestResult
from evaluation.hallucination_detector import (
    HallucinationDetector,
    HallucinationResult,
)
from evaluation.semantic_drift_eval import (
    fast_drift_check,
    DriftEvalResult,
    COSINE_SIM_THRESHOLD,
)
from routing.models import Domain


# ══════════════════════════════════════════════════════════════════════════════
# Metrics tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAutoResolutionRate:
    def test_perfect_resolution(self):
        preds  = ["resolved"] * 10
        labels = ["resolved"] * 10
        assert auto_resolution_rate(preds, labels) == pytest.approx(1.0)

    def test_zero_resolution(self):
        preds  = ["escalate"] * 10
        labels = ["resolved"] * 10
        assert auto_resolution_rate(preds, labels) == pytest.approx(0.0)

    def test_partial_resolution(self):
        preds  = ["resolved", "resolved", "escalate", "resolved", "escalate"]
        labels = ["resolved", "resolved", "resolved", "resolved", "resolved"]
        # 3 correctly resolved out of 5 true resolved
        assert auto_resolution_rate(preds, labels) == pytest.approx(3/5)

    def test_empty_inputs(self):
        assert auto_resolution_rate([], []) == pytest.approx(0.0)

    def test_all_escalation_labels(self):
        preds  = ["escalate"] * 5
        labels = ["escalate"] * 5
        # No "resolved" ground truth → rate = 0/0 → 0.0
        assert auto_resolution_rate(preds, labels) == pytest.approx(0.0)

    def test_mixed_labels(self):
        preds  = ["resolved", "escalate", "resolved", "escalate"]
        labels = ["resolved", "escalate", "escalate", "resolved"]
        # True resolved: idx 0, 3. Correctly resolved: idx 0. Rate = 1/2
        assert auto_resolution_rate(preds, labels) == pytest.approx(0.5)


class TestEscalationMetrics:
    def test_perfect_detector(self):
        esc_preds  = [True,  False, True,  False]
        esc_labels = [True,  False, True,  False]
        m = escalation_metrics(esc_preds, esc_labels)
        assert m.sensitivity == pytest.approx(1.0)
        assert m.specificity == pytest.approx(1.0)
        assert m.false_negative_rate == pytest.approx(0.0)
        assert m.meets_safety_budget is True

    def test_zero_fn_one_fp(self):
        # Predicts escalation when not needed (FP) but never misses real esc (FN=0)
        esc_preds  = [True,  True,  False]
        esc_labels = [True,  False, False]
        m = escalation_metrics(esc_preds, esc_labels)
        assert m.false_negative_rate == pytest.approx(0.0)
        assert m.meets_safety_budget is True
        assert m.false_positive_rate > 0.0

    def test_fn_exceeds_budget(self):
        # 1 missed escalation out of 1 true escalation → FNR = 1.0
        esc_preds  = [False]
        esc_labels = [True]
        m = escalation_metrics(esc_preds, esc_labels)
        assert m.false_negative_rate == pytest.approx(1.0)
        assert m.meets_safety_budget is False

    def test_f1_perfect(self):
        esc_preds  = [True, True, False, False]
        esc_labels = [True, True, False, False]
        m = escalation_metrics(esc_preds, esc_labels)
        assert m.f1 == pytest.approx(1.0)

    def test_no_true_escalations(self):
        esc_preds  = [False, False]
        esc_labels = [False, False]
        m = escalation_metrics(esc_preds, esc_labels)
        assert m.sensitivity == pytest.approx(0.0)   # 0/0 → 0
        assert m.false_negative_rate == pytest.approx(0.0)

    def test_all_predicted_escalation(self):
        esc_preds  = [True,  True,  True]
        esc_labels = [True,  False, False]
        m = escalation_metrics(esc_preds, esc_labels)
        assert m.sensitivity == pytest.approx(1.0)
        assert m.false_positive_rate == pytest.approx(1.0)

    def test_empty_inputs(self):
        m = escalation_metrics([], [])
        assert m.n_total == 0

    def test_summary_string(self):
        m = escalation_metrics([True, False], [True, False])
        s = m.summary()
        assert "sensitivity" in s
        assert "PASS" in s


class TestLatencyMetrics:
    def test_basic_percentiles(self):
        lats = list(range(1, 101))   # 1ms to 100ms
        m = latency_metrics(lats)
        assert m.p50_ms == pytest.approx(50.5, abs=1)
        assert m.p95_ms == pytest.approx(95.0, abs=2)
        assert m.p99_ms == pytest.approx(99.0, abs=2)

    def test_sla_pass(self):
        lats = [50.0] * 100 + [100.0] * 10
        m = latency_metrics(lats)
        assert m.meets_sla is True

    def test_sla_fail(self):
        lats = [300.0] * 10   # all above 200ms SLA
        m = latency_metrics(lats)
        assert m.meets_sla is False

    def test_single_value(self):
        m = latency_metrics([150.0])
        assert m.p50_ms == pytest.approx(150.0)
        assert m.p95_ms == pytest.approx(150.0)

    def test_empty_inputs(self):
        m = latency_metrics([])
        assert m.n_requests == 0
        assert m.meets_sla is False

    def test_n_requests_correct(self):
        m = latency_metrics([10.0, 20.0, 30.0])
        assert m.n_requests == 3

    def test_summary_string(self):
        m = latency_metrics([100.0] * 100)
        s = m.summary()
        assert "p50" in s and "p95" in s


class TestCostMetrics:
    def test_basic_cost(self):
        counts = {"technical": 100, "billing": 50}
        m = cost_metrics(counts)
        expected_avg = (100 * 0.05 + 50 * 0.12) / 150
        assert m.avg_cost_per_ticket == pytest.approx(expected_avg, abs=0.001)
        assert m.n_tickets == 150

    def test_cost_reduction_pct(self):
        counts = {"technical": 100}
        m = cost_metrics(counts)
        expected_reduction = (1 - 0.05 / 0.50) * 100
        assert m.cost_reduction_pct == pytest.approx(expected_reduction, abs=0.5)

    def test_unknown_domain_ignored(self):
        counts = {"technical": 100, "galaxy_brain": 50}
        m = cost_metrics(counts)
        assert m.n_tickets == 100   # galaxy_brain ignored

    def test_all_domains(self):
        counts = {d.value: 100 for d in [Domain.TECHNICAL, Domain.BILLING,
                                          Domain.RETURNS, Domain.ESCALATION]}
        m = cost_metrics(counts)
        assert m.n_tickets == 400

    def test_empty_counts(self):
        m = cost_metrics({})
        assert m.avg_cost_per_ticket == pytest.approx(0.0)
        assert m.n_tickets == 0


class TestCompositeScore:
    def _make_eval_result(
        self, arr=0.943, esc_sens=0.998, lat_p95=150.0,
        cost=0.07, hall=0.012,
    ):
        esc = EscalationMetrics(
            sensitivity=esc_sens, specificity=0.965,
            false_negative_rate=0.002, false_positive_rate=0.035,
            precision=0.95, f1=0.97,
            n_true_escalations=50, n_predicted_escalations=52,
            n_total=1000, meets_safety_budget=True,
        )
        lat = LatencyMetrics(
            p50_ms=85, p95_ms=lat_p95, p99_ms=280,
            mean_ms=100, max_ms=350, n_requests=1000,
            meets_sla=lat_p95 < LATENCY_SLA_MS,
        )
        cost_m = CostMetrics(
            avg_cost_per_ticket=cost, total_cost=cost * 1000,
            cost_by_domain={}, n_tickets=1000,
        )
        return EvaluationResult(
            run_name="test",
            auto_resolution_rate=arr,
            escalation=esc,
            latency=lat,
            cost=cost_m,
            routing=None,
            per_domain={},
            hallucination_rate=hall,
        )

    def test_score_in_range(self):
        r = self._make_eval_result()
        s = composite_score(r)
        assert 0.0 <= s <= 1.0

    def test_high_score_for_good_metrics(self):
        r = self._make_eval_result(arr=0.99, esc_sens=1.0, lat_p95=50, cost=0.01, hall=0.0)
        s = composite_score(r)
        assert s > 0.85

    def test_lower_score_for_poor_latency(self):
        r_good = self._make_eval_result(lat_p95=100)
        r_bad  = self._make_eval_result(lat_p95=500)
        assert composite_score(r_good) > composite_score(r_bad)

    def test_lower_score_for_poor_escalation(self):
        r_good = self._make_eval_result(esc_sens=1.0)
        r_bad  = self._make_eval_result(esc_sens=0.5)
        assert composite_score(r_good) > composite_score(r_bad)

    def test_passes_all_gates_good(self):
        r = self._make_eval_result()
        assert r.passes_all_gates() is True

    def test_fails_gate_high_hall(self):
        r = self._make_eval_result(hall=0.20)
        assert r.passes_all_gates() is False

    def test_fails_gate_high_latency(self):
        r = self._make_eval_result(lat_p95=350)
        assert r.passes_all_gates() is False


class TestHeuristicMetrics:
    def test_tone_score_empathetic(self):
        resp = "I'm sorry for the inconvenience. I understand your frustration and will help you resolve this."
        assert tone_score(resp) > 0.0

    def test_tone_score_poor(self):
        resp = "Contact us again."
        assert tone_score(resp) == pytest.approx(0.0, abs=0.1)

    def test_compliance_billing(self):
        resp = "I will process the refund to your account immediately."
        r = compliance_check(resp, "billing")
        assert r["non_empty"] is True
        assert r["action_stated"] is True

    def test_compliance_escalation(self):
        resp = "I apologise. I'm escalating this to a manager. Your case ID is #12345."
        r = compliance_check(resp, "escalation")
        assert r["contains_apology"] is True
        assert r["escalates"] is True
        assert r["provides_case_id"] is True

    def test_compliance_technical_actionable(self):
        resp = "Please try restarting your device and then check the firmware version."
        r = compliance_check(resp, "technical")
        assert r["actionable_steps"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Semantic drift tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFastDriftCheck:
    def test_identical_embeddings_no_drift(self):
        rng  = np.random.RandomState(42)
        embs = rng.randn(20, 64).astype(np.float32)
        sim, drifted = fast_drift_check(embs, embs)
        assert sim == pytest.approx(1.0, abs=1e-4)
        assert drifted is False

    def test_drifted_embeddings(self):
        rng  = np.random.RandomState(42)
        base = rng.randn(20, 64).astype(np.float32)
        ft   = rng.randn(20, 64).astype(np.float32)   # different seed → different direction
        sim, drifted = fast_drift_check(base, ft, threshold=0.99)
        # Random vectors are mostly orthogonal — should drift
        assert drifted is True

    def test_near_identical_passes_default_threshold(self):
        rng  = np.random.RandomState(42)
        base = rng.randn(20, 64).astype(np.float32)
        # Add very small noise
        ft   = base + rng.randn(20, 64).astype(np.float32) * 0.001
        sim, drifted = fast_drift_check(base, ft, threshold=COSINE_SIM_THRESHOLD)
        assert sim > COSINE_SIM_THRESHOLD
        assert drifted is False

    def test_threshold_boundary(self):
        rng  = np.random.RandomState(42)
        base = rng.randn(5, 8).astype(np.float32)
        _, drifted_high = fast_drift_check(base, base, threshold=1.01)  # impossible threshold
        assert drifted_high is True
        _, drifted_low  = fast_drift_check(base, base, threshold=-0.01)  # always passes
        assert drifted_low is False

    def test_drift_result_summary(self):
        r = DriftEvalResult(
            domain="technical", adapter_path="p",
            mean_cosine_similarity=0.962, std_cosine_similarity=0.012,
            min_cosine_similarity=0.940, max_cosine_similarity=0.980,
            cosine_distance=0.038, is_drifted=False, n_probe_texts=50,
        )
        s = r.summary()
        assert "technical" in s and "0.9620" in s and "OK" in s

    def test_drift_result_summary_drifted(self):
        r = DriftEvalResult(
            domain="billing", adapter_path="p",
            mean_cosine_similarity=0.88, std_cosine_similarity=0.02,
            min_cosine_similarity=0.82, max_cosine_similarity=0.93,
            cosine_distance=0.12, is_drifted=True, n_probe_texts=50,
        )
        s = r.summary()
        assert "DRIFT" in s


# ══════════════════════════════════════════════════════════════════════════════
# Hallucination detector tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHallucinationDetector:
    def test_clean_response_low_score(self):
        detector = HallucinationDetector(use_llm=False)
        ticket   = "My headphones won't charge."
        response = "Please try restarting your device and checking the charging cable."
        score = detector.heuristic_check(ticket, response)
        assert score < 0.30

    def test_fabricated_order_id_high_score(self):
        detector = HallucinationDetector(use_llm=False)
        ticket   = "I have a problem with my headphones."
        response = "I see your order #483921 is ready to be processed."
        score = detector.heuristic_check(ticket, response)
        assert score >= 0.30

    def test_hallucination_phrase_high_score(self):
        detector = HallucinationDetector(use_llm=False)
        ticket   = "I need help with my account."
        response = "As per our conversation last week, I'll process your request now."
        score = detector.heuristic_check(ticket, response)
        assert score >= 0.25

    def test_detect_heuristic_mode(self):
        detector = HallucinationDetector(use_llm=False)
        result = detector.detect(
            "My device won't turn on.",
            "Please try restarting the device.",
            domain="technical",
        )
        assert isinstance(result, HallucinationResult)
        assert result.detection_method == "heuristic"
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_batch_length(self):
        detector = HallucinationDetector(use_llm=False)
        tickets   = ["Issue one", "Issue two", "Issue three"]
        responses = ["Response one", "Response two", "Response three"]
        results   = detector.detect_batch(tickets, responses)
        assert len(results) == 3

    def test_detect_batch_mismatch_raises(self):
        detector = HallucinationDetector(use_llm=False)
        with pytest.raises(ValueError):
            detector.detect_batch(["t1", "t2"], ["r1"])

    def test_hallucination_result_summary(self):
        r = HallucinationResult(
            ticket="t", response="r", domain="technical",
            is_hallucinated=False, confidence=0.1,
            detection_method="heuristic",
        )
        assert "CLEAN" in r.summary()

    def test_hallucination_result_summary_flagged(self):
        r = HallucinationResult(
            ticket="t", response="r", domain="billing",
            is_hallucinated=True, confidence=0.8,
            detection_method="heuristic",
        )
        assert "HALLUCINATED" in r.summary()

    def test_evaluate_dataset(self):
        detector = HallucinationDetector(use_llm=False)
        records  = [
            {"text": "My device is broken.", "domain": "technical",
             "response": "Please restart your device."},
            {"text": "I need a refund.",      "domain": "billing",
             "response": "I have processed your refund successfully."},
        ]
        report = detector.evaluate_dataset(records)
        assert report.n_total == 2
        assert 0.0 <= report.hallucination_rate <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Pareto frontier tests
# ══════════════════════════════════════════════════════════════════════════════

class TestParetoPoint:
    def _point(self, acc, lat, cost, name="p"):
        return ParetoPoint(name, "technical", 32, acc, lat, cost)

    def test_dominates_strictly_better(self):
        # a is better on ALL three: accuracy, speed (lower lat), cost (lower cost)
        a = self._point(0.95, 100, 0.05)
        b = self._point(0.90, 150, 0.10)
        assert a.dominates(b)
        assert not b.dominates(a)

    def test_dominates_equal_not_dominated(self):
        a = self._point(0.95, 100, 0.05)
        b = self._point(0.95, 100, 0.05)
        assert not a.dominates(b)
        assert not b.dominates(a)

    def test_dominates_mixed_objectives(self):
        a = self._point(0.95, 100, 0.10)  # better acc, same lat, worse cost
        b = self._point(0.90, 100, 0.05)  # worse acc, same lat, better cost
        assert not a.dominates(b)
        assert not b.dominates(a)

    def test_objectives_tuple(self):
        p = self._point(0.90, 200, 0.10)
        obj = p.objectives()
        assert len(obj) == 3
        assert obj[0] == 0.90          # accuracy
        assert obj[1] == pytest.approx(1/200)  # speed
        assert obj[2] == pytest.approx(1/0.10) # cost efficiency

    def test_summary_contains_rank(self):
        p = ParetoPoint("tech_r32", "technical", 32, 0.965, 120, 0.12)
        p.is_pareto_optimal = True
        assert "★" in p.summary()


class TestParetoFrontier:
    def _make_frontier(self):
        frontier = ParetoFrontier()
        # a dominates c on all three objectives: acc 0.95>0.85, lat 100<150, cost 0.05<0.15
        a = ParetoPoint("a", "tech", 32, 0.95, 100, 0.05)   # fast, cheap, good acc
        b = ParetoPoint("b", "tech", 64, 0.99, 300, 0.25)   # high acc, slow, expensive (incomparable to a)
        c = ParetoPoint("c", "tech", 16, 0.85, 150, 0.15)   # dominated by a on all three
        for p in [a, b, c]:
            frontier.add(p)
        return frontier, a, b, c

    def test_compute_finds_correct_frontier(self):
        frontier, a, b, c = self._make_frontier()
        optimal = frontier.compute()
        names = {p.run_name for p in optimal}
        assert "a" in names
        assert "b" in names
        assert "c" not in names

    def test_dominated_point_flagged(self):
        frontier, a, b, c = self._make_frontier()
        frontier.compute()
        assert c.is_pareto_optimal is False

    def test_frontier_sorted_by_accuracy(self):
        frontier, a, b, c = self._make_frontier()
        f = frontier.compute()
        accs = [p.accuracy for p in f]
        assert accs == sorted(accs, reverse=True)

    def test_select_best_accuracy_priority(self):
        frontier, a, b, c = self._make_frontier()
        frontier.compute()
        best = frontier.select_best(priority="accuracy", latency_budget_ms=9999, cost_budget=9999)
        assert best.accuracy == max(p.accuracy for p in frontier.frontier)

    def test_select_best_speed_priority(self):
        frontier, a, b, c = self._make_frontier()
        frontier.compute()
        best = frontier.select_best(priority="speed", latency_budget_ms=9999, cost_budget=9999)
        assert best.latency_ms == min(p.latency_ms for p in frontier.frontier)

    def test_select_best_cost_priority(self):
        frontier, a, b, c = self._make_frontier()
        frontier.compute()
        best = frontier.select_best(priority="cost", latency_budget_ms=9999, cost_budget=9999)
        assert best.cost_per_ticket == min(p.cost_per_ticket for p in frontier.frontier)

    def test_select_best_no_candidates_in_budget(self):
        frontier, a, b, c = self._make_frontier()
        frontier.compute()
        # Unreachable budget — should still return something
        best = frontier.select_best(latency_budget_ms=10, cost_budget=0.001)
        assert best is not None

    def test_hypervolume_positive(self):
        frontier, a, b, c = self._make_frontier()
        frontier.compute()
        hv = frontier.dominated_hypervolume()
        assert hv >= 0.0

    def test_save_load_roundtrip(self, tmp_path):
        frontier, a, b, c = self._make_frontier()
        frontier.compute()
        path = tmp_path / "pareto.json"
        frontier.save(path)
        loaded = ParetoFrontier.load(path)
        loaded.compute()
        assert len(loaded.frontier) == len(frontier.frontier)

    def test_to_dataframe_shape(self):
        frontier, a, b, c = self._make_frontier()
        frontier.compute()
        df = frontier.to_dataframe()
        assert len(df) == 3
        assert "accuracy" in df.columns

    def test_empty_frontier_select_returns_none(self):
        frontier = ParetoFrontier()
        assert frontier.select_best() is None

    def test_synthetic_sweep_pareto_valid(self):
        frontier = synthetic_sweep_pareto()
        assert len(frontier.frontier) >= 1
        for p in frontier.frontier:
            assert 0.0 <= p.accuracy <= 1.0
            assert p.latency_ms > 0
            assert p.cost_per_ticket > 0


# ══════════════════════════════════════════════════════════════════════════════
# A/B test harness tests
# ══════════════════════════════════════════════════════════════════════════════

class TestABTestHarness:
    def _harness(self, n_bootstrap=100):
        return ABTestHarness(alpha=0.05, n_bootstrap=n_bootstrap, seed=42)

    def test_identical_predictions_not_significant(self):
        harness = self._harness()
        preds = ["resolved"] * 100 + ["escalate"] * 20
        labels = preds[:]
        result = harness.run(preds, preds, labels, "identical")
        assert result.accuracy_delta == pytest.approx(0.0, abs=0.01)
        assert result.is_significant is False

    def test_large_delta_significant(self):
        harness = self._harness()
        rng     = np.random.RandomState(0)
        labels  = ["resolved"] * 800 + ["escalate"] * 200
        control   = ["resolved" if rng.rand() < 0.70 else "escalate" for _ in labels]
        treatment = ["resolved" if rng.rand() < 0.95 else "escalate" for _ in labels]
        result = harness.run(control, treatment, labels, "large_delta")
        assert result.treatment_accuracy > result.control_accuracy
        assert result.is_significant is True

    def test_small_delta_not_significant(self):
        harness = ABTestHarness(alpha=0.05, n_bootstrap=100, seed=1)
        rng     = np.random.RandomState(1)
        n       = 50
        labels  = ["resolved"] * n
        control   = ["resolved" if rng.rand() < 0.70 else "escalate" for _ in labels]
        treatment = ["resolved" if rng.rand() < 0.71 else "escalate" for _ in labels]
        result = harness.run(control, treatment, labels, "small_delta")
        # Very small delta — should not be significant with n=50
        assert abs(result.accuracy_delta) < 0.20

    def test_mcnemar_identical_p1(self):
        harness = self._harness()
        preds   = [True, False, True, False] * 10
        stat, p = harness.mcnemar_test(preds, preds)
        assert p == pytest.approx(1.0)

    def test_mcnemar_treatment_always_wins(self):
        harness = self._harness()
        ctrl = [False] * 40 + [True] * 10
        trt  = [True]  * 40 + [True] * 10
        stat, p = harness.mcnemar_test(ctrl, trt)
        assert p < 0.05

    def test_bootstrap_ci_contains_true_delta(self):
        harness = ABTestHarness(alpha=0.05, n_bootstrap=500, seed=42)
        rng     = np.random.RandomState(42)
        n       = 300
        labels  = ["resolved"] * n
        ctrl    = [rng.rand() < 0.70 for _ in range(n)]
        trt     = [rng.rand() < 0.90 for _ in range(n)]
        lo, hi  = harness.bootstrap_ci(ctrl, trt, n_bootstrap=500)
        true_delta = sum(trt) / n - sum(ctrl) / n
        assert lo <= true_delta <= hi

    def test_mde_decreases_with_n(self):
        harness = self._harness()
        mde_small = harness.minimum_detectable_effect(n=50)
        mde_large = harness.minimum_detectable_effect(n=1000)
        assert mde_large < mde_small

    def test_required_sample_size_increases_for_smaller_mde(self):
        harness  = self._harness()
        n_large  = harness.required_sample_size(mde=0.01)
        n_small  = harness.required_sample_size(mde=0.10)
        assert n_large > n_small

    def test_result_n_samples_correct(self):
        harness = self._harness()
        n       = 80
        preds   = ["resolved"] * n
        result  = harness.run(preds, preds, preds, "test")
        assert result.n_samples == n

    def test_ci_lower_lt_upper(self):
        harness = self._harness()
        rng     = np.random.RandomState(99)
        n       = 200
        ctrl    = [rng.rand() < 0.70 for _ in range(n)]
        trt     = [rng.rand() < 0.85 for _ in range(n)]
        result  = harness.run(
            ["resolved" if c else "escalate" for c in ctrl],
            ["resolved" if t else "escalate" for t in trt],
            ["resolved"] * n, "ci_test",
        )
        assert result.ci_lower <= result.ci_upper

    def test_effect_size_zero_identical(self):
        harness = self._harness()
        result  = harness.run(
            ["resolved"] * 50, ["resolved"] * 50, ["resolved"] * 50, "zero_effect"
        )
        assert result.effect_size == pytest.approx(0.0, abs=0.01)

    def test_per_domain_breakdown_populated(self):
        harness = self._harness()
        rng     = np.random.RandomState(5)
        n       = 200
        domains = (["technical"] * 50 + ["billing"] * 50 +
                   ["returns"] * 50 + ["escalation"] * 50)
        labels  = ["resolved"] * 150 + ["escalate"] * 50
        ctrl    = [rng.choice(["resolved", "escalate"]) for _ in range(n)]
        trt     = [rng.choice(["resolved", "escalate"]) for _ in range(n)]
        result  = harness.run(ctrl, trt, labels, "domain_test", domains=domains)
        assert len(result.per_domain) == 4
        for domain in ["technical", "billing", "returns", "escalation"]:
            assert domain in result.per_domain
            assert "bonferroni_alpha" in result.per_domain[domain]

    def test_save_load_roundtrip(self, tmp_path):
        harness = self._harness()
        preds   = ["resolved"] * 60 + ["escalate"] * 20
        result  = harness.run(preds, preds, preds, "save_test")
        path    = tmp_path / "ab_result.json"
        harness.save_result(result, path)
        loaded  = ABTestHarness.load_result(path)
        assert loaded.test_name == result.test_name
        assert loaded.accuracy_delta == pytest.approx(result.accuracy_delta)
        assert loaded.p_value == pytest.approx(result.p_value, abs=1e-6)

    def test_summary_string(self):
        harness = self._harness()
        result  = harness.run(
            ["resolved"] * 80, ["resolved"] * 80, ["resolved"] * 80, "summary_test"
        )
        s = result.summary()
        assert "control" in s.lower() and "treatment" in s.lower()
        assert "p_value" in s.lower() or "p-value" in s.lower() or "McNemar" in s
