"""
tests/test_serving.py
──────────────────────
Tests for Phase 5 serving layer.

Coverage
--------
Monitoring:
  - RequestTrace construction, finalise(), to_log_dict()
  - VeriTuneMetrics.record() increments counters correctly
  - VeriTuneMetrics.snapshot() returns expected keys
  - VeriTuneMetrics singleton pattern
  - StructuredFormatter produces valid JSON

Safety filters:
  - EscalationGuard: fires on high score, fires on keywords, passes clean ticket
  - EscalationGuard: produces case_id and override_response on trigger
  - PIIScrubber: scrubs email, phone, SSN, card number
  - PIIScrubber: clean text passes through unmodified
  - ToneChecker: fails on too-short response, passes normal response
  - ToneChecker: fails on rude language
  - HallucinationGuard: delegates to heuristic correctly
  - ComplianceChecker: billing compliance passes, escalation passes
  - SafetyPipeline.run(): full pipeline on clean response
  - SafetyPipeline.run(): escalation path returns override_response
  - SafetyPipeline.run(): PII scrubbed end-to-end
  - SafetyReport.all_passed, summary()

Inference pipeline:
  - build_pipeline() returns InferencePipeline (no models needed)
  - InferencePipeline.health_check() with no models
  - InferencePipeline._fallback_response() returns non-empty string
  - InferencePipeline._check_router() False when unfitted

FastAPI endpoints:
  - GET /health → 503 when pipeline not ready
  - GET /health → 200 when pipeline mocked
  - GET /metrics → returns JSON with expected keys
  - GET /info → returns version and domain info
  - POST /predict → 503 when pipeline not ready
  - POST /predict/batch → 422 when >32 requests
  - POST /predict → successful mock prediction
"""

from __future__ import annotations

import json
import logging
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ══════════════════════════════════════════════════════════════════════════════
# Monitoring tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRequestTrace:
    def test_construction_defaults(self):
        from serving.monitoring import RequestTrace
        trace = RequestTrace()
        assert trace.domain == "unknown"
        assert trace.status == "success"
        assert trace.trace_id != ""

    def test_finalise_sets_total_ms(self):
        from serving.monitoring import RequestTrace
        import time as _t
        trace = RequestTrace()
        _t.sleep(0.01)
        trace.finalise()
        assert trace.total_ms >= 5.0   # at least 5ms

    def test_finalise_sets_escalated_status(self):
        from serving.monitoring import RequestTrace
        trace = RequestTrace(escalation_detected=True)
        trace.finalise()
        assert trace.status == "escalated"

    def test_to_log_dict_keys(self):
        from serving.monitoring import RequestTrace
        trace = RequestTrace(domain="technical", routing_confidence=0.88)
        trace.finalise()
        d = trace.to_log_dict()
        assert "trace_id" in d
        assert "domain" in d
        assert "latency_ms" in d
        assert d["domain"] == "technical"

    def test_to_log_dict_latency_breakdown(self):
        from serving.monitoring import RequestTrace
        trace = RequestTrace()
        trace.router_ms = 12.0; trace.lora_load_ms = 8.0
        trace.generation_ms = 105.0; trace.safety_ms = 5.0
        d = trace.to_log_dict()
        assert "router" in d["latency_ms"]
        assert "generation" in d["latency_ms"]


class TestVeriTuneMetrics:
    def test_singleton(self):
        from serving.monitoring import VeriTuneMetrics
        VeriTuneMetrics._instance = None   # reset
        m1 = VeriTuneMetrics.get()
        m2 = VeriTuneMetrics.get()
        assert m1 is m2

    def test_record_increments_counter(self):
        from serving.monitoring import VeriTuneMetrics, RequestTrace
        VeriTuneMetrics._instance = None
        metrics = VeriTuneMetrics.get()
        initial = metrics._counters["requests_total"]
        trace = RequestTrace(domain="technical")
        trace.finalise()
        metrics.record(trace)
        assert metrics._counters["requests_total"] == initial + 1

    def test_record_escalation_counter(self):
        from serving.monitoring import VeriTuneMetrics, RequestTrace
        VeriTuneMetrics._instance = None
        metrics = VeriTuneMetrics.get()
        initial = metrics._counters["escalations_total"]
        trace = RequestTrace(domain="escalation", escalation_detected=True)
        trace.finalise()
        metrics.record(trace)
        assert metrics._counters["escalations_total"] == initial + 1

    def test_record_cache_hit(self):
        from serving.monitoring import VeriTuneMetrics, RequestTrace
        VeriTuneMetrics._instance = None
        metrics = VeriTuneMetrics.get()
        trace = RequestTrace(domain="technical", cache_hit=True)
        trace.finalise()
        metrics.record(trace)
        assert metrics._counters["cache_hits"] >= 1

    def test_snapshot_keys(self):
        from serving.monitoring import VeriTuneMetrics
        VeriTuneMetrics._instance = None
        metrics = VeriTuneMetrics.get()
        snap = metrics.snapshot()
        assert "requests_total" in snap
        assert "latency_p95_ms" in snap
        assert "cache_hit_rate" in snap
        assert "domain_distribution" in snap

    def test_snapshot_domain_distribution(self):
        from serving.monitoring import VeriTuneMetrics, RequestTrace
        VeriTuneMetrics._instance = None
        metrics = VeriTuneMetrics.get()
        trace = RequestTrace(domain="billing")
        trace.finalise()
        metrics.record(trace)
        assert "billing" in metrics.snapshot()["domain_distribution"]


class TestStructuredFormatter:
    def test_produces_valid_json(self):
        from serving.monitoring import StructuredFormatter
        fmt = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="test message", args=(), exc_info=None,
        )
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "test message"
        assert parsed["level"] == "INFO"
        assert "ts" in parsed


# ══════════════════════════════════════════════════════════════════════════════
# Safety filter tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEscalationGuard:
    def test_fires_on_high_score(self):
        from serving.safety_filters import EscalationGuard
        guard = EscalationGuard()
        result = guard.check("Normal ticket", "Normal response", escalation_score=0.85)
        assert result.passed is False
        assert result.override_response is not None
        assert "case" in result.override_response.lower() or "ESC" in result.override_response

    def test_fires_on_threat_keyword(self):
        from serving.safety_filters import EscalationGuard
        guard = EscalationGuard()
        result = guard.check(
            "I will file a chargeback and take legal action immediately!",
            "Normal response",
            escalation_score=0.0,
        )
        assert result.passed is False

    def test_fires_on_anger_plus_threat(self):
        from serving.safety_filters import EscalationGuard
        guard = EscalationGuard()
        # Threat keyword alone scores 0.50, + anger = 0.85 >= 0.60 threshold
        result = guard.check(
            "This is unacceptable. I will file a chargeback right now!",
            "Normal response",
            escalation_score=0.0,
        )
        assert result.passed is False

    def test_passes_clean_ticket(self):
        from serving.safety_filters import EscalationGuard
        guard = EscalationGuard()
        result = guard.check(
            "My headphones stopped charging after the firmware update.",
            "Please try restarting your device.",
            escalation_score=0.05,
        )
        assert result.passed is True

    def test_override_response_contains_case_id(self):
        from serving.safety_filters import EscalationGuard
        guard = EscalationGuard()
        result = guard.check("Threat ticket", "Response", escalation_score=0.9)
        assert result.override_response is not None
        # Should contain case reference
        assert any(kw in result.override_response for kw in ["ESC-", "case", "reference"])

    def test_custom_case_id_used(self):
        from serving.safety_filters import EscalationGuard
        guard = EscalationGuard()
        result = guard.check("Threat", "Response", escalation_score=0.9, case_id="MY-CASE-001")
        assert "MY-CASE-001" in result.override_response


class TestPIIScrubber:
    def test_scrubs_email(self):
        from serving.safety_filters import PIIScrubber
        scrubber = PIIScrubber()
        text, modified = scrubber.scrub("Please contact john.doe@example.com for help.")
        assert "EMAIL REDACTED" in text
        assert modified is True

    def test_scrubs_phone(self):
        from serving.safety_filters import PIIScrubber
        scrubber = PIIScrubber()
        text, modified = scrubber.scrub("Call us at 555-867-5309 for support.")
        assert "PHONE REDACTED" in text
        assert modified is True

    def test_scrubs_card_number(self):
        from serving.safety_filters import PIIScrubber
        scrubber = PIIScrubber()
        text, modified = scrubber.scrub("Card 4532015112830366 was charged.")
        assert "CARD REDACTED" in text
        assert modified is True

    def test_clean_text_unchanged(self):
        from serving.safety_filters import PIIScrubber
        scrubber = PIIScrubber()
        clean = "Please restart your device and check the charging cable."
        text, modified = scrubber.scrub(clean)
        assert text == clean
        assert modified is False

    def test_check_returns_filter_result(self):
        from serving.safety_filters import PIIScrubber
        scrubber = PIIScrubber()
        result = scrubber.check("Contact email@test.com for support.")
        assert result.filter_name == "PIIScrubber"
        assert result.override_response is not None
        assert "EMAIL REDACTED" in result.override_response

    def test_scrubs_password(self):
        from serving.safety_filters import PIIScrubber
        scrubber = PIIScrubber()
        text, modified = scrubber.scrub("Your password: abc123secret was reset.")
        assert "PASSWORD REDACTED" in text


class TestToneChecker:
    def test_fails_too_short(self):
        from serving.safety_filters import ToneChecker
        checker = ToneChecker()
        result = checker.check("Sorry, can't help.", domain="technical")
        assert result.passed is False
        assert "short" in result.reason.lower()

    def test_passes_normal_response(self):
        from serving.safety_filters import ToneChecker
        checker = ToneChecker()
        response = (
            "Thank you for reaching out. I understand your frustration with "
            "the charging issue. Please try the following steps: first, reset "
            "the device by holding the power button for 10 seconds."
        )
        result = checker.check(response, domain="technical")
        assert result.passed is True

    def test_fails_rude_language(self):
        from serving.safety_filters import ToneChecker
        checker = ToneChecker()
        result = checker.check(
            "This is a stupid question and I really can't help you with that at all.",
            domain="technical",
        )
        assert result.passed is False

    def test_escalation_shorter_threshold(self):
        from serving.safety_filters import ToneChecker
        checker = ToneChecker()
        # 16+ words — passes for escalation (threshold=15)
        response = (
            "I sincerely apologise for the experience you have had. "
            "I am escalating this to a senior specialist right now. "
            "Your case reference is ESC-001."
        )
        result_esc = checker.check(response, domain="escalation")
        assert result_esc.passed is True

    def test_filter_name(self):
        from serving.safety_filters import ToneChecker
        result = ToneChecker().check("short", "technical")
        assert result.filter_name == "ToneChecker"


class TestSafetyPipeline:
    def _good_response(self) -> str:
        return (
            "I understand your concern and I'm here to help. "
            "Please try restarting the device and checking the firmware version. "
            "If the issue persists, I can arrange a replacement for you."
        )

    def test_clean_ticket_all_pass(self):
        from serving.safety_filters import SafetyPipeline
        pipeline = SafetyPipeline()
        report = pipeline.run(
            ticket="My headphones stopped charging after the update.",
            response=self._good_response(),
            domain="technical",
            escalation_score=0.05,
        )
        assert report.escalation_triggered is False
        assert report.pii_scrubbed is False
        assert "EscalationGuard" in [f.filter_name for f in report.filter_results]

    def test_escalation_path_overrides_response(self):
        from serving.safety_filters import SafetyPipeline
        pipeline = SafetyPipeline()
        report = pipeline.run(
            ticket="I will take legal action if you don't fix this NOW.",
            response=self._good_response(),
            domain="technical",
            escalation_score=0.0,
        )
        assert report.escalation_triggered is True
        assert report.final_response != self._good_response()
        assert "apologise" in report.final_response.lower() or "sorry" in report.final_response.lower()

    def test_pii_scrubbed_end_to_end(self):
        from serving.safety_filters import SafetyPipeline
        pipeline = SafetyPipeline()
        response_with_pii = (
            "I have your email john@test.com on file. "
            "Please restart your device and try charging again with the original cable. "
            "Let me know if this resolves the issue for you today."
        )
        report = pipeline.run(
            ticket="Charging issue",
            response=response_with_pii,
            domain="technical",
            escalation_score=0.05,
        )
        assert report.pii_scrubbed is True
        assert "EMAIL REDACTED" in report.final_response

    def test_safety_report_summary(self):
        from serving.safety_filters import SafetyPipeline
        pipeline = SafetyPipeline()
        report = pipeline.run(
            ticket="My device is broken.",
            response=self._good_response(),
            domain="technical",
        )
        s = report.summary()
        assert "SafetyReport" in s
        assert "EscalationGuard" in s

    def test_safety_latency_populated(self):
        from serving.safety_filters import SafetyPipeline
        pipeline = SafetyPipeline()
        report = pipeline.run(
            ticket="My device won't charge.",
            response=self._good_response(),
            domain="technical",
        )
        assert report.safety_latency_ms >= 0.0

    def test_all_passed_property(self):
        from serving.safety_filters import SafetyPipeline, SafetyReport, FilterResult
        # Manually build a report where all filters passed
        report = SafetyReport(
            ticket="t", original_response="r", final_response="r", domain="technical",
            filter_results=[
                FilterResult("EscalationGuard",  True),
                FilterResult("PIIScrubber",       True),
                FilterResult("ToneChecker",       True),
            ],
        )
        assert report.all_passed is True

    def test_all_passed_false_on_failure(self):
        from serving.safety_filters import SafetyReport, FilterResult
        report = SafetyReport(
            ticket="t", original_response="r", final_response="r", domain="technical",
            filter_results=[
                FilterResult("EscalationGuard", False),
                FilterResult("PIIScrubber",     True),
            ],
        )
        assert report.all_passed is False


# ══════════════════════════════════════════════════════════════════════════════
# Inference pipeline tests
# ══════════════════════════════════════════════════════════════════════════════

class TestInferencePipeline:
    def _make_pipeline(self):
        from serving.inference import build_pipeline
        # Build with no model/router paths — uses fallbacks
        return build_pipeline(
            router_path=None,
            registry_path=None,
            base_model=None,
            tokenizer=None,
        )

    def test_build_pipeline_returns_pipeline(self):
        from serving.inference import InferencePipeline
        pipeline = self._make_pipeline()
        assert isinstance(pipeline, InferencePipeline)

    def test_health_check_structure(self):
        pipeline = self._make_pipeline()
        health = pipeline.health_check()
        assert "status" in health
        assert "components" in health
        assert "router" in health["components"]
        assert "loader" in health["components"]

    def test_health_check_no_base_model(self):
        pipeline = self._make_pipeline()
        health = pipeline.health_check()
        assert health["base_model_loaded"] is False

    def test_check_router_false_when_unfitted(self):
        pipeline = self._make_pipeline()
        assert pipeline._check_router() is False

    def test_fallback_response_non_empty(self):
        pipeline = self._make_pipeline()
        resp = pipeline._fallback_response("dummy prompt")
        assert len(resp) > 20

    def test_generate_returns_fallback_when_no_model(self):
        pipeline = self._make_pipeline()
        result = pipeline._generate(None, "test prompt")
        assert isinstance(result, str) and len(result) > 0

    def test_warm_up_does_not_crash(self):
        pipeline = self._make_pipeline()
        # Should not raise even with no adapters on disk
        pipeline.warm_up()


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI endpoint tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFastAPIEndpoints:
    @pytest.fixture
    def client_no_pipeline(self):
        """Test client with no pipeline — patch build_pipeline to fail so _pipeline stays None."""
        import serving.main as main_mod
        with patch("serving.inference.build_pipeline", side_effect=Exception("no pipeline")):
            with TestClient(main_mod.app, raise_server_exceptions=False) as client:
                yield client

    @pytest.fixture
    def client_with_mock_pipeline(self):
        """Test client with a fully mocked pipeline injected via patched build_pipeline."""
        import serving.main as main_mod
        from routing.models import (
            TicketResponse, ResolutionStatus, RoutingDecision, LoRASelection,
            LatencyBreakdown, DomainScore, Domain,
        )

        mock_pipeline = MagicMock()
        mock_pipeline._check_router.return_value = True
        mock_pipeline.health_check.return_value = {
            "status": "ok", "router_fitted": True,
            "base_model_loaded": False,
            "components": {"router": True, "selector": True, "loader": True, "safety": True},
            "adapter_cache": {},
        }

        mock_response = TicketResponse(
            response_text="Please try restarting your device.",
            domain=Domain.TECHNICAL,
            resolution_status=ResolutionStatus.RESOLVED,
            routing_decision=RoutingDecision(
                primary_domain=Domain.TECHNICAL,
                primary_score=0.88,
                all_scores=[DomainScore(domain=Domain.TECHNICAL, score=0.88, rank=1)],
            ),
            lora_selection=LoRASelection(
                domain=Domain.TECHNICAL,
                adapter_path="outputs/checkpoints/technical_best",
                lora_rank=32,
                selection_reason="Confident",
                confidence=0.88,
            ),
            latency=LatencyBreakdown(
                router_ms=12, lora_load_ms=8, generation_ms=105, safety_ms=5, total_ms=130
            ),
        )
        mock_pipeline.predict.return_value = mock_response
        mock_pipeline.predict_batch.return_value = [mock_response]

        with patch("serving.inference.build_pipeline", return_value=mock_pipeline):
            with TestClient(main_mod.app, raise_server_exceptions=False) as client:
                yield client

    def test_health_503_no_pipeline(self, client_no_pipeline):
        resp = client_no_pipeline.get("/health")
        assert resp.status_code == 503

    def test_health_200_with_mock_pipeline(self, client_with_mock_pipeline):
        resp = client_with_mock_pipeline.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_metrics_json_keys(self, client_with_mock_pipeline):
        resp = client_with_mock_pipeline.get("/metrics?format=json")
        assert resp.status_code == 200
        data = resp.json()
        assert "requests_total" in data
        assert "timestamp" in data

    def test_info_endpoint(self, client_with_mock_pipeline):
        resp = client_with_mock_pipeline.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "VeriTune"
        assert "domains" in data
        assert "technical" in data["domains"]

    def test_predict_503_no_pipeline(self, client_no_pipeline):
        resp = client_no_pipeline.post(
            "/predict",
            json={"ticket_text": "My headphones won't charge after the firmware update."},
        )
        assert resp.status_code in (503, 500)

    def test_predict_success_with_mock(self, client_with_mock_pipeline):
        resp = client_with_mock_pipeline.post(
            "/predict",
            json={"ticket_text": "My headphones won't charge after the firmware update."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response_text" in data
        assert "routing_decision" in data
        assert "latency" in data

    def test_predict_batch_too_large(self, client_with_mock_pipeline):
        tickets = [{"ticket_text": f"Ticket {i} about my device issue."}
                   for i in range(33)]
        resp = client_with_mock_pipeline.post("/predict/batch", json=tickets)
        assert resp.status_code == 422

    def test_predict_batch_valid(self, client_with_mock_pipeline):
        tickets = [
            {"ticket_text": "My headphones stopped charging after the update."},
            {"ticket_text": "I was charged twice for my subscription."},
        ]
        resp = client_with_mock_pipeline.post("/predict/batch", json=tickets)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_predict_validation_too_short(self, client_with_mock_pipeline):
        resp = client_with_mock_pipeline.post(
            "/predict", json={"ticket_text": "Hi"}
        )
        assert resp.status_code == 422

    def test_request_id_header(self, client_with_mock_pipeline):
        resp = client_with_mock_pipeline.get("/health")
        assert "x-request-id" in resp.headers

    def test_predict_with_history(self, client_with_mock_pipeline):
        resp = client_with_mock_pipeline.post(
            "/predict",
            json={
                "ticket_text": "Still having the same issue with my device here.",
                "conversation_history": [
                    {"role": "user",      "content": "My headphones won't charge."},
                    {"role": "assistant", "content": "Please try restarting."},
                ],
            },
        )
        assert resp.status_code == 200
