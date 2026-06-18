"""
tests/test_monitoring.py  — Phase 6 monitoring tests
"""
from __future__ import annotations
import json, time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from monitoring.drift_monitor import DriftMonitor, DriftCheckResult
from monitoring.dashboard_data import DashboardDataAggregator, DashboardSnapshot
from monitoring.alert_rules import AlertEngine, AlertLevel, AlertRule, Alert, BUILT_IN_RULES


# ── DriftMonitor ──────────────────────────────────────────────────────────────

class TestDriftMonitor:
    def _m(self, tmp_path):
        return DriftMonitor(probe_size=10, results_dir=str(tmp_path/"results"))

    def test_ingest_populates(self, tmp_path):
        m = self._m(tmp_path)
        m.ingest_request("My device.", "technical")
        assert len(m._request_buffer["technical"]) == 1

    def test_ingest_caps_buffer(self, tmp_path):
        m = self._m(tmp_path)
        for i in range(30): m.ingest_request(f"T{i}", "technical")
        assert len(m._request_buffer["technical"]) <= m.probe_size * 2

    def test_run_check_returns_result(self, tmp_path):
        m = self._m(tmp_path)
        for i in range(5): m.ingest_request(f"T{i}", "technical")
        r = m.run_check()
        assert isinstance(r, DriftCheckResult)
        assert r.check_duration_ms >= 0
        assert r.timestamp != ""

    def test_run_check_saves_file(self, tmp_path):
        m = self._m(tmp_path)
        m.run_check()
        assert (tmp_path/"results"/"latest_drift_check.json").exists()

    def test_run_check_fires_alert(self, tmp_path):
        fired = []
        m = DriftMonitor(probe_size=5, alert_fn=lambda r: fired.append(r),
                         threshold=1.01, results_dir=str(tmp_path/"results"))
        for i in range(3): m.ingest_request(f"T{i}", "technical")
        r = m.run_check()
        assert r.drift_detected is True

    def test_history_accumulates(self, tmp_path):
        m = self._m(tmp_path)
        for _ in range(4): m.run_check()
        assert len(m.get_history()) == 4

    def test_get_history_slice(self, tmp_path):
        m = self._m(tmp_path)
        for _ in range(6): m.run_check()
        assert len(m.get_history(3)) <= 3

    def test_export_report(self, tmp_path):
        m = self._m(tmp_path)
        m.run_check()
        path = tmp_path/"report.json"
        m.export_report(path)
        data = json.loads(path.read_text())
        assert "history" in data and "threshold" in data

    def test_start_stop_scheduler(self, tmp_path):
        m = self._m(tmp_path)
        m.start_scheduler(interval_minutes=9999)
        assert m._scheduler.is_alive()
        m.stop_scheduler()
        assert m._stop_event.is_set()

    def test_no_duplicate_scheduler(self, tmp_path):
        m = self._m(tmp_path)
        m.start_scheduler(9999); t1 = m._scheduler
        m.start_scheduler(9999)
        assert m._scheduler is t1
        m.stop_scheduler()

    def test_summary_ok(self):
        r = DriftCheckResult("2024-01-01T00:00:00", ["technical"],
            {"technical": 0.962}, {"technical":1.0}, False, [], False, 0.002, 10, 5.0)
        assert "OK" in r.summary()

    def test_summary_drift(self):
        r = DriftCheckResult("2024-01-01T00:00:00", ["technical"],
            {"technical": 0.88}, {"technical":1.0}, True, ["technical"], False, 0.001, 10, 5.0)
        assert "DRIFT" in r.summary()

    def test_kl_identical(self, tmp_path):
        m = self._m(tmp_path)
        p = {"a":0.5,"b":0.5}
        assert abs(m._kl_divergence(p, p)) < 1e-6

    def test_kl_different(self, tmp_path):
        m = self._m(tmp_path)
        assert m._kl_divergence({"a":0.9,"b":0.1}, {"a":0.1,"b":0.9}) > 0


# ── DashboardDataAggregator ───────────────────────────────────────────────────

class TestDashboardAggregator:
    def test_snapshot_returns_snapshot(self):
        snap = DashboardDataAggregator().get_snapshot()
        assert isinstance(snap, DashboardSnapshot) and snap.timestamp > 0

    def test_snapshot_all_keys(self):
        d = DashboardDataAggregator().get_snapshot().to_dict()
        for k in ["timestamp","metrics","eval_results","drift","pareto","ab_test","sparklines","checkpoints"]:
            assert k in d

    def test_metrics_demo_keys(self):
        m = DashboardDataAggregator().get_metrics()
        assert "requests_total" in m and "cache_hit_rate" in m

    def test_metrics_fn_called(self):
        fn = MagicMock(return_value={"requests_total":42,"cache_hit_rate":0.9})
        m  = DashboardDataAggregator(metrics_fn=fn).get_metrics()
        fn.assert_called_once()
        assert m["requests_total"] == 42

    def test_eval_demo_structure(self):
        ev = DashboardDataAggregator().get_eval()
        assert "auto_resolution_rate" in ev
        assert 0.0 <= ev["auto_resolution_rate"] <= 1.0

    def test_eval_loads_from_file(self, tmp_path):
        rd = tmp_path/"results"; rd.mkdir()
        data = {"results":[{"auto_resolution_rate":0.91,"escalation":{},"latency":{},"cost":{},"hallucination_rate":0.01,"composite_score":0.8,"per_domain":{}}]}
        (rd/"domain_evaluation.json").write_text(json.dumps(data))
        ev = DashboardDataAggregator(results_dir=rd).get_eval()
        assert ev["auto_resolution_rate"] == 0.91

    def test_pareto_has_points(self):
        p = DashboardDataAggregator().get_pareto()
        assert "points" in p and len(p["points"]) > 0

    def test_ab_test_significance(self):
        ab = DashboardDataAggregator().get_ab_test()
        assert "accuracy_delta" in ab and "is_significant" in ab
        assert ab["is_significant"] is True

    def test_sparklines_length(self):
        sl = DashboardDataAggregator().get_sparklines()
        n  = sl["n_points"]
        assert len(sl["auto_resolution_rate"]) == n
        assert len(sl["latency_p95_ms"]) == n

    def test_update_sparkline_appends(self):
        agg = DashboardDataAggregator()
        agg.update_sparkline("latency_p95_ms", 150.0)
        assert 150.0 in agg._sparkline_buffer["latency_p95_ms"]

    def test_update_sparkline_caps(self):
        agg = DashboardDataAggregator()
        for i in range(110): agg.update_sparkline("latency_p95_ms", float(i))
        assert len(agg._sparkline_buffer["latency_p95_ms"]) == 100

    def test_snapshot_serialisable(self):
        d = DashboardDataAggregator().get_snapshot().to_dict()
        assert len(json.dumps(d, default=str)) > 100


# ── AlertEngine ───────────────────────────────────────────────────────────────

def _snap(arr=0.943, fnr=0.002, lat=148.0, drift=False, hall=0.012, cache=0.94):
    return {
        "eval_results": {
            "auto_resolution_rate": arr,
            "escalation": {"false_negative_rate": fnr},
            "latency":    {"p95_ms": lat},
            "hallucination_rate": hall,
        },
        "drift":   {"drift_detected": drift, "drifted_domains": ["t"] if drift else [],
                    "kl_divergence": 0.2 if drift else 0.001, "distribution_shift": False},
        "metrics": {"cache_hit_rate": cache},
    }


class TestAlertEngine:
    def test_no_alerts_good(self):
        assert len(AlertEngine().evaluate(_snap())) == 0

    def test_warn_low_arr(self):
        names = [a.rule_name for a in AlertEngine().evaluate(_snap(arr=0.88))]
        assert "auto_resolution_warn" in names

    def test_critical_very_low_arr(self):
        names = [a.rule_name for a in AlertEngine().evaluate(_snap(arr=0.82))]
        assert "auto_resolution_critical" in names

    def test_critical_escalation_fnr(self):
        names = [a.rule_name for a in AlertEngine().evaluate(_snap(fnr=0.01))]
        assert "escalation_fnr_critical" in names

    def test_warn_high_latency(self):
        names = [a.rule_name for a in AlertEngine().evaluate(_snap(lat=250))]
        assert "latency_sla_warn" in names

    def test_critical_severe_latency(self):
        names = [a.rule_name for a in AlertEngine().evaluate(_snap(lat=400))]
        assert "latency_sla_critical" in names

    def test_warn_drift(self):
        names = [a.rule_name for a in AlertEngine().evaluate(_snap(drift=True))]
        assert "semantic_drift_warn" in names

    def test_warn_hallucination(self):
        names = [a.rule_name for a in AlertEngine().evaluate(_snap(hall=0.05))]
        assert "hallucination_warn" in names

    def test_critical_hallucination(self):
        names = [a.rule_name for a in AlertEngine().evaluate(_snap(hall=0.15))]
        assert "hallucination_critical" in names

    def test_warn_cache_low(self):
        names = [a.rule_name for a in AlertEngine().evaluate(_snap(cache=0.70))]
        assert "cache_hit_rate_low" in names

    def test_alert_resolves(self):
        e = AlertEngine()
        e.evaluate(_snap(arr=0.88))
        assert "auto_resolution_warn" in [a.rule_name for a in e.get_active()]
        e.evaluate(_snap(arr=0.95))
        assert "auto_resolution_warn" not in [a.rule_name for a in e.get_active()]

    def test_cooldown_prevents_duplicate(self):
        rule = AlertRule("cd_test","desc",AlertLevel.WARN,lambda s:True,lambda s:"msg",cooldown_s=9999)
        e    = AlertEngine(rules=[rule])
        e.evaluate({})
        assert len(e.evaluate({})) == 0

    def test_dispatch_calls_notifier(self):
        calls = []
        AlertEngine(notifiers=[lambda a: calls.append(a)]).evaluate(_snap(arr=0.82))
        assert len(calls) >= 1

    def test_dispatch_logs_to_file(self, tmp_path):
        log = tmp_path/"alerts.jsonl"
        AlertEngine(alert_log=log).evaluate(_snap(arr=0.82))
        assert log.exists()
        assert json.loads(log.read_text().strip().split("\n")[0])["rule_name"]

    def test_get_active_by_level(self):
        e = AlertEngine()
        e.evaluate(_snap(fnr=0.02, arr=0.80, lat=400))
        assert len(e.get_active_by_level(AlertLevel.CRITICAL)) >= 1

    def test_add_slack_notifier(self):
        e = AlertEngine()
        n = len(e.notifiers)
        e.add_slack_notifier("https://hooks.slack.com/x")
        assert len(e.notifiers) == n + 1

    def test_alert_to_dict(self):
        d = Alert("r","warn","msg").to_dict()
        assert "rule_name" in d and "level" in d

    def test_rule_check_exception_safe(self):
        r = AlertRule("x","d",AlertLevel.WARN,lambda s:1/0,lambda s:"m")
        assert r.check({}) is False

    def test_rule_message_exception_fallback(self):
        r = AlertRule("x","fallback",AlertLevel.WARN,lambda s:False,lambda s:1/0)
        assert r.message({}) == "fallback"

    def test_built_in_rules_count(self):
        assert len(BUILT_IN_RULES) >= 8


# ── Dashboard FastAPI endpoints ───────────────────────────────────────────────

class TestDashboardEndpoints:
    @pytest.fixture
    def client(self):
        import serving.main as m
        from fastapi.testclient import TestClient
        with patch("serving.inference.build_pipeline", side_effect=Exception("no model")):
            with TestClient(m.app, raise_server_exceptions=False) as c:
                yield c

    def test_snapshot_200(self, client):
        r = client.get("/dashboard/snapshot")
        assert r.status_code == 200
        d = r.json()
        for k in ["timestamp","metrics","eval_results","drift","sparklines","active_alerts"]:
            assert k in d

    def test_metrics_200(self, client):
        r = client.get("/dashboard/metrics")
        assert r.status_code == 200 and "requests_total" in r.json()

    def test_eval_200(self, client):
        r = client.get("/dashboard/eval")
        assert r.status_code == 200 and "auto_resolution_rate" in r.json()

    def test_drift_200(self, client):
        r = client.get("/dashboard/drift")
        assert r.status_code == 200

    def test_pareto_200(self, client):
        r = client.get("/dashboard/pareto")
        assert r.status_code == 200 and len(r.json()["points"]) > 0

    def test_ab_test_200(self, client):
        r = client.get("/dashboard/ab-test")
        assert r.status_code == 200 and "accuracy_delta" in r.json()

    def test_sparklines_200(self, client):
        r = client.get("/dashboard/sparklines")
        assert r.status_code == 200 and "n_points" in r.json()

    def test_alerts_200(self, client):
        r = client.get("/dashboard/alerts")
        assert r.status_code == 200
        d = r.json()
        assert "active" in d and "n_critical" in d and "n_warn" in d

    def test_drift_check_post_200(self, client):
        r = client.post("/dashboard/drift/check")
        assert r.status_code == 200
        d = r.json()
        assert "drift_detected" in d and "cosine_sims" in d

    def test_dashboard_health_200(self, client):
        r = client.get("/dashboard/health")
        assert r.status_code == 200 and r.json()["dashboard"] == "ok"
