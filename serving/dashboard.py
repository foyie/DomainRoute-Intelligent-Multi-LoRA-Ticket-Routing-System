"""
serving/dashboard.py
─────────────────────
FastAPI router that exposes live dashboard data endpoints.
Mounted at /dashboard/* in serving/main.py.

Endpoints
---------
  GET /dashboard/snapshot          → Full DashboardSnapshot JSON
  GET /dashboard/metrics           → Live Prometheus counters
  GET /dashboard/eval              → Latest evaluation results
  GET /dashboard/drift             → Latest drift check
  GET /dashboard/pareto            → Pareto frontier plot data
  GET /dashboard/ab-test           → A/B test significance results
  GET /dashboard/sparklines        → Rolling time-series for charts
  GET /dashboard/alerts            → Active alert list
  POST /dashboard/drift/check      → Trigger manual drift check
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from monitoring.dashboard_data import DashboardDataAggregator
from monitoring.alert_rules import AlertEngine, BUILT_IN_RULES
from monitoring.drift_monitor import DriftMonitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# ── Singletons (initialised once, shared across requests) ──────────────────────
_aggregator: Optional[DashboardDataAggregator] = None
_alert_engine: Optional[AlertEngine] = None
_drift_monitor: Optional[DriftMonitor] = None


def init_dashboard(
    metrics_fn=None,
    alert_log: str = "outputs/results/alerts.jsonl",
) -> None:
    """
    Initialise dashboard singletons. Called from serving/main.py lifespan.
    """
    global _aggregator, _alert_engine, _drift_monitor

    _aggregator   = DashboardDataAggregator(metrics_fn=metrics_fn)
    _alert_engine = AlertEngine(
        rules=BUILT_IN_RULES,
        alert_log=alert_log,
    )
    _drift_monitor = DriftMonitor()
    logger.info("Dashboard layer initialised")


def get_aggregator() -> DashboardDataAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = DashboardDataAggregator()
    return _aggregator


def get_alert_engine() -> AlertEngine:
    global _alert_engine
    if _alert_engine is None:
        _alert_engine = AlertEngine()
    return _alert_engine


def get_drift_monitor() -> DriftMonitor:
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/snapshot")
async def snapshot():
    """
    Full dashboard data snapshot — all metrics, eval results, drift, and alerts
    in a single request. Used by the dashboard frontend for initial load.
    """
    agg      = get_aggregator()
    snap     = agg.get_snapshot()
    engine   = get_alert_engine()
    alerts   = engine.evaluate(snap.to_dict())
    snap_d   = snap.to_dict()
    snap_d["active_alerts"] = [a.to_dict() for a in engine.get_active()]
    snap_d["n_active_alerts"] = len(engine.get_active())
    return JSONResponse(content=snap_d)


@router.get("/metrics")
async def live_metrics():
    """Live Prometheus counters — requests/sec, latency percentiles, cache hit rate."""
    return JSONResponse(content=get_aggregator().get_metrics())


@router.get("/eval")
async def eval_results():
    """Latest domain evaluation: auto-resolution, escalation, latency, cost."""
    return JSONResponse(content=get_aggregator().get_eval())


@router.get("/drift")
async def drift_status():
    """Latest semantic drift check result across all domains."""
    return JSONResponse(content=get_aggregator().get_drift())


@router.get("/pareto")
async def pareto_data():
    """Pareto frontier data for the accuracy × cost × latency scatter plot."""
    return JSONResponse(content=get_aggregator().get_pareto())


@router.get("/ab-test")
async def ab_test_results():
    """A/B test statistical results: routed vs single LoRA."""
    return JSONResponse(content=get_aggregator().get_ab_test())


@router.get("/sparklines")
async def sparklines():
    """Rolling time-series data for dashboard sparkline charts (last 24 points)."""
    return JSONResponse(content=get_aggregator().get_sparklines())


@router.get("/alerts")
async def active_alerts():
    """Currently active alerts with level (info/warn/critical) and message."""
    engine = get_alert_engine()
    snap   = get_aggregator().get_snapshot().to_dict()
    engine.evaluate(snap)
    return JSONResponse(content={
        "active": [a.to_dict() for a in engine.get_active()],
        "n_critical": len(engine.get_active_by_level(
            __import__("monitoring.alert_rules", fromlist=["AlertLevel"]).AlertLevel.CRITICAL
        )),
        "n_warn": len(engine.get_active_by_level(
            __import__("monitoring.alert_rules", fromlist=["AlertLevel"]).AlertLevel.WARN
        )),
        "timestamp": time.time(),
    })


@router.post("/drift/check")
async def trigger_drift_check():
    """
    Manually trigger a semantic drift check.
    Runs synchronously and returns the result immediately.
    """
    monitor = get_drift_monitor()
    result  = monitor.run_check()
    return JSONResponse(content={
        "drift_detected":    result.drift_detected,
        "drifted_domains":   result.drifted_domains,
        "cosine_sims":       result.cosine_sims,
        "kl_divergence":     result.kl_divergence,
        "distribution_shift": result.distribution_shift,
        "check_duration_ms": result.check_duration_ms,
        "timestamp":         result.timestamp,
        "alert_fired":       result.alert_fired,
    })


@router.get("/health")
async def dashboard_health():
    """Dashboard subsystem health check."""
    return JSONResponse(content={
        "dashboard": "ok",
        "aggregator": _aggregator is not None,
        "alert_engine": _alert_engine is not None,
        "drift_monitor": _drift_monitor is not None,
        "timestamp": time.time(),
    })
