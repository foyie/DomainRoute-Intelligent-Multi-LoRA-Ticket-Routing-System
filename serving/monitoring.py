"""
serving/monitoring.py
──────────────────────
Observability layer for VeriTune serving.

Exposes:
  - Prometheus counters / histograms for all key business metrics
  - Structured JSON logging (compatible with Datadog / CloudWatch)
  - Per-request trace context (trace_id, session_id, domain, latency)
  - /metrics endpoint for Prometheus scraping

Metrics exposed
---------------
  veritune_requests_total          counter   {domain, status}
  veritune_request_latency_seconds histogram {domain, stage}
  veritune_escalations_total       counter   {reason}
  veritune_domain_routing_total    counter   {domain, method, confident}
  veritune_lora_cache_hits_total   counter   {hit}
  veritune_hallucination_total     counter   {domain, flagged}
  veritune_cost_dollars_total      counter   {domain}
  veritune_auto_resolution_rate    gauge     {domain}

Public API
----------
VeriTuneMetrics    – Prometheus registry + all metric objects
RequestTrace       – Per-request context dataclass
get_metrics()      – Singleton accessor
record_request(trace) → None
setup_structured_logging() → None
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Request trace dataclass ────────────────────────────────────────────────────

@dataclass
class RequestTrace:
    """Captures all timing and metadata for a single /predict request."""
    trace_id:          str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id:        Optional[str] = None
    customer_id:       Optional[str] = None
    domain:            str   = "unknown"
    routing_method:    str   = "semantic"
    routing_confidence: float = 0.0
    escalation_detected: bool = False
    escalation_score:  float  = 0.0
    lora_rank:         int    = 32
    cache_hit:         bool   = False
    resolution_status: str    = "resolved"
    hallucination_flagged: bool = False
    cost_dollars:      float   = 0.0
    # Stage latencies (ms)
    router_ms:         float  = 0.0
    lora_load_ms:      float  = 0.0
    generation_ms:     float  = 0.0
    safety_ms:         float  = 0.0
    total_ms:          float  = 0.0
    # Request metadata
    ticket_length:     int    = 0
    response_length:   int    = 0
    tokens_generated:  int    = 0
    status:            str    = "success"   # "success" | "error" | "escalated"
    error_message:     Optional[str] = None
    request_start:     float  = field(default_factory=time.perf_counter)

    def finalise(self) -> None:
        """Compute total latency and set final status."""
        self.total_ms = (time.perf_counter() - self.request_start) * 1000
        if self.escalation_detected:
            self.status = "escalated"

    def to_log_dict(self) -> dict:
        return {
            "trace_id":             self.trace_id,
            "session_id":           self.session_id,
            "domain":               self.domain,
            "routing_method":       self.routing_method,
            "routing_confidence":   round(self.routing_confidence, 3),
            "escalation_detected":  self.escalation_detected,
            "resolution_status":    self.resolution_status,
            "cache_hit":            self.cache_hit,
            "lora_rank":            self.lora_rank,
            "latency_ms": {
                "router":     round(self.router_ms, 1),
                "lora_load":  round(self.lora_load_ms, 1),
                "generation": round(self.generation_ms, 1),
                "safety":     round(self.safety_ms, 1),
                "total":      round(self.total_ms, 1),
            },
            "cost_dollars":         round(self.cost_dollars, 5),
            "hallucination_flagged": self.hallucination_flagged,
            "status":               self.status,
        }


# ── Prometheus metrics registry ────────────────────────────────────────────────

class VeriTuneMetrics:
    """
    Singleton Prometheus metrics registry for VeriTune.
    Falls back gracefully if prometheus_client is not installed.
    """

    _instance: Optional["VeriTuneMetrics"] = None

    def __init__(self) -> None:
        self._prom_available = False
        self._init_prometheus()
        # In-memory counters as fallback / supplement
        self._counters: dict = {
            "requests_total": 0,
            "escalations_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "hallucinations_flagged": 0,
        }
        self._domain_counts:    dict = {}
        self._latency_samples:  list = []
        self._resolution_rates: dict = {}

    def _init_prometheus(self) -> None:
        try:
            from prometheus_client import (
                Counter, Histogram, Gauge, CollectorRegistry, REGISTRY,
            )
            # Use default registry to work with /metrics endpoint
            self.requests_total = Counter(
                "veritune_requests_total",
                "Total requests processed",
                ["domain", "status"],
            )
            self.request_latency = Histogram(
                "veritune_request_latency_seconds",
                "Request latency by stage",
                ["domain", "stage"],
                buckets=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0],
            )
            self.escalations_total = Counter(
                "veritune_escalations_total",
                "Total escalations triggered",
                ["reason"],
            )
            self.domain_routing_total = Counter(
                "veritune_domain_routing_total",
                "Routing decisions by domain",
                ["domain", "method", "confident"],
            )
            self.lora_cache_hits = Counter(
                "veritune_lora_cache_hits_total",
                "LoRA adapter cache hit/miss",
                ["hit"],
            )
            self.hallucination_total = Counter(
                "veritune_hallucination_total",
                "Hallucination detection results",
                ["domain", "flagged"],
            )
            self.cost_dollars = Counter(
                "veritune_cost_dollars_total",
                "Estimated cost in dollars",
                ["domain"],
            )
            self.auto_resolution_rate = Gauge(
                "veritune_auto_resolution_rate",
                "Rolling auto-resolution rate",
                ["domain"],
            )
            self._prom_available = True
            logger.info("Prometheus metrics initialised")
        except Exception as e:
            logger.warning("Prometheus not available (%s) — using in-memory counters", e)

    @classmethod
    def get(cls) -> "VeriTuneMetrics":
        if cls._instance is None:
            cls._instance = VeriTuneMetrics()
        return cls._instance

    def record(self, trace: RequestTrace) -> None:
        """Record all metrics from a completed RequestTrace."""
        # In-memory counters (always)
        self._counters["requests_total"] += 1
        d = trace.domain
        self._domain_counts[d] = self._domain_counts.get(d, 0) + 1
        self._latency_samples.append(trace.total_ms)
        if len(self._latency_samples) > 10_000:
            self._latency_samples = self._latency_samples[-5_000:]

        if trace.escalation_detected:
            self._counters["escalations_total"] += 1
        if trace.cache_hit:
            self._counters["cache_hits"] += 1
        else:
            self._counters["cache_misses"] += 1
        if trace.hallucination_flagged:
            self._counters["hallucinations_flagged"] += 1

        # Prometheus (if available)
        if not self._prom_available:
            return
        try:
            self.requests_total.labels(domain=d, status=trace.status).inc()
            self.request_latency.labels(domain=d, stage="router").observe(trace.router_ms / 1000)
            self.request_latency.labels(domain=d, stage="lora_load").observe(trace.lora_load_ms / 1000)
            self.request_latency.labels(domain=d, stage="generation").observe(trace.generation_ms / 1000)
            self.request_latency.labels(domain=d, stage="total").observe(trace.total_ms / 1000)
            self.domain_routing_total.labels(
                domain=d,
                method=trace.routing_method,
                confident=str(trace.routing_confidence >= 0.70),
            ).inc()
            self.lora_cache_hits.labels(hit=str(trace.cache_hit)).inc()
            self.hallucination_total.labels(domain=d, flagged=str(trace.hallucination_flagged)).inc()
            self.cost_dollars.labels(domain=d).inc(trace.cost_dollars)
            if trace.escalation_detected:
                reason = "keyword" if trace.escalation_score > 0.5 else "model"
                self.escalations_total.labels(reason=reason).inc()
        except Exception as e:
            logger.debug("Prometheus record error: %s", e)

    def snapshot(self) -> dict:
        """Return current in-memory metric snapshot."""
        import numpy as np
        lats = self._latency_samples or [0.0]
        return {
            "requests_total":        self._counters["requests_total"],
            "escalations_total":     self._counters["escalations_total"],
            "cache_hit_rate":        round(
                self._counters["cache_hits"] /
                max(self._counters["cache_hits"] + self._counters["cache_misses"], 1), 3
            ),
            "hallucinations_flagged": self._counters["hallucinations_flagged"],
            "latency_p50_ms":        round(float(np.percentile(lats, 50)), 1),
            "latency_p95_ms":        round(float(np.percentile(lats, 95)), 1),
            "domain_distribution":   dict(self._domain_counts),
        }


# ── Structured logging setup ───────────────────────────────────────────────────

class StructuredFormatter(logging.Formatter):
    """Emit log records as JSON for ingestion by Datadog / CloudWatch."""

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
            "service": "veritune",
        }
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields
        for key in ("trace_id", "domain", "latency_ms"):
            if hasattr(record, key):
                log[key] = getattr(record, key)
        return json.dumps(log)


def setup_structured_logging(level: str = "INFO") -> None:
    """Configure root logger to emit structured JSON."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        root.addHandler(handler)
    logger.info("Structured logging configured (level=%s)", level)


# ── Singleton accessor ─────────────────────────────────────────────────────────

def get_metrics() -> VeriTuneMetrics:
    return VeriTuneMetrics.get()
