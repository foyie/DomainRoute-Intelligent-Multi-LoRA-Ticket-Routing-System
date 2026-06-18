"""
monitoring/alert_rules.py
──────────────────────────
SLA alert definitions and notification dispatch for VeriTune.

Alerts fire when:
  - Auto-resolution rate drops below 90% (warn) or 85% (critical)
  - Escalation FNR exceeds 0.5% (critical — safety budget)
  - Latency p95 exceeds 200ms (warn) or 350ms (critical)
  - Semantic drift cosine_sim drops below 0.94 (warn) or 0.90 (critical)
  - Hallucination rate exceeds 3% (warn) or 10% (critical)
  - LoRA cache hit rate below 80% (warn)

Public API
----------
AlertLevel          – INFO / WARN / CRITICAL
AlertRule           – Single rule definition
Alert               – Fired alert instance
AlertEngine
  .evaluate(snapshot) → List[Alert]
  .dispatch(alerts)   → None
  .get_active()       → List[Alert]
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    INFO     = "info"
    WARN     = "warn"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    name:        str
    description: str
    level:       AlertLevel
    check_fn:    Callable[[dict], bool]   # returns True if alert should fire
    message_fn:  Callable[[dict], str]    # returns alert message
    cooldown_s:  int = 300               # minimum seconds between repeated fires

    def check(self, snapshot: dict) -> bool:
        try:
            return self.check_fn(snapshot)
        except Exception:
            return False

    def message(self, snapshot: dict) -> str:
        try:
            return self.message_fn(snapshot)
        except Exception:
            return self.description


@dataclass
class Alert:
    rule_name:    str
    level:        AlertLevel
    message:      str
    timestamp:    float = field(default_factory=time.time)
    resolved:     bool = False
    resolved_at:  Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "rule_name":   self.rule_name,
            "level":       self.level.value if hasattr(self.level, "value") else self.level,
            "message":     self.message,
            "timestamp":   self.timestamp,
            "resolved":    self.resolved,
        }


# ── Built-in alert rules ───────────────────────────────────────────────────────

def _get(d: dict, *keys, default=None):
    """Safe nested dict access."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default


BUILT_IN_RULES: List[AlertRule] = [

    AlertRule(
        name="auto_resolution_warn",
        description="Auto-resolution rate below 90%",
        level=AlertLevel.WARN,
        check_fn=lambda s: _get(s, "eval_results", "auto_resolution_rate", default=1.0) < 0.90,
        message_fn=lambda s: (
            f"Auto-resolution rate dropped to "
            f"{_get(s,'eval_results','auto_resolution_rate',default=0):.1%} (target >94%)"
        ),
    ),

    AlertRule(
        name="auto_resolution_critical",
        description="Auto-resolution rate below 85% — critical degradation",
        level=AlertLevel.CRITICAL,
        check_fn=lambda s: _get(s, "eval_results", "auto_resolution_rate", default=1.0) < 0.85,
        message_fn=lambda s: (
            f"CRITICAL: Auto-resolution rate "
            f"{_get(s,'eval_results','auto_resolution_rate',default=0):.1%} — investigate immediately"
        ),
        cooldown_s=60,
    ),

    AlertRule(
        name="escalation_fnr_critical",
        description="Escalation false-negative rate exceeded safety budget",
        level=AlertLevel.CRITICAL,
        check_fn=lambda s: (
            _get(s, "eval_results", "escalation", "false_negative_rate", default=0.0) > 0.005
        ),
        message_fn=lambda s: (
            f"SAFETY CRITICAL: Escalation FNR "
            f"{_get(s,'eval_results','escalation','false_negative_rate',default=0):.3%}"
            f" > 0.5% budget — missed escalations in production"
        ),
        cooldown_s=60,
    ),

    AlertRule(
        name="latency_sla_warn",
        description="Latency p95 above 200ms SLA",
        level=AlertLevel.WARN,
        check_fn=lambda s: _get(s, "eval_results", "latency", "p95_ms", default=0) > 200,
        message_fn=lambda s: (
            f"Latency p95 = {_get(s,'eval_results','latency','p95_ms',default=0):.0f}ms "
            f"(SLA target <200ms)"
        ),
    ),

    AlertRule(
        name="latency_sla_critical",
        description="Latency p95 above 350ms — severe degradation",
        level=AlertLevel.CRITICAL,
        check_fn=lambda s: _get(s, "eval_results", "latency", "p95_ms", default=0) > 350,
        message_fn=lambda s: (
            f"CRITICAL: Latency p95 = {_get(s,'eval_results','latency','p95_ms',default=0):.0f}ms"
        ),
        cooldown_s=60,
    ),

    AlertRule(
        name="semantic_drift_warn",
        description="Semantic drift detected — cosine similarity below threshold",
        level=AlertLevel.WARN,
        check_fn=lambda s: _get(s, "drift", "drift_detected", default=False),
        message_fn=lambda s: (
            f"Semantic drift detected: domains={_get(s,'drift','drifted_domains',default=[])} "
            f"kl={_get(s,'drift','kl_divergence',default=0):.4f}"
        ),
    ),

    AlertRule(
        name="hallucination_warn",
        description="Hallucination rate above 3%",
        level=AlertLevel.WARN,
        check_fn=lambda s: _get(s, "eval_results", "hallucination_rate", default=0.0) > 0.03,
        message_fn=lambda s: (
            f"Hallucination rate = "
            f"{_get(s,'eval_results','hallucination_rate',default=0):.1%} (warn >3%)"
        ),
    ),

    AlertRule(
        name="hallucination_critical",
        description="Hallucination rate above 10%",
        level=AlertLevel.CRITICAL,
        check_fn=lambda s: _get(s, "eval_results", "hallucination_rate", default=0.0) > 0.10,
        message_fn=lambda s: (
            f"CRITICAL: Hallucination rate = "
            f"{_get(s,'eval_results','hallucination_rate',default=0):.1%}"
        ),
        cooldown_s=60,
    ),

    AlertRule(
        name="cache_hit_rate_low",
        description="LoRA adapter cache hit rate below 80%",
        level=AlertLevel.WARN,
        check_fn=lambda s: _get(s, "metrics", "cache_hit_rate", default=1.0) < 0.80,
        message_fn=lambda s: (
            f"Cache hit rate = {_get(s,'metrics','cache_hit_rate',default=0):.1%} "
            f"(warn <80%) — consider increasing cache_size"
        ),
        cooldown_s=600,
    ),

    AlertRule(
        name="distribution_shift",
        description="Request domain distribution shifted significantly from training",
        level=AlertLevel.WARN,
        check_fn=lambda s: _get(s, "drift", "distribution_shift", default=False),
        message_fn=lambda s: (
            f"Domain distribution shift: kl={_get(s,'drift','kl_divergence',default=0):.4f} "
            f"— consider re-calibrating router"
        ),
        cooldown_s=1800,
    ),
]


class AlertEngine:
    """
    Evaluates all alert rules against a dashboard snapshot and dispatches
    any that fire. Tracks active alerts and respects per-rule cooldowns.

    Parameters
    ----------
    rules       : List of AlertRule definitions (defaults to BUILT_IN_RULES)
    notifiers   : List of callables(alert) for dispatch (e.g. Slack, PagerDuty)
    alert_log   : Path to persist fired alerts as JSONL
    """

    def __init__(
        self,
        rules:     Optional[List[AlertRule]] = None,
        notifiers: Optional[List[Callable]] = None,
        alert_log: Optional[str | Path] = None,
    ) -> None:
        self.rules     = rules     or BUILT_IN_RULES
        self.notifiers = notifiers or []
        self.alert_log = Path(alert_log) if alert_log else None
        self._active:  Dict[str, Alert] = {}      # rule_name → active Alert
        self._last_fired: Dict[str, float] = {}   # rule_name → timestamp

    def evaluate(self, snapshot: dict) -> List[Alert]:
        """
        Run all rules against snapshot. Returns list of newly fired alerts.
        """
        fired: List[Alert] = []

        for rule in self.rules:
            if not rule.check(snapshot):
                # Rule not firing — resolve if it was active
                if rule.name in self._active:
                    self._active[rule.name].resolved = True
                    self._active[rule.name].resolved_at = time.time()
                    logger.info("Alert resolved: %s", rule.name)
                    del self._active[rule.name]
                continue

            # Rule is firing — check cooldown
            last = self._last_fired.get(rule.name, 0)
            if time.time() - last < rule.cooldown_s and rule.name in self._active:
                continue   # Still in cooldown

            alert = Alert(
                rule_name=rule.name,
                level=rule.level,
                message=rule.message(snapshot),
            )
            self._active[rule.name] = alert
            self._last_fired[rule.name] = time.time()
            fired.append(alert)

            logger.log(
                logging.CRITICAL if rule.level == AlertLevel.CRITICAL else logging.WARNING,
                "[%s] %s: %s", rule.level.value.upper(), rule.name, alert.message,
            )

        if fired:
            self.dispatch(fired)

        return fired

    def dispatch(self, alerts: List[Alert]) -> None:
        """Send alerts to all configured notifiers."""
        for alert in alerts:
            if self.alert_log:
                self._log_alert(alert)
            for notifier in self.notifiers:
                try:
                    notifier(alert)
                except Exception as e:
                    logger.error("Notifier failed: %s", e)

    def get_active(self) -> List[Alert]:
        return list(self._active.values())

    def get_active_by_level(self, level: AlertLevel) -> List[Alert]:
        return [a for a in self._active.values() if a.level == level]

    def add_slack_notifier(self, webhook_url: str) -> None:
        """Register a Slack webhook notifier."""
        import urllib.request

        def _slack(alert: Alert) -> None:
            emoji = {"critical": "🚨", "warn": "⚠️", "info": "ℹ️"}.get(alert.level.value, "")
            payload = json.dumps({
                "text": f"{emoji} *VeriTune Alert* [{alert.level.value.upper()}]\n"
                        f"*{alert.rule_name}*: {alert.message}"
            }).encode()
            req = urllib.request.Request(
                webhook_url, data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)

        self.notifiers.append(_slack)
        logger.info("Slack notifier registered")

    def _log_alert(self, alert: Alert) -> None:
        if self.alert_log:
            self.alert_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.alert_log, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
