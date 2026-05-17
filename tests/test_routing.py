"""
tests/test_routing.py
──────────────────────
Tests for Phase 3 routing pipeline.

Coverage
--------
Models · IntentRouter · LoRASelector · LoRAComposer · LRU cache · LoRALoader helpers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from routing.models import (
    Domain, DomainScore, EscalationEvent, LatencyBreakdown,
    LoRASelection, RoutingDecision, RoutingMethod, TicketRequest,
)
from routing.lora_selector import (
    AdapterRegistry, LoRASelector,
    GLOBAL_LOW_CONFIDENCE_ESCALATION_THRESHOLD,
)
from routing.lora_composer import LoRAComposer, AblationResult
from routing.lora_loader import LoRALoader, _LRUCache, adapter_exists, load_adapter_config


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decision(
    primary: Domain = Domain.TECHNICAL,
    score: float = 0.85,
    escalation_detected: bool = False,
    escalation_score: float = 0.05,
    all_scores: List[DomainScore] | None = None,
) -> RoutingDecision:
    if all_scores is None:
        rem = (1.0 - score) / 3
        others = [d for d in [Domain.TECHNICAL, Domain.BILLING,
                               Domain.RETURNS, Domain.ESCALATION] if d != primary]
        all_scores = [DomainScore(domain=primary, score=score, rank=1)] + [
            DomainScore(domain=d, score=rem, rank=i + 2) for i, d in enumerate(others)
        ]
    return RoutingDecision(
        primary_domain=primary, primary_score=score,
        all_scores=all_scores, escalation_detected=escalation_detected,
        escalation_score=escalation_score, is_confident=score >= 0.70,
    )


def _state_dict(keys: List[str], dim: int = 16, seed: int = 0) -> dict:
    import torch
    rng = np.random.RandomState(seed)
    return {k: torch.tensor(rng.randn(dim, dim).astype(np.float32)) for k in keys}


def _make_router():
    """Fitted IntentRouter with mocked SBERT — no network or GPU."""
    from routing.intent_router import IntentRouter
    rng = np.random.RandomState(42)
    DIM = 64
    centres = {
        Domain.TECHNICAL:  rng.randn(DIM).astype(np.float32),
        Domain.BILLING:    rng.randn(DIM).astype(np.float32),
        Domain.RETURNS:    rng.randn(DIM).astype(np.float32),
        Domain.ESCALATION: rng.randn(DIM).astype(np.float32),
    }
    for d in centres:
        centres[d] /= np.linalg.norm(centres[d])

    router = IntentRouter(temperature=0.05)
    router._prototype_embeddings = dict(centres)
    router._training_texts = {d: [f"sample {i}"] for d in centres}
    router._is_fitted = True

    def mock_encode(texts, batch_size=64):
        embs = []
        for text in texts:
            matched = next((d for d in centres if d.value in text.lower()), None)
            vec = (centres[matched] + rng.randn(DIM).astype(np.float32) * 0.05
                   if matched else rng.randn(DIM).astype(np.float32))
            vec /= np.linalg.norm(vec) + 1e-9
            embs.append(vec)
        return np.stack(embs).astype(np.float32)

    router._encode = mock_encode
    return router, centres


def _make_selector(comp_thresh: float = 0.15) -> LoRASelector:
    registry = AdapterRegistry(
        paths={d: f"outputs/checkpoints/{d.value}_best"
               for d in Domain if d != Domain.UNKNOWN},
        ranks={Domain.TECHNICAL: 32, Domain.BILLING: 24,
               Domain.RETURNS: 28, Domain.ESCALATION: 8},
    )
    return LoRASelector(registry=registry, composition_threshold=comp_thresh)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic model tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTicketRequest:
    def test_valid(self):
        r = TicketRequest(ticket_text="My headphones won't charge.")
        assert r.ticket_text == "My headphones won't charge."

    def test_strips_whitespace(self):
        r = TicketRequest(ticket_text="  spaces  ")
        assert r.ticket_text == "spaces"

    def test_too_short_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TicketRequest(ticket_text="Hi")

    def test_too_long_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TicketRequest(ticket_text="x" * 4001)

    def test_valid_history(self):
        r = TicketRequest(
            ticket_text="Follow-up question right here.",
            conversation_history=[{"role": "user", "content": "hi"},
                                   {"role": "assistant", "content": "hello"}],
        )
        assert len(r.conversation_history) == 2

    def test_invalid_history_role_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TicketRequest(ticket_text="Valid ticket text here.",
                          conversation_history=[{"role": "alien", "content": "x"}])

    def test_force_domain(self):
        r = TicketRequest(ticket_text="Test ticket for routing.", force_domain=Domain.BILLING)
        assert r.force_domain == Domain.BILLING


class TestRoutingDecisionProperties:
    def test_runner_up(self):
        scores = [
            DomainScore(domain=Domain.BILLING,   score=0.80, rank=1),
            DomainScore(domain=Domain.TECHNICAL, score=0.12, rank=2),
            DomainScore(domain=Domain.RETURNS,   score=0.05, rank=3),
            DomainScore(domain=Domain.ESCALATION,score=0.03, rank=4),
        ]
        d = RoutingDecision(primary_domain=Domain.BILLING, primary_score=0.80, all_scores=scores)
        assert d.runner_up_domain == Domain.TECHNICAL

    def test_confidence_gap(self):
        scores = [
            DomainScore(domain=Domain.BILLING,   score=0.80, rank=1),
            DomainScore(domain=Domain.TECHNICAL, score=0.12, rank=2),
            DomainScore(domain=Domain.RETURNS,   score=0.05, rank=3),
            DomainScore(domain=Domain.ESCALATION,score=0.03, rank=4),
        ]
        d = RoutingDecision(primary_domain=Domain.BILLING, primary_score=0.80, all_scores=scores)
        assert d.confidence_gap == pytest.approx(0.68, abs=0.01)


class TestLoRASelectionValidation:
    def test_composition_mismatch_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LoRASelection(
                domain=Domain.TECHNICAL, adapter_path="p", lora_rank=32,
                selection_reason="x", confidence=0.75,
                composition_domains=[Domain.TECHNICAL, Domain.BILLING],
                composition_weights=[0.7],  # wrong length
            )

    def test_composition_valid(self):
        sel = LoRASelection(
            domain=Domain.TECHNICAL, adapter_path="p", lora_rank=32,
            selection_reason="x", confidence=0.75,
            composition_domains=[Domain.TECHNICAL, Domain.BILLING],
            composition_weights=[0.7, 0.3],
        )
        assert sum(sel.composition_weights) == pytest.approx(1.0)


class TestLatencyBreakdown:
    def test_compute_total(self):
        lb = LatencyBreakdown(router_ms=12, lora_load_ms=8, generation_ms=105, safety_ms=5)
        lb.compute_total()
        assert lb.total_ms == pytest.approx(130.0)


# ══════════════════════════════════════════════════════════════════════════════
# IntentRouter tests
# ══════════════════════════════════════════════════════════════════════════════

class TestIntentRouter:
    def test_route_technical(self):
        router, _ = _make_router()
        d = router.route("My technical device won't sync to technical bluetooth.")
        assert d.primary_domain == Domain.TECHNICAL

    def test_route_billing(self):
        router, _ = _make_router()
        d = router.route("I need help with my billing charge and billing refund.")
        assert d.primary_domain == Domain.BILLING

    def test_escalation_keyword_threat(self):
        from routing.intent_router import IntentRouter
        r = IntentRouter()
        assert r._keyword_escalation_score("I will file a chargeback and take legal action.") >= 0.50

    def test_escalation_keyword_anger(self):
        from routing.intent_router import IntentRouter
        r = IntentRouter()
        assert r._keyword_escalation_score("This is furious and unacceptable!") > 0.0

    def test_clean_ticket_zero_escalation(self):
        from routing.intent_router import IntentRouter
        r = IntentRouter()
        assert r._keyword_escalation_score("My headphones stopped charging after firmware update.") == 0.0

    def test_force_domain_override(self):
        router, _ = _make_router()
        req = TicketRequest(ticket_text="Ambiguous ticket text is right here.", force_domain=Domain.BILLING)
        d = router.route("Ambiguous ticket text is right here.", request=req)
        assert d.primary_domain == Domain.BILLING
        assert d.routing_method == RoutingMethod.DEFAULT

    def test_batch_route_length(self):
        router, _ = _make_router()
        tickets = ["technical device issue", "billing payment refund", "returns exchange label"]
        decisions = router.batch_route(tickets)
        assert len(decisions) == 3

    def test_all_scores_four_domains(self):
        router, _ = _make_router()
        d = router.route("My technical headphones won't connect to technical bluetooth.")
        assert {s.domain for s in d.all_scores} >= {
            Domain.TECHNICAL, Domain.BILLING, Domain.RETURNS, Domain.ESCALATION
        }

    def test_scores_sum_approx_one(self):
        router, _ = _make_router()
        d = router.route("I have a billing question about my billing subscription.")
        assert abs(sum(s.score for s in d.all_scores) - 1.0) < 0.05

    def test_router_latency_populated(self):
        router, _ = _make_router()
        d = router.route("Technical issue with my device firmware update.")
        assert d.router_latency_ms >= 0.0

    def test_save_load_roundtrip(self, tmp_path):
        from routing.intent_router import IntentRouter
        router, _ = _make_router()
        router.save(tmp_path)
        loaded = IntentRouter.load(tmp_path)
        assert loaded._is_fitted
        for domain in router._prototype_embeddings:
            np.testing.assert_allclose(
                loaded._prototype_embeddings[domain],
                router._prototype_embeddings[domain], atol=1e-5,
            )

    def test_unfitted_raises(self):
        from routing.intent_router import IntentRouter
        r = IntentRouter()
        with pytest.raises(RuntimeError, match="not fitted"):
            r.route("test ticket text here now")

    def test_keyword_domain_technical(self):
        from routing.intent_router import IntentRouter
        r = IntentRouter()
        assert r._keyword_domain_score("firmware update crashed and bluetooth won't sync") == Domain.TECHNICAL

    def test_keyword_domain_billing(self):
        from routing.intent_router import IntentRouter
        r = IntentRouter()
        assert r._keyword_domain_score("I need a refund for my billing charge on my invoice") == Domain.BILLING

    def test_keyword_domain_returns(self):
        from routing.intent_router import IntentRouter
        r = IntentRouter()
        assert r._keyword_domain_score("I want to return this item and exchange for replacement") == Domain.RETURNS

    def test_build_context_with_history(self):
        from routing.intent_router import IntentRouter
        r = IntentRouter()
        ctx = r._build_context("New question", [{"role": "user", "content": "Prior msg"}])
        assert "Prior msg" in ctx and "New question" in ctx

    def test_build_context_no_history(self):
        from routing.intent_router import IntentRouter
        r = IntentRouter()
        assert r._build_context("Just ticket", []) == "Just ticket"

    def test_ece_perfect_predictions(self):
        from routing.intent_router import IntentRouter
        r = IntentRouter()
        labels = [Domain.TECHNICAL, Domain.BILLING] * 10
        decisions = [
            RoutingDecision(primary_domain=l, primary_score=0.90,
                            all_scores=[DomainScore(domain=l, score=0.90, rank=1)])
            for l in labels
        ]
        assert 0.0 <= r.compute_ece(decisions, labels) <= 0.15

    def test_escalation_fast_path_fires_on_strong_signal(self):
        from routing.intent_router import IntentRouter
        router = IntentRouter(temperature=0.05)
        # Flat prototype embeddings — escalation only wins via keyword
        flat = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        router._prototype_embeddings = {
            Domain.TECHNICAL:  flat.copy(), Domain.BILLING: flat.copy(),
            Domain.RETURNS:    flat.copy(), Domain.ESCALATION: flat.copy(),
        }
        router._is_fitted = True
        router._encode = lambda texts, **kw: np.tile(flat, (len(texts), 1)).astype(np.float32)
        d = router.route("I am filing a chargeback and taking legal action NOW.")
        assert d.escalation_detected is True


# ══════════════════════════════════════════════════════════════════════════════
# LoRASelector tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLoRASelector:
    def test_escalation_override(self):
        sel = _make_selector()
        d = _decision(Domain.TECHNICAL, 0.88, escalation_detected=True, escalation_score=0.82)
        assert sel.select(d).domain == Domain.ESCALATION

    def test_escalation_override_even_high_confidence(self):
        sel = _make_selector()
        d = _decision(Domain.TECHNICAL, 0.99, escalation_detected=True, escalation_score=0.75)
        assert sel.select(d).domain == Domain.ESCALATION

    def test_confident_single_domain(self):
        sel = _make_selector(comp_thresh=0.05)
        d = _decision(Domain.BILLING, 0.88)
        s = sel.select(d)
        assert s.domain == Domain.BILLING
        assert s.fallback_used is False

    def test_very_low_confidence_escalates(self):
        sel = _make_selector()
        d = _decision(Domain.TECHNICAL, score=GLOBAL_LOW_CONFIDENCE_ESCALATION_THRESHOLD - 0.05)
        s = sel.select(d)
        assert s.domain == Domain.ESCALATION
        assert s.fallback_used is True

    def test_below_threshold_sets_fallback(self):
        sel = _make_selector()
        d = _decision(Domain.TECHNICAL, score=0.55)
        assert sel.select(d).fallback_used is True

    def test_composition_on_close_scores(self):
        sel = _make_selector(comp_thresh=0.20)
        scores = [
            DomainScore(domain=Domain.BILLING,   score=0.55, rank=1),
            DomainScore(domain=Domain.TECHNICAL, score=0.40, rank=2),
            DomainScore(domain=Domain.RETURNS,   score=0.03, rank=3),
            DomainScore(domain=Domain.ESCALATION,score=0.02, rank=4),
        ]
        d = RoutingDecision(primary_domain=Domain.BILLING, primary_score=0.55,
                            all_scores=scores, is_confident=True)
        s = sel.select(d)
        if s.composition_domains:
            assert len(s.composition_domains) == 2
            assert sum(s.composition_weights) == pytest.approx(1.0, abs=0.01)

    def test_no_composition_on_wide_gap(self):
        sel = _make_selector(comp_thresh=0.10)
        d = _decision(Domain.RETURNS, score=0.90)
        assert sel.select(d).composition_domains == []

    def test_select_with_escalation_check_true(self):
        sel = _make_selector()
        d = _decision(Domain.ESCALATION, 0.92, True, 0.92)
        selection, should_esc = sel.select_with_escalation_check(d)
        assert selection.domain == Domain.ESCALATION
        assert should_esc is True

    def test_select_with_escalation_check_false(self):
        sel = _make_selector()
        d = _decision(Domain.TECHNICAL, 0.88, False, 0.05)
        selection, should_esc = sel.select_with_escalation_check(d)
        assert selection.domain == Domain.TECHNICAL
        assert should_esc is False

    def test_registry_from_checkpoint_manager(self):
        manager = MagicMock()
        manager.best_across_all_domains.return_value = {
            "technical": MagicMock(adapter_path="outputs/checkpoints/technical_best"),
            "billing":   MagicMock(adapter_path="outputs/checkpoints/billing_best"),
        }
        registry = AdapterRegistry.from_checkpoint_manager(manager)
        assert Domain.TECHNICAL in registry.paths

    def test_update_registry_hot(self):
        sel = _make_selector()
        new_reg = AdapterRegistry(paths={Domain.TECHNICAL: "new/path"}, ranks={Domain.TECHNICAL: 64})
        sel.update_adapter_registry(new_reg)
        assert sel.registry.paths[Domain.TECHNICAL] == "new/path"

    def test_adapter_path_contains_domain(self):
        sel = _make_selector()
        d = _decision(Domain.RETURNS, 0.85)
        s = sel.select(d)
        assert "returns" in s.adapter_path.lower()

    @pytest.mark.parametrize("domain,rank", [
        (Domain.TECHNICAL, 32), (Domain.BILLING, 24),
        (Domain.RETURNS, 28),   (Domain.ESCALATION, 8),
    ])
    def test_lora_rank_per_domain(self, domain, rank):
        sel = _make_selector()
        d = _decision(domain, 0.85, escalation_detected=(domain == Domain.ESCALATION),
                      escalation_score=0.85 if domain == Domain.ESCALATION else 0.05)
        s = sel.select(d)
        if s.domain == domain:
            assert s.lora_rank == rank


# ══════════════════════════════════════════════════════════════════════════════
# LoRAComposer tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLoRAComposer:
    def test_linear_blend_equal_weights(self):
        import torch
        keys = ["lora_A.weight", "lora_B.weight"]
        sd1 = _state_dict(keys, 4, 0)
        sd2 = _state_dict(keys, 4, 1)
        blended = LoRAComposer().linear_blend([sd1, sd2], [0.5, 0.5])
        for k in keys:
            torch.testing.assert_close(blended[k], (sd1[k].float() + sd2[k].float()) / 2, atol=1e-5, rtol=0)

    def test_linear_blend_normalises_weights(self):
        import torch
        keys = ["lora_A.weight"]
        sd1 = _state_dict(keys, 4, 0)
        sd2 = _state_dict(keys, 4, 1)
        b1 = LoRAComposer().linear_blend([sd1, sd2], [2.0, 2.0])
        b2 = LoRAComposer().linear_blend([sd1, sd2], [0.5, 0.5])
        torch.testing.assert_close(b1["lora_A.weight"], b2["lora_A.weight"], atol=1e-5, rtol=0)

    def test_linear_blend_weight_one_returns_first(self):
        import torch
        keys = ["lora_A.weight"]
        sd1 = _state_dict(keys, 4, 0)
        sd2 = _state_dict(keys, 4, 1)
        blended = LoRAComposer().linear_blend([sd1, sd2], [1.0, 0.0])
        torch.testing.assert_close(blended["lora_A.weight"].float(), sd1["lora_A.weight"].float(), atol=1e-5, rtol=0)

    def test_linear_blend_length_mismatch_raises(self):
        keys = ["lora_A.weight"]
        with pytest.raises(ValueError, match="must match"):
            LoRAComposer().linear_blend([_state_dict(keys), _state_dict(keys)], [0.5])

    def test_task_arithmetic_add(self):
        import torch
        keys = ["lora_A.weight"]
        base = _state_dict(keys, 4, 0)
        add  = _state_dict(keys, 4, 1)
        result = LoRAComposer().task_arithmetic(base, add_dicts=[add], scaling_factor=1.0)
        torch.testing.assert_close(result["lora_A.weight"],
                                   base["lora_A.weight"].float() + add["lora_A.weight"].float(),
                                   atol=1e-5, rtol=0)

    def test_task_arithmetic_subtract(self):
        import torch
        keys = ["lora_A.weight"]
        base = _state_dict(keys, 4, 0)
        sub  = _state_dict(keys, 4, 1)
        result = LoRAComposer().task_arithmetic(base, add_dicts=[], subtract_dicts=[sub], scaling_factor=1.0)
        torch.testing.assert_close(result["lora_A.weight"],
                                   base["lora_A.weight"].float() - sub["lora_A.weight"].float(),
                                   atol=1e-5, rtol=0)

    def test_interference_identical_is_zero(self):
        import torch
        keys = ["lora_A.weight", "lora_B.weight"]
        sd = _state_dict(keys, 8, 42)
        assert LoRAComposer().measure_interference(sd, sd) == pytest.approx(0.0, abs=1e-4)

    def test_interference_orthogonal_is_one(self):
        import torch
        v1 = {"k": torch.tensor([[1.0, 0.0], [0.0, 0.0]])}
        v2 = {"k": torch.tensor([[0.0, 1.0], [0.0, 0.0]])}
        assert LoRAComposer().measure_interference(v1, v2) == pytest.approx(1.0, abs=0.01)

    def test_interference_no_common_keys(self):
        assert LoRAComposer().measure_interference({"a": None}, {"b": None}) == 1.0

    def test_ablation_result_summary(self):
        result = AblationResult(
            primary_only_accuracy=0.912, secondary_only_accuracy=0.874,
            composed_accuracy=0.934, full_model_accuracy=0.950,
            composition_benefit=0.022, cost_reduction_pct=55.0,
            interference_score=0.042, composition_weights=(0.7, 0.3),
            primary_domain=Domain.TECHNICAL, secondary_domain=Domain.BILLING, n_samples=200,
        )
        s = result.summary()
        assert "technical" in s and "billing" in s and "0.934" in s


# ══════════════════════════════════════════════════════════════════════════════
# LRU cache + LoRALoader helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestLRUCache:
    def test_miss_returns_none(self):
        assert _LRUCache(2).get("x") is None

    def test_put_and_get(self):
        c = _LRUCache(2)
        c.put("k", {"v": 1})
        assert c.get("k") == {"v": 1}

    def test_evicts_lru(self):
        c = _LRUCache(2)
        c.put("a", 1); c.put("b", 2)
        c.get("a")       # a is now MRU; b is LRU
        c.put("c", 3)    # evicts b
        assert "b" not in c
        assert "a" in c and "c" in c

    def test_hit_rate(self):
        c = _LRUCache(3)
        c.put("k", "v")
        c.get("k"); c.get("x"); c.get("k")
        assert c._hits == 2 and c._misses == 1
        assert c.hit_rate == pytest.approx(2/3, abs=0.01)

    def test_len(self):
        c = _LRUCache(5)
        c.put("a", 1); c.put("b", 2)
        assert len(c) == 2

    def test_clear(self):
        c = _LRUCache(3)
        c.put("a", 1); c.get("a")
        c.clear()
        assert len(c) == 0 and c._hits == 0

    def test_stats_keys(self):
        c = _LRUCache(4); c.put("x", 1)
        s = c.stats()
        assert {"size","max_size","hits","misses","hit_rate","cached"}.issubset(s)

    def test_overwrite_no_growth(self):
        c = _LRUCache(2)
        c.put("k", "v1"); c.put("k", "v2")
        assert len(c) == 1 and c.get("k") == "v2"


class TestLoRALoaderHelpers:
    def test_not_exists_missing_dir(self, tmp_path):
        assert not adapter_exists(tmp_path / "nope")

    def test_not_exists_no_config(self, tmp_path):
        (tmp_path / "a").mkdir()
        assert not adapter_exists(tmp_path / "a")

    def test_exists_with_config(self, tmp_path):
        d = tmp_path / "a"; d.mkdir()
        (d / "adapter_config.json").write_text('{"r":32}')
        assert adapter_exists(d)

    def test_load_config(self, tmp_path):
        d = tmp_path / "a"; d.mkdir()
        (d / "adapter_config.json").write_text('{"r":32,"lora_alpha":64}')
        cfg = load_adapter_config(d)
        assert cfg["r"] == 32 and cfg["lora_alpha"] == 64

    def test_load_config_missing_empty(self, tmp_path):
        assert load_adapter_config(tmp_path / "nope") == {}

    def test_loader_cache_stats_structure(self):
        s = LoRALoader(cache_size=4).cache_stats()
        assert {"size","hit_rate","cached"}.issubset(s)

    def test_loader_estimated_memory_empty(self):
        assert LoRALoader(cache_size=4).estimated_memory_mb() == 0.0

    def test_loader_clear_cache(self):
        loader = LoRALoader(cache_size=4)
        loader._cache.put("p", {"w": 1})
        loader.clear_cache()
        assert len(loader._cache) == 0 and loader._active_adapter is None
