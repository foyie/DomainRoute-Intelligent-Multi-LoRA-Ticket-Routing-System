"""
serving/inference.py
─────────────────────
End-to-end VeriTune inference pipeline.

Pipeline stages
---------------
  1. IntentRouter.route(ticket)         → RoutingDecision   ~12ms
  2. LoRASelector.select(decision)      → LoRASelection      ~1ms
  3. LoRALoader.load(adapter_path)      → model              ~8ms (cache hit) / ~500ms (miss)
  4. LoRAComposer.compose(model, sel)   → model              ~5ms (if composing)
  5. generate(model, prompt, cfg)       → text              ~105ms
  6. SafetyPipeline.run(...)            → SafetyReport        ~5ms
  7. Build TicketResponse               → response            ~1ms

Total target: < 150ms p95 (cache warm)

Public API
----------
InferencePipeline
  .predict(request)          → TicketResponse
  .predict_batch(requests)   → List[TicketResponse]
  .health_check()            → dict
  .warm_up()                 → None

build_pipeline(cfg)          → InferencePipeline
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    End-to-end VeriTune inference pipeline.

    Holds references to all shared resources (router, selector, loader, etc.)
    that are initialised once at startup and reused across requests.

    Parameters
    ----------
    router          : Fitted IntentRouter
    selector        : LoRASelector with AdapterRegistry
    loader          : LoRALoader with LRU cache
    composer        : LoRAComposer
    safety_pipeline : SafetyPipeline
    base_model      : Loaded base LLM (Mistral-7B or similar)
    tokenizer       : Matching tokenizer
    max_new_tokens  : Generation length cap
    temperature     : Sampling temperature
    """

    def __init__(
        self,
        router,
        selector,
        loader,
        composer,
        safety_pipeline,
        base_model=None,
        tokenizer=None,
        max_new_tokens: int  = 256,
        temperature:    float = 0.3,
        top_p:          float = 0.9,
        device:         str   = "cpu",
    ) -> None:
        self.router          = router
        self.selector        = selector
        self.loader          = loader
        self.composer        = composer
        self.safety_pipeline = safety_pipeline
        self.base_model      = base_model
        self.tokenizer       = tokenizer
        self.max_new_tokens  = max_new_tokens
        self.temperature     = temperature
        self.top_p           = top_p
        self.device          = device

    # ── Main prediction entry point ────────────────────────────────────────────

    def predict(self, request) -> "TicketResponse":
        """
        Run the full inference pipeline for a single ticket request.

        Parameters
        ----------
        request : TicketRequest — validated Pydantic request object

        Returns
        -------
        TicketResponse with response text, routing metadata, and latency breakdown
        """
        from routing.models import (
            TicketResponse, ResolutionStatus, LatencyBreakdown, EscalationEvent,
        )
        from serving.monitoring import RequestTrace, get_metrics

        trace = RequestTrace(
            session_id=request.session_id,
            customer_id=request.customer_id,
            request_start=time.perf_counter(),
        )

        try:
            response = self._run_pipeline(request, trace)
            trace.finalise()
            get_metrics().record(trace)
            return response
        except Exception as e:
            trace.status = "error"
            trace.error_message = str(e)
            trace.finalise()
            get_metrics().record(trace)
            logger.error(
                "Pipeline error [trace=%s]: %s", trace.trace_id, e, exc_info=True
            )
            raise

    def predict_batch(self, requests: List) -> List:
        """Run prediction for a batch of requests sequentially."""
        return [self.predict(req) for req in requests]

    def health_check(self) -> dict:
        """Return component health status."""
        return {
            "status": "ok",
            "router_fitted":   self._check_router(),
            "adapter_cache":   self.loader.cache_stats() if self.loader else {},
            "base_model_loaded": self.base_model is not None,
            "components": {
                "router":   self.router is not None,
                "selector": self.selector is not None,
                "loader":   self.loader is not None,
                "safety":   self.safety_pipeline is not None,
            },
        }

    def warm_up(self) -> None:
        """Warm up the adapter cache by preloading all domain adapters."""
        if self.loader and self.selector:
            self.loader.preload_all(self.selector.registry)
            logger.info("Pipeline warm-up complete")

    # ── Internal pipeline ──────────────────────────────────────────────────────

    def _run_pipeline(self, request, trace: "RequestTrace"):
        from routing.models import (
            TicketResponse, ResolutionStatus, LatencyBreakdown,
            GenerationRequest,
        )
        from training.utils import format_inference_prompt

        ticket = request.ticket_text
        history = list(request.conversation_history)

        # ── Stage 1: Route ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        decision = self.router.route(ticket, history, request)
        trace.router_ms          = (time.perf_counter() - t0) * 1000
        trace.domain             = decision.primary_domain.value
        trace.routing_method     = decision.routing_method.value
        trace.routing_confidence = decision.primary_score
        trace.escalation_detected = decision.escalation_detected
        trace.escalation_score   = decision.escalation_score

        # ── Stage 2: Select LoRA ───────────────────────────────────────────────
        selection, should_escalate_immediately = (
            self.selector.select_with_escalation_check(decision)
        )
        trace.lora_rank = selection.lora_rank

        # ── Stage 3: Load adapter ──────────────────────────────────────────────
        t1 = time.perf_counter()
        model = self._load_adapter(selection)
        trace.lora_load_ms = (time.perf_counter() - t1) * 1000
        trace.cache_hit    = self.loader._cache.hit_rate > 0 if self.loader else False

        # ── Stage 4: Compose (if multi-LoRA) ──────────────────────────────────
        if selection.composition_domains and len(selection.composition_domains) > 1:
            model = self._compose_adapters(model, selection)

        # ── Stage 5: Generate ──────────────────────────────────────────────────
        t2 = time.perf_counter()
        prompt   = format_inference_prompt(ticket, decision.primary_domain.value)
        gen_text = self._generate(model, prompt)
        trace.generation_ms    = (time.perf_counter() - t2) * 1000
        trace.response_length  = len(gen_text)

        # ── Stage 6: Safety filters ────────────────────────────────────────────
        t3 = time.perf_counter()
        case_id = f"ESC-{uuid.uuid4().hex[:6].upper()}" if should_escalate_immediately else None
        safety_report = self.safety_pipeline.run(
            ticket=ticket,
            response=gen_text,
            domain=decision.primary_domain.value,
            escalation_score=decision.escalation_score,
            case_id=case_id,
        )
        trace.safety_ms            = (time.perf_counter() - t3) * 1000
        trace.hallucination_flagged = safety_report.hallucination_flagged

        # ── Stage 7: Build response ────────────────────────────────────────────
        from evaluation.metrics import DEFAULT_COST_MAP
        from routing.models import Domain
        domain_enum = decision.primary_domain
        trace.cost_dollars = DEFAULT_COST_MAP.get(domain_enum, 0.10)

        resolution_status = (
            ResolutionStatus.ESCALATED
            if safety_report.escalation_triggered
            else ResolutionStatus.RESOLVED
        )

        latency = LatencyBreakdown(
            router_ms=round(trace.router_ms, 1),
            lora_load_ms=round(trace.lora_load_ms, 1),
            generation_ms=round(trace.generation_ms, 1),
            safety_ms=round(trace.safety_ms, 1),
        )
        latency.compute_total()

        from routing.models import TicketResponse
        return TicketResponse(
            response_text=safety_report.final_response,
            domain=decision.primary_domain,
            resolution_status=resolution_status,
            routing_decision=decision,
            lora_selection=selection,
            latency=latency,
            escalation_detected=safety_report.escalation_triggered,
            escalation_reason=(
                next((f.reason for f in safety_report.filter_results
                      if f.filter_name == "EscalationGuard" and not f.passed), None)
            ),
            case_id=safety_report.case_id,
            session_id=request.session_id,
            hallucination_score=float(
                next((f.confidence for f in safety_report.filter_results
                      if f.filter_name == "HallucinationGuard"), 0.0)
            ),
            quality_passed=safety_report.quality_passed,
        )

    def _load_adapter(self, selection):
        """Load the LoRA adapter via the loader's LRU cache."""
        if self.loader is None or self.base_model is None:
            logger.warning("No loader or base model — returning base model as-is")
            return self.base_model

        try:
            return self.loader.load(selection.adapter_path)
        except FileNotFoundError:
            logger.warning(
                "Adapter not found at %s — using base model", selection.adapter_path
            )
            return self.base_model

    def _compose_adapters(self, model, selection):
        """Apply multi-LoRA composition if requested."""
        if self.composer is None:
            return model
        try:
            return self.composer.compose(model, selection)
        except Exception as e:
            logger.warning("LoRA composition failed: %s — using primary adapter", e)
            return model

    def _generate(self, model, prompt: str) -> str:
        """
        Run inference on the loaded model.

        In production this calls the model's .generate() method.
        Falls back to a domain-appropriate canned response if no model is loaded.
        """
        if model is None:
            return self._fallback_response(prompt)

        try:
            import torch
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # Decode only the newly generated tokens
            n_prompt_tokens = enc["input_ids"].shape[1]
            new_tokens = output_ids[0][n_prompt_tokens:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        except Exception as e:
            logger.error("Generation failed: %s", e, exc_info=True)
            return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        """Canned fallback when model is unavailable."""
        return (
            "Thank you for contacting us. I'm looking into your issue now and "
            "will help you resolve it as quickly as possible. Please allow me a "
            "moment to review the details of your request."
        )

    def _check_router(self) -> bool:
        return self.router is not None and getattr(self.router, "_is_fitted", False)


# ── Pipeline factory ───────────────────────────────────────────────────────────

def build_pipeline(
    router_path:   Optional[str] = None,
    registry_path: Optional[str] = None,
    base_model=None,
    tokenizer=None,
    device: str = "cpu",
    cache_size: int = 4,
) -> InferencePipeline:
    """
    Build and return a fully initialised InferencePipeline.

    Loads the router from disk if router_path is provided.
    Falls back to an unfitted router if the path doesn't exist.

    Parameters
    ----------
    router_path   : Path to saved router state (outputs/router/)
    registry_path : Path to checkpoint_registry.json
    base_model    : Pre-loaded base LLM (optional — if None, uses fallback responses)
    tokenizer     : Pre-loaded tokenizer
    device        : Compute device
    cache_size    : LoRA adapter LRU cache size
    """
    from routing.intent_router import IntentRouter
    from routing.lora_selector import LoRASelector, AdapterRegistry
    from routing.lora_loader   import LoRALoader
    from routing.lora_composer import LoRAComposer
    from serving.safety_filters import SafetyPipeline
    from training.checkpoint_manager import CheckpointManager

    # Router
    if router_path and Path(router_path).exists():
        router = IntentRouter.load(router_path)
        logger.info("Router loaded from %s", router_path)
    else:
        router = IntentRouter()
        logger.warning("Router not found at %s — using unfitted router", router_path)

    # Adapter registry
    if registry_path and Path(registry_path).exists():
        manager = CheckpointManager()
        manager.load_registry(registry_path)
        registry = AdapterRegistry.from_checkpoint_manager(manager)
    else:
        registry = AdapterRegistry.default()

    selector = LoRASelector(registry=registry)
    loader   = LoRALoader(
        base_model=base_model,
        tokenizer=tokenizer,
        cache_size=cache_size,
        device=device,
    )
    composer        = LoRAComposer(base_model=base_model, device=device)
    safety_pipeline = SafetyPipeline()

    pipeline = InferencePipeline(
        router=router,
        selector=selector,
        loader=loader,
        composer=composer,
        safety_pipeline=safety_pipeline,
        base_model=base_model,
        tokenizer=tokenizer,
        device=device,
    )

    logger.info(
        "InferencePipeline built: router_fitted=%s cache_size=%d",
        pipeline._check_router(), cache_size,
    )
    return pipeline
