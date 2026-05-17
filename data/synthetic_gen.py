"""
data/synthetic_gen.py
──────────────────────
Two-stage synthetic data generation for low-resource domains:

  Stage 1 – Template augmentation
      Fill slot templates with domain-specific entities, numbers, and phrases.
      Fast, zero cost, infinite supply, but lower diversity.

  Stage 2 – LLM augmentation (GPT-4o-mini via instructor)
      Generate structurally diverse tickets from seed examples.
      Higher diversity, small cost, rate-limited.

Public API
----------
generate_template_examples(domain, n, cfg)  → List[dict]
generate_llm_examples(domain, n, cfg)       → List[dict]
augment_domain(domain, n, cfg, use_llm)     → List[dict]
save_synthetic(examples, path)              → None
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


# ── Domain slot templates ──────────────────────────────────────────────────────

TEMPLATES: dict[str, list[str]] = {
    "technical": [
        "My {product} won't {action} after the {version} update.",
        "The {product} {component} keeps {failure_mode} randomly.",
        "I've tried {fix_attempt} but the {product} still {failure_mode}.",
        "{product} shows a '{error_code}' error whenever I try to {action}.",
        "Since the firmware update, my {product}'s {component} has been {failure_mode}.",
        "I can't get my {product} to {action} on {os_version}.",
        "The {product} app crashes every time I open the {feature} screen.",
        "My {product} disconnects from {connection} every {duration}.",
        "After factory reset, {product} still won't {action}.",
        "Is there a fix for the {product} {component} {failure_mode} issue?",
    ],
    "billing": [
        "I was charged {amount} on {date} but I didn't authorise this.",
        "I want to cancel my {plan} subscription and get a refund.",
        "My invoice for {date} shows {amount} but it should be {lower_amount}.",
        "I need to upgrade from {plan_from} to {plan_to} plan.",
        "My {payment_method} was declined but the charge still went through.",
        "Can I get a receipt for my {date} payment of {amount}?",
        "I was billed twice for the same order #{order_id}.",
        "My promo code {code} didn't apply at checkout.",
        "I'd like to pause my subscription for {duration}.",
        "When does my free trial end and how much will I be charged?",
    ],
    "returns": [
        "I received the wrong {product} in my order #{order_id}.",
        "My {product} arrived damaged — there is {damage_description}.",
        "I need to exchange my {product} for a {alternative} size.",
        "How do I return a {product} I bought {duration} ago?",
        "I lost my return label for order #{order_id}.",
        "My order #{order_id} says delivered but I never received it.",
        "The {product} stopped working after {duration}. I want a replacement.",
        "Can I return a gift without a receipt?",
        "How long does it take to process a return for order #{order_id}?",
        "I returned the {product} {duration} ago but haven't got my refund.",
    ],
    "escalation": [
        "This is absolutely unacceptable. I've been waiting {duration} and nobody has helped.",
        "I'm going to file a chargeback if this isn't resolved in {duration}.",
        "I've contacted support {n_contacts} times about order #{order_id}. Fix this NOW.",
        "I'm posting a 1-star review everywhere unless you refund my {amount} immediately.",
        "Your service is a disgrace. I want to speak to a manager RIGHT NOW.",
        "I'm a lawyer and I will be taking legal action if this isn't resolved by {date}.",
        "This is fraud. I'm reporting you to {authority} and my bank.",
        "I've been a loyal customer for {duration} and this is how you treat me?",
        "One more ignored message and I'm cancelling everything and disputing every charge.",
        "Completely useless. I've spent {duration} on the phone and nothing is resolved.",
    ],
}

SLOTS: dict[str, dict[str, list]] = {
    "product": [
        "wireless headphones", "Bluetooth speaker", "smart watch", "laptop",
        "tablet", "earbuds", "webcam", "keyboard", "mouse", "router",
    ],
    "component": [
        "battery", "screen", "charging port", "microphone", "speaker",
        "touch controls", "camera", "USB port", "power button",
    ],
    "action": [
        "connect to Bluetooth", "charge", "turn on", "sync", "update",
        "pair with my phone", "play audio", "power off", "recognise my computer",
    ],
    "failure_mode": [
        "disconnecting", "freezing", "crashing", "draining quickly",
        "not responding", "overheating", "making noise", "restarting randomly",
    ],
    "version": ["v2.4", "v3.1", "v1.8.2", "v4.0", "the latest", "v2.9.1"],
    "fix_attempt": [
        "restarting it", "a factory reset", "reinstalling the app",
        "updating the firmware", "clearing the cache",
    ],
    "error_code": [
        "ERR_0042", "SYNC_FAIL", "AUTH_ERROR", "CONN_TIMEOUT", "DEVICE_NOT_FOUND",
        "UPDATE_FAILED", "0x80070002",
    ],
    "os_version": ["iOS 17", "Android 14", "Windows 11", "macOS Ventura", "Ubuntu 22.04"],
    "connection": ["Bluetooth", "Wi-Fi", "USB", "my phone", "my laptop"],
    "feature": ["settings", "playlist", "equaliser", "notifications", "account"],
    "amount": ["$29.99", "$49.99", "$9.99", "$14.99", "$99.99", "$199.99"],
    "lower_amount": ["$24.99", "$39.99", "$7.99", "$12.99"],
    "date": ["last Tuesday", "3 days ago", "on the 15th", "yesterday", "last month"],
    "plan": ["Basic", "Pro", "Enterprise", "Student", "Family"],
    "plan_from": ["Basic", "Starter", "Free"],
    "plan_to": ["Pro", "Enterprise", "Business"],
    "payment_method": ["Visa ending in 4242", "Mastercard", "PayPal", "Apple Pay"],
    "order_id": ["#384921", "#773401", "#920183", "#651023", "#489302"],
    "code": ["SAVE20", "WELCOME10", "PROMO50", "SUMMER25"],
    "alternative": ["small", "medium", "large", "XL", "different colour"],
    "damage_description": [
        "a cracked screen", "a dented corner", "missing components",
        "a broken hinge", "scratches on the surface",
    ],
    "n_contacts": ["3", "4", "5", "6", "seven"],
    "authority": ["the BBB", "consumer protection", "the FTC", "trading standards"],
    "duration": [
        "two weeks", "three days", "a month", "48 hours", "over a week",
        "ten days", "more than two weeks",
    ],
}


def _fill_template(template: str) -> str:
    """Replace {slot} placeholders with random values from SLOTS."""
    import re
    slots_needed = re.findall(r"\{(\w+)\}", template)
    result = template
    for slot in slots_needed:
        if slot in SLOTS:
            result = result.replace(f"{{{slot}}}", random.choice(SLOTS[slot]), 1)
    return result


# ── Template generation ────────────────────────────────────────────────────────

def generate_template_examples(
    domain: str,
    n: int,
    cfg: Optional[dict] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Generate `n` synthetic examples for `domain` using slot-filled templates.

    Returns
    -------
    List of dicts: {text, domain, label, source}
    """
    random.seed(seed)
    templates = TEMPLATES.get(domain)
    if not templates:
        raise ValueError(f"Unknown domain '{domain}'. Valid: {list(TEMPLATES)}")

    label = "escalate" if domain == "escalation" else "resolved"
    examples = []

    for i in range(n):
        template = templates[i % len(templates)]
        text = _fill_template(template)
        examples.append({
            "text": text,
            "domain": domain,
            "label": label,
            "source": "template",
            "response": "",   # populated during training data prep
        })

    logger.info("Generated %d template examples for domain='%s'", n, domain)
    return examples


# ── LLM generation ─────────────────────────────────────────────────────────────

def generate_llm_examples(
    domain: str,
    n: int,
    cfg: Optional[dict] = None,
    seed_examples: Optional[List[dict]] = None,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    batch_size: int = 10,
) -> List[dict]:
    """
    Generate `n` diverse synthetic tickets using an LLM (GPT-4o-mini).
    Uses instructor for structured output and automatic retries.

    Requires OPENAI_API_KEY in environment.
    Falls back to template generation if the API is unavailable.

    Returns
    -------
    List of dicts: {text, domain, label, source}
    """
    try:
        import instructor
        from openai import OpenAI
        from pydantic import BaseModel, Field

        class TicketBatch(BaseModel):
            tickets: List[str] = Field(
                description="List of realistic customer support ticket texts"
            )

    except ImportError:
        logger.warning(
            "instructor or openai not installed. Falling back to template generation."
        )
        return generate_template_examples(domain, n, cfg, seed)

    if cfg is None:
        from data.loaders import load_config
        cfg = load_config()

    domain_cfg = cfg["domains"].get(domain, {})
    intent_examples = domain_cfg.get("intent_examples", [])[:5]
    seed_text = "\n".join(f"- {ex}" for ex in intent_examples)

    label = "escalate" if domain == "escalation" else "resolved"
    examples: List[dict] = []
    generated = 0

    client = instructor.from_openai(OpenAI())

    while generated < n:
        batch_n = min(batch_size, n - generated)
        prompt = (
            f"Generate {batch_n} realistic, diverse customer support ticket messages "
            f"for the '{domain}' domain.\n\n"
            f"Domain description: {domain_cfg.get('description', '')}\n\n"
            f"Example tickets for this domain:\n{seed_text}\n\n"
            f"Requirements:\n"
            f"- Each ticket should sound like a real customer wrote it\n"
            f"- Vary the tone (frustrated, polite, confused, urgent)\n"
            f"- Vary the issue type within the domain\n"
            f"- Keep tickets 15–150 words\n"
            f"- Do NOT duplicate the examples above\n"
            f"- Return exactly {batch_n} tickets\n"
        )

        for attempt in range(max_retries):
            try:
                result = client.chat.completions.create(
                    model=model,
                    response_model=TicketBatch,
                    messages=[{"role": "user", "content": prompt}],
                    max_retries=2,
                )
                for ticket_text in result.tickets:
                    examples.append({
                        "text": ticket_text.strip(),
                        "domain": domain,
                        "label": label,
                        "source": "llm",
                        "response": "",
                    })
                generated += len(result.tickets)
                time.sleep(0.5)   # rate-limit courtesy pause
                break
            except Exception as e:
                logger.warning(
                    "LLM generation attempt %d/%d failed: %s",
                    attempt + 1, max_retries, e,
                )
                if attempt == max_retries - 1:
                    logger.error(
                        "LLM generation failed after %d attempts. "
                        "Falling back to templates for remaining %d examples.",
                        max_retries, n - generated,
                    )
                    examples.extend(
                        generate_template_examples(domain, n - generated, cfg, seed=99)
                    )
                    generated = n
                time.sleep(2 ** attempt)

    logger.info("Generated %d LLM examples for domain='%s'", len(examples), domain)
    return examples[:n]


# ── Combined augmentation ──────────────────────────────────────────────────────

def augment_domain(
    domain: str,
    n: int,
    cfg: Optional[dict] = None,
    use_llm: bool = False,
    llm_fraction: float = 0.50,
    seed: int = 42,
) -> List[dict]:
    """
    Generate `n` synthetic examples using a mix of template and LLM generation.

    Parameters
    ----------
    domain       : One of technical / billing / returns / escalation
    n            : Total examples to generate
    cfg          : Loaded domains.yaml config
    use_llm      : If True, generate llm_fraction of examples via LLM
    llm_fraction : Fraction of examples to generate via LLM (default 50%)

    Returns
    -------
    Shuffled list of {text, domain, label, source} dicts
    """
    if cfg is None:
        from data.loaders import load_config
        cfg = load_config()

    if use_llm:
        n_llm      = int(n * llm_fraction)
        n_template = n - n_llm
        llm_examples      = generate_llm_examples(domain, n_llm, cfg)
        template_examples = generate_template_examples(domain, n_template, cfg, seed)
        all_examples = llm_examples + template_examples
    else:
        all_examples = generate_template_examples(domain, n, cfg, seed)

    random.shuffle(all_examples)
    logger.info(
        "Augmented domain='%s': %d examples (template=%d, llm=%d)",
        domain,
        len(all_examples),
        sum(1 for e in all_examples if e["source"] == "template"),
        sum(1 for e in all_examples if e["source"] == "llm"),
    )
    return all_examples


# ── Persistence ────────────────────────────────────────────────────────────────

def save_synthetic(
    examples: List[dict],
    path: str | Path,
    append: bool = False,
) -> None:
    """
    Save synthetic examples to a JSONL file.

    Parameters
    ----------
    examples : List of example dicts
    path     : Output path (.jsonl)
    append   : If True, append to existing file; otherwise overwrite
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    with open(path, mode) as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(
        "%s %d synthetic examples to %s",
        "Appended" if append else "Saved", len(examples), path,
    )


def load_synthetic(path: str | Path) -> List[dict]:
    """Load previously saved synthetic examples from a JSONL file."""
    path = Path(path)
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples
