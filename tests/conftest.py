"""
tests/conftest.py
──────────────────
Shared pytest fixtures for VeriTune tests.

Fixtures
--------
sample_tickets          – list of raw ticket dicts (all domains)
technical_tickets       – list of technical domain dicts
billing_tickets         – list of billing domain dicts
returns_tickets         – list of returns domain dicts
escalation_tickets      – list of escalation domain dicts
sample_dataset          – HuggingFace Dataset from sample_tickets
sample_dataset_noisy    – Dataset with 10% injected label noise
cfg                     – loaded domains.yaml config dict
svm_classifier          – fitted SVMClassifier on sample_dataset
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Raw sample data ────────────────────────────────────────────────────────────

SAMPLE_TICKETS_BY_DOMAIN = {
    "technical": [
        "My headphones won't connect to Bluetooth after the firmware update.",
        "The app crashes every time I try to open the equaliser settings.",
        "Battery drains from 100% to 20% in under two hours.",
        "Left earbud has no sound after I cleaned it with a damp cloth.",
        "ERR_0042 error appears whenever I try to sync the device.",
        "Touch controls stopped responding completely after I dropped the device.",
        "Noise cancellation stopped working after the v2.4 update.",
        "Device not recognised when plugged into my Windows 11 laptop.",
        "Bluetooth disconnects from my phone every five minutes.",
        "Screen flicker every time the device wakes from sleep mode.",
        "Firmware update failed halfway and device won't power on anymore.",
        "Mic is not picking up my voice during video calls.",
    ],
    "billing": [
        "I was charged twice for my monthly subscription last week.",
        "I want to cancel my Pro plan before the next billing date.",
        "My promo code SAVE20 didn't apply at checkout.",
        "Can I get a VAT invoice for my $49.99 payment on the 15th?",
        "I'd like to upgrade from Basic to Enterprise plan.",
        "When does my free trial end and what will I be charged after?",
        "My credit card was declined but the charge still went through.",
        "I didn't authorise a $29.99 charge that appeared on my account.",
        "Invoice for order #384921 shows $49.99 but I was quoted $39.99.",
        "I need to update my payment method to a new Visa card.",
        "I was billed for an annual plan but I signed up for monthly.",
        "Student discount is not showing up on my account.",
    ],
    "returns": [
        "I received the wrong colour headphones in order #773401.",
        "My speaker arrived damaged with a cracked corner.",
        "I need to exchange my medium jacket for a large please.",
        "How do I return an item I bought two weeks ago online?",
        "I lost my return label — can you send me a new one?",
        "My order says delivered but I never received the package.",
        "The headphones stopped working after three days, I want a replacement.",
        "Can I return a gift I received without a receipt?",
        "I returned the item two weeks ago but haven't received my refund.",
        "There are scratches on the device that weren't there when I sent it back.",
        "I was sent a refurbished product but paid for a new one.",
        "The box arrived open with items missing.",
    ],
    "escalation": [
        "This is absolutely unacceptable. I've been waiting three weeks.",
        "I'm filing a chargeback if this isn't fixed in 24 hours.",
        "I've contacted you five times about order #920183 and nothing has changed.",
        "I'm posting one-star reviews everywhere unless you refund me NOW.",
        "I want to speak to a manager immediately. This is disgraceful.",
        "I'm a lawyer and will take legal action if this isn't resolved by Friday.",
        "This is fraud. I'm reporting you to the FTC.",
        "I've been a loyal customer for ten years and this is how you treat me?",
        "One more ignored message and I'm cancelling everything.",
        "I've spent four hours on hold and nobody has helped.",
        "Your service is completely useless. Fix this NOW.",
        "My elderly mother has been without her device for a week.",
    ],
}

SAMPLE_RESPONSES = {
    "resolved": [
        "Thank you for reaching out. I've looked into your account and can see the issue. "
        "Please try the following steps: first, restart the device by holding the power button "
        "for 10 seconds. If that doesn't help, I can arrange a replacement for you.",
        "I'm sorry to hear you're experiencing this issue. I've processed your request and "
        "you'll receive a confirmation email within the next few minutes.",
    ],
    "escalate": [
        "I sincerely apologise for the experience you've had. I'm escalating this to a "
        "senior specialist who will contact you within the hour. Your case ID is #ESC-40291.",
    ],
}


@pytest.fixture(scope="session")
def cfg():
    """Loaded domains.yaml config."""
    from data.loaders import load_config
    return load_config(ROOT / "config" / "domains.yaml")


@pytest.fixture(scope="session")
def sample_tickets() -> list[dict]:
    """Flat list of all sample ticket dicts across all domains."""
    tickets = []
    for domain, texts in SAMPLE_TICKETS_BY_DOMAIN.items():
        label = "escalate" if domain == "escalation" else "resolved"
        for text in texts:
            tickets.append({
                "text":     text,
                "domain":   domain,
                "label":    label,
                "source":   "fixture",
                "response": random.choice(SAMPLE_RESPONSES[label]),
            })
    return tickets


@pytest.fixture(scope="session")
def technical_tickets() -> list[dict]:
    return [
        {"text": t, "domain": "technical", "label": "resolved", "source": "fixture"}
        for t in SAMPLE_TICKETS_BY_DOMAIN["technical"]
    ]


@pytest.fixture(scope="session")
def billing_tickets() -> list[dict]:
    return [
        {"text": t, "domain": "billing", "label": "resolved", "source": "fixture"}
        for t in SAMPLE_TICKETS_BY_DOMAIN["billing"]
    ]


@pytest.fixture(scope="session")
def returns_tickets() -> list[dict]:
    return [
        {"text": t, "domain": "returns", "label": "resolved", "source": "fixture"}
        for t in SAMPLE_TICKETS_BY_DOMAIN["returns"]
    ]


@pytest.fixture(scope="session")
def escalation_tickets() -> list[dict]:
    return [
        {"text": t, "domain": "escalation", "label": "escalate", "source": "fixture"}
        for t in SAMPLE_TICKETS_BY_DOMAIN["escalation"]
    ]


@pytest.fixture(scope="session")
def sample_dataset(sample_tickets) -> Dataset:
    """HuggingFace Dataset from all sample tickets."""
    return Dataset.from_list(sample_tickets)


@pytest.fixture(scope="session")
def sample_dataset_noisy(sample_tickets) -> Dataset:
    """
    Dataset with ~10% randomly flipped labels to simulate label noise.
    Used for testing quality gate detection sensitivity.
    """
    random.seed(99)
    noisy_tickets = []
    noise_rate = 0.10
    domain_labels = {"technical": "resolved", "billing": "resolved",
                     "returns": "resolved", "escalation": "escalate"}

    for ticket in sample_tickets:
        t = dict(ticket)
        if random.random() < noise_rate:
            # Flip label
            t["label"] = "escalate" if t["label"] == "resolved" else "resolved"
            t["_noisy"] = True
        else:
            t["_noisy"] = False
        noisy_tickets.append(t)

    return Dataset.from_list(noisy_tickets)


@pytest.fixture(scope="session")
def svm_classifier(sample_dataset):
    """Fitted SVMClassifier on the sample dataset (domain classification)."""
    from data.intent_classifier import SVMClassifier
    clf = SVMClassifier()
    clf.fit(sample_dataset["text"], sample_dataset["domain"])
    return clf


@pytest.fixture(scope="session")
def sample_embeddings(sample_dataset) -> np.ndarray:
    """
    Pre-computed sentence embeddings for sample_dataset.
    Uses a tiny random embedding to avoid loading the full SBERT model in tests.
    Replace with real SBERT embeddings for integration tests.
    """
    np.random.seed(42)
    # Simulate 384-dim embeddings (all-MiniLM-L6-v2 output size)
    n = len(sample_dataset)
    raw = np.random.randn(n, 384).astype(np.float32)
    # Normalise rows
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


@pytest.fixture()
def tmp_jsonl(tmp_path) -> Path:
    """Create a temporary JSONL file with sample data and return its path."""
    path = tmp_path / "sample.jsonl"
    records = [
        {"text": "My headphones won't charge.", "domain": "technical", "label": "resolved"},
        {"text": "I was charged twice.",         "domain": "billing",   "label": "resolved"},
        {"text": "Wrong item received.",          "domain": "returns",   "label": "resolved"},
        {"text": "This is unacceptable NOW!",     "domain": "escalation","label": "escalate"},
    ]
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


@pytest.fixture()
def tmp_csv(tmp_path) -> Path:
    """Create a temporary CSV file and return its path."""
    import pandas as pd
    path = tmp_path / "sample.csv"
    data = {
        "text":   ["Bluetooth keeps dropping", "I need a refund", "Item damaged"],
        "domain": ["technical",                 "billing",         "returns"],
        "label":  ["resolved",                  "resolved",        "resolved"],
    }
    pd.DataFrame(data).to_csv(path, index=False)
    return path
