"""
data/intent_classifier.py
──────────────────────────
Baseline intent classifier used during data preparation and as a fast
pre-filter before the SBERT-based semantic router.

Two classifiers are provided:
  1. SVMClassifier      – TF-IDF + LinearSVC (fast, no GPU, strong baseline)
  2. EmbeddingClassifier – SBERT embeddings + cosine nearest-neighbour
                           (higher accuracy, used in production routing)

Both implement a common ClassifierProtocol:
  .fit(texts, labels)   → self
  .predict(texts)       → List[str]
  .predict_proba(texts) → np.ndarray   (N, n_classes)

Public API
----------
SVMClassifier
EmbeddingClassifier
train_baseline_classifier(dataset, cfg)  → SVMClassifier
evaluate_classifier(clf, dataset)        → ClassificationReport
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DOMAINS = ["technical", "billing", "returns", "escalation"]


# ── Protocol ───────────────────────────────────────────────────────────────────

class ClassifierProtocol(Protocol):
    def fit(self, texts: List[str], labels: List[str]) -> "ClassifierProtocol": ...
    def predict(self, texts: List[str]) -> List[str]: ...
    def predict_proba(self, texts: List[str]) -> np.ndarray: ...


# ── Report dataclass ───────────────────────────────────────────────────────────

@dataclass
class ClassificationReport:
    accuracy: float
    per_class: dict          # {class_name: {precision, recall, f1, support}}
    confusion_matrix: np.ndarray
    domains: List[str]

    def summary(self) -> str:
        lines = [f"Accuracy: {self.accuracy:.3f}"]
        for cls, m in self.per_class.items():
            lines.append(
                f"  {cls:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                f"F1={m['f1']:.3f}  n={m['support']}"
            )
        return "\n".join(lines)


# ── SVM Classifier ─────────────────────────────────────────────────────────────

class SVMClassifier:
    """
    TF-IDF feature extraction + LinearSVC.
    ~97% accuracy on clean domain-labelled ticket data.
    Inference: ~0.5ms per ticket on CPU.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0,
    ) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline

        self.labels_: Optional[List[str]] = None
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
            )),
            ("svm", CalibratedClassifierCV(
                LinearSVC(C=C, max_iter=2000, class_weight="balanced"),
                cv=3,
            )),
        ])

    def fit(self, texts: List[str], labels: List[str]) -> "SVMClassifier":
        self.labels_ = sorted(set(labels))
        self.pipeline.fit(texts, labels)
        logger.info(
            "SVMClassifier trained on %d examples, %d classes: %s",
            len(texts), len(self.labels_), self.labels_,
        )
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns (N, n_classes) probability matrix."""
        return self.pipeline.predict_proba(texts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("SVMClassifier saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SVMClassifier":
        with open(path, "rb") as f:
            clf = pickle.load(f)
        logger.info("SVMClassifier loaded from %s", path)
        return clf


# ── Embedding Classifier ───────────────────────────────────────────────────────

class EmbeddingClassifier:
    """
    SentenceTransformer embeddings + cosine-similarity nearest-neighbour.

    This is the production classifier used in the intent router.
    - Encodes all training examples once at fit() time
    - At inference, embeds the query and finds k nearest neighbours
    - Returns confidence as the softmax-normalised mean similarity

    Model: all-MiniLM-L6-v2 (22M params, 80ms/batch on CPU)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        k: int = 5,
        temperature: float = 0.1,
    ) -> None:
        self.model_name  = model_name
        self.k           = k
        self.temperature = temperature
        self._model      = None
        self._embeddings: Optional[np.ndarray] = None
        self._train_labels: Optional[List[str]] = None
        self.labels_: Optional[List[str]] = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: List[str], labels: List[str]) -> "EmbeddingClassifier":
        self.labels_        = sorted(set(labels))
        self._train_labels  = labels

        logger.info(
            "EmbeddingClassifier: encoding %d training examples...", len(texts)
        )
        self._embeddings = self._encode(texts)
        logger.info(
            "EmbeddingClassifier fitted. Embedding shape: %s",
            self._embeddings.shape,
        )
        return self

    def predict(self, texts: List[str]) -> List[str]:
        proba = self.predict_proba(texts)
        return [self.labels_[i] for i in proba.argmax(axis=1)]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Returns (N, n_classes) probability matrix using softmax over
        mean cosine similarities to k nearest training neighbours per class.
        """
        if self._embeddings is None or self._train_labels is None:
            raise RuntimeError("Call .fit() before .predict_proba()")

        from sklearn.preprocessing import normalize

        query_embs = self._encode(texts)                     # (N, D)
        train_embs = self._embeddings                        # (M, D)

        # Cosine similarities: (N, M)
        query_norm = normalize(query_embs)
        train_norm = normalize(train_embs)
        sims = query_norm @ train_norm.T                     # (N, M)

        n_queries = len(texts)
        n_classes = len(self.labels_)
        proba = np.zeros((n_queries, n_classes))

        for q_idx in range(n_queries):
            q_sims = sims[q_idx]                             # (M,)
            top_k_idx = np.argsort(q_sims)[::-1][:self.k]

            # Aggregate similarities per class
            class_sims = {cls: [] for cls in self.labels_}
            for idx in top_k_idx:
                cls = self._train_labels[idx]
                class_sims[cls].append(q_sims[idx])

            # Mean similarity per class
            class_scores = np.array([
                np.mean(class_sims[cls]) if class_sims[cls] else 0.0
                for cls in self.labels_
            ])

            # Softmax with temperature
            scaled = class_scores / self.temperature
            scaled -= scaled.max()   # numerical stability
            exp_s = np.exp(scaled)
            proba[q_idx] = exp_s / exp_s.sum()

        return proba

    def encode(self, texts: List[str]) -> np.ndarray:
        """Public method to get sentence embeddings (for quality gates, drift, etc.)."""
        return self._encode(texts)

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def save(self, path: str | Path) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save everything except the heavy SentenceTransformer model
        state = {
            "model_name":     self.model_name,
            "k":              self.k,
            "temperature":    self.temperature,
            "embeddings":     self._embeddings,
            "train_labels":   self._train_labels,
            "labels_":        self.labels_,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("EmbeddingClassifier saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingClassifier":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            model_name=state["model_name"],
            k=state["k"],
            temperature=state["temperature"],
        )
        obj._embeddings    = state["embeddings"]
        obj._train_labels  = state["train_labels"]
        obj.labels_        = state["labels_"]
        logger.info("EmbeddingClassifier loaded from %s", path)
        return obj


# ── Training helper ────────────────────────────────────────────────────────────

def train_baseline_classifier(
    dataset,
    cfg: Optional[dict] = None,
    classifier_type: str = "svm",
) -> SVMClassifier | EmbeddingClassifier:
    """
    Train a baseline intent classifier on a HuggingFace Dataset.

    Parameters
    ----------
    dataset          : Dataset with columns: text, domain
    cfg              : Loaded domains.yaml config
    classifier_type  : "svm" (default) or "embedding"

    Returns
    -------
    Fitted classifier
    """
    texts  = dataset["text"]
    labels = dataset["domain"]

    if classifier_type == "svm":
        clf = SVMClassifier()
    elif classifier_type == "embedding":
        clf = EmbeddingClassifier()
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    clf.fit(texts, labels)
    return clf


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_classifier(
    clf: SVMClassifier | EmbeddingClassifier,
    dataset,
) -> ClassificationReport:
    """
    Evaluate a fitted classifier on a HuggingFace Dataset.

    Returns
    -------
    ClassificationReport with accuracy, per-class metrics, confusion matrix
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    texts  = dataset["text"]
    labels = dataset["domain"]
    preds  = clf.predict(texts)

    accuracy = accuracy_score(labels, preds)

    # Get all classes present in either truth or predictions
    all_classes = sorted(set(labels) | set(preds))

    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=all_classes, average=None, zero_division=0,
    )

    per_class = {
        cls: {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]),  4),
            "f1":        round(float(f1[i]),   4),
            "support":   int(support[i]),
        }
        for i, cls in enumerate(all_classes)
    }

    cm = confusion_matrix(labels, preds, labels=all_classes)

    report = ClassificationReport(
        accuracy=round(accuracy, 4),
        per_class=per_class,
        confusion_matrix=cm,
        domains=all_classes,
    )

    logger.info("Classifier evaluation:\n%s", report.summary())
    return report
