"""Metrics for PII detection, document-level leakage, and OCR.

Kept dependency-free (pure Python) so they run and unit-test anywhere. These
are the metrics named in proposal sections 18.3-18.4 (PII, DZLR) and 18.1 (OCR).
"""

from __future__ import annotations

from typing import Sequence


# --- binary classification (candidate-level PII) ---------------------------

def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Precision / recall / F1 / F2 for the positive (PII) class.

    F2 weights recall higher, matching the proposal's safety priority
    (missing PII costs more than over-redaction).
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = _fbeta(precision, recall, 1.0)
    f2 = _fbeta(precision, recall, 2.0)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "f2": f2,
    }


def _fbeta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


# --- document-level privacy ------------------------------------------------

def document_zero_leakage_rate(
    doc_ids: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict:
    """Fraction of documents with zero residual PII (no false negatives).

    A document leaks if any gold-PII candidate was predicted non-PII.
    """
    leaked: dict[str, bool] = {}
    for did, t, p in zip(doc_ids, y_true, y_pred):
        if did not in leaked:
            leaked[did] = False
        if t == 1 and p == 0:  # residual PII
            leaked[did] = True
    n = len(leaked)
    clean = sum(1 for v in leaked.values() if not v)
    residual_docs = n - clean
    return {
        "n_documents": n,
        "zero_leakage_documents": clean,
        "dzlr": clean / n if n else 0.0,
        "residual_pii_documents": residual_docs,
    }


# --- OCR -------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    # word-level Levenshtein
    prev = list(range(len(hyp_words) + 1))
    for i, rw in enumerate(ref_words, 1):
        cur = [i]
        for j, hw in enumerate(hyp_words, 1):
            cost = 0 if rw == hw else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(ref_words)
