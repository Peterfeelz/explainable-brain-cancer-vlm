"""Evaluation metrics of Table 3.7."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true, y_pred, probs=None, no_tumour_idx: int = 3) -> dict:
    y_true = np.asarray(y_true, int); y_pred = np.asarray(y_pred, int)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else float("nan"),
    }
    pos = y_true != no_tumour_idx
    out["tumour_sensitivity"] = float((y_pred[pos] != no_tumour_idx).mean()) if pos.any() else float("nan")
    neg = ~pos
    out["specificity"] = float((y_pred[neg] == no_tumour_idx).mean()) if neg.any() else float("nan")
    if probs is not None and len(np.unique(y_true)) > 1:
        probs = np.asarray(probs, float)
        try:
            out["auroc"] = float(roc_auc_score(y_true, probs, multi_class="ovr",
                                               labels=list(range(probs.shape[1]))))
        except ValueError:
            out["auroc"] = float("nan")
        out["ece"] = ece(probs, y_true)
    return out


def ece(probs, y_true, n_bins: int = 15) -> float:
    probs = np.asarray(probs, float); y_true = np.asarray(y_true, int)
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def risk_coverage(scores, correct):
    """Risk-coverage curve: accept highest-score cases first. Returns coverage, risk, AURC."""
    scores = np.asarray(scores, float); correct = np.asarray(correct, bool)
    order = np.argsort(-scores)
    err = (~correct[order]).astype(float)
    n = len(err)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return cov, risk, float(trap(risk, cov)) if n > 1 else float(risk.mean() if n else np.nan)


def far_at_escalation(scores, correct, escalation_rate: float) -> float:
    """False-accept rate among the (1-escalation_rate) highest-scoring cases."""
    scores = np.asarray(scores, float); correct = np.asarray(correct, bool)
    k = int(round((1 - escalation_rate) * len(scores)))
    if k == 0:
        return float("nan")
    acc = np.argsort(-scores)[:k]
    return float((~correct[acc]).mean())
