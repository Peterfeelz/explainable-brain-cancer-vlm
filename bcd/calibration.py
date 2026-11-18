"""Gate recalibration — Step 10 (Equations 3.19-3.21) and threshold fitting."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

from .gate import softmax


def fit_temperature(logits: np.ndarray, labels: np.ndarray, bounds=(0.05, 10.0)) -> float:
    """Eq. 3.21: T* = argmin_T NLL(softmax(z/T), y)."""
    logits = np.asarray(logits, dtype=np.float64); labels = np.asarray(labels, dtype=int)
    if len(labels) == 0:
        return 1.0

    def nll(logT):
        p = softmax(logits, np.exp(logT))
        return -np.mean(np.log(p[np.arange(len(labels)), labels] + 1e-12))

    r = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(r.x))


def update_threshold(tau: float, n_wrong: int, n_audited: int, far_target=0.02, eta=0.5,
                     lo=0.5, hi=0.95) -> tuple[float, float]:
    """Eq. 3.19-3.20: proportional controller on the observed false-accept rate."""
    if n_audited == 0:
        return tau, float("nan")
    far = n_wrong / n_audited
    return float(np.clip(tau + eta * (far - far_target), lo, hi)), far


def fit_tau_for_fnr(g_scores, is_positive_true, is_correct, target_fnr=0.05, grid=None) -> float:
    """Choose the smallest tau_gate such that, among true tumour cases ACCEPTED locally,
    the fraction misclassified (missed or wrong class) is <= target_fnr. Falls back to the
    tau with minimum such rate."""
    g = np.asarray(g_scores); pos = np.asarray(is_positive_true, bool); ok = np.asarray(is_correct, bool)
    grid = np.linspace(0.3, 0.98, 69) if grid is None else grid
    best, best_rate = grid[-1], 1.0
    for t in grid:
        acc = (g >= t) & pos
        rate = 0.0 if acc.sum() == 0 else float((~ok[acc]).mean())
        if rate <= target_fnr:
            return float(t)
        if rate < best_rate:
            best, best_rate = t, rate
    return float(best)
