import numpy as np
from bcd.calibration import fit_temperature, update_threshold, fit_tau_for_fnr
from bcd.metrics import classification_metrics, ece, risk_coverage, far_at_escalation

def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 2000)
    z = rng.normal(0, 1, (2000, 4)); z[np.arange(2000), y] += 1.0
    z_over = z * 4.0                                   # overconfident logits
    T = fit_temperature(z_over, y)
    assert 2.5 < T < 6.0

def test_threshold_controller():
    tau, far = update_threshold(0.7, 10, 100, 0.02, 0.5)
    assert abs(far - 0.1) < 1e-12 and abs(tau - 0.74) < 1e-12
    tau2, _ = update_threshold(0.94, 50, 100, 0.02, 0.5)
    assert tau2 == 0.95

def test_fit_tau():
    g = np.linspace(0, 1, 100); pos = np.ones(100, bool); ok = g > 0.5
    t = fit_tau_for_fnr(g, pos, ok, 0.0)
    assert 0.49 < t < 0.52

def test_metrics():
    y = np.array([0, 1, 2, 3, 3]); p = np.eye(4)[[0, 1, 2, 3, 0]]
    m = classification_metrics(y, p.argmax(1), p * 0.97 + 0.0075, 3)
    assert m["accuracy"] == 0.8 and m["tumour_sensitivity"] == 1.0 and m["specificity"] == 0.5
    cov, risk, aurc = risk_coverage(np.array([.9, .8, .1]), np.array([1, 1, 0], bool))
    assert risk[0] == 0 and abs(risk[-1] - 1 / 3) < 1e-9 and aurc >= 0
    assert far_at_escalation(np.array([.9, .8, .1]), np.array([1, 0, 0], bool), 1 / 3) == 0.5
