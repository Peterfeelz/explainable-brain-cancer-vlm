"""v2.1: plain-language explanations, bounding-box mismatch prompt and calibration scores."""
import numpy as np
from bcd.explain import explain, mismatch, mismatch_prompt, region_name
from bcd.gate import Grounding, GateConfig, composite_gate, GROUNDINGS
from bcd.compare import calibration_scores

C = ["glioma", "meningioma", "pituitary", "no_tumour"]


def _grounding():
    rng = np.random.default_rng(0); out = []
    centres = {0: (0.42, 0.40), 1: (0.85, 0.5), 2: (0.5, 0.78)}
    for i in range(400):
        y = i % 4
        if y == 3:
            out.append(([], 3)); continue
        cx, cy = centres[y]; cx += rng.normal(0, .03); cy += rng.normal(0, .03)
        out.append(([(448 * (cx - .05), 448 * (cy - .05), 448 * (cx + .05), 448 * (cy + .05), .8)], y))
    return Grounding(4).fit_location([b for b, _ in out], [y for _, y in out]).fit_xai(list(np.linspace(0, 1, 30)), [1] * 30)


def test_region_names():
    assert "skull-base" in region_name((200, 330, 248, 380, .9))
    assert "surface" in region_name((380, 200, 440, 260, .9))


def test_explain_mismatch_case_reads_like_the_demo():
    g = _grounding(); GROUNDINGS["ex"] = g
    cfg = GateConfig(w1=0.6, w2=0.35, w3=0.05, tau_gate=0.52, grounding="ex", use_overrides=False)
    p = np.array([0.49, 0.50, 0.01, 0.0])                       # meningioma vs glioma, lesion where gliomas lie
    boxes = [(165, 155, 215, 205, 0.8)]
    cam = np.zeros((448, 448)); cam[160:200, 170:210] = 1
    r = composite_gate(p, cam, boxes, None, 3, cfg)
    assert r.a_yolo < 0.3 and r.decision == "escalate"
    txt = explain(C, p, r, boxes, g, np.array([.14, .06, .01, .79]), "clinician", ["disagreement"], (0.6, 0.35, 0.05), 0.52)
    assert "uncertain" in txt and "does not support" in txt and "glioma" in txt and "specialist" in txt
    assert "no tumour although the detector found a lesion" in txt
    assert mismatch(C, p, r, boxes, g)
    q = mismatch_prompt(C, p, boxes, g)
    assert "meningioma" in q and "box from x=165" in q


def test_no_box_positive_is_a_mismatch():
    cfg = GateConfig(tau_gate=0.0, use_overrides=False)
    p = np.array([0.9, 0.05, 0.03, 0.02])
    r = composite_gate(p, None, [], None, 3, cfg)
    assert mismatch(C, p, r, [], None)
    assert "found no lesion" in mismatch_prompt(C, p, [], None)


def test_calibration_scores():
    z = np.array([[5, 0, 0, 0], [0, 5, 0, 0], [5, 0, 0, 0], [0, 0, 5, 0]], float)
    s, rel = calibration_scores(z, [0, 1, 1, 2], 1.0)
    assert 0 <= s["ece10"] <= 1 and s["brier"] > 0 and s["nll"] > 0 and rel
