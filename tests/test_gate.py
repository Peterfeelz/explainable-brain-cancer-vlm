import numpy as np
from bcd.gate import (softmax, norm_entropy, g_base, cam_to_mask, bbox_of_mask, iou, a_yolo, a_xai,
                      composite_gate, GateConfig, kl_div, cloud_route, AgreementTracker)

def test_softmax_temperature_keeps_argmax():
    z = np.array([2.0, 0.5, -1.0, 0.0])
    for T in (0.5, 1, 3):
        assert softmax(z, T).argmax() == 0
    assert softmax(z, 3).max() < softmax(z, 1).max()

def test_entropy_bounds():
    assert abs(norm_entropy(np.ones(4) / 4) - 1) < 1e-9
    assert norm_entropy(np.array([1, 0, 0, 0.0])) < 1e-6

def test_g_base():
    p = np.array([0.7, 0.1, 0.1, 0.1])
    assert abs(g_base(p, 0.6) - (0.6 * 0.7 + 0.4 * (1 - norm_entropy(p)))) < 1e-12

def test_mask_and_iou():
    cam = np.zeros((20, 20)); cam[5:10, 5:10] = 1
    m = cam_to_mask(cam, 0.10)
    assert m.sum() > 0 and m[5:10, 5:10].sum() == m.sum()
    assert bbox_of_mask(m)[0] >= 5
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9

def test_a_yolo_edge_cases():
    m = np.zeros((20, 20), bool); m[2:8, 2:8] = True
    assert a_yolo(m, [], positive=True) == 0.0
    assert a_yolo(m, [], positive=False) == 1.0
    assert a_yolo(m, [(2, 2, 8, 8, .9)], positive=False) == 0.0
    assert a_yolo(m, [(2, 2, 8, 8, .9)], positive=True) == 1.0

def test_a_xai():
    m = np.zeros((10, 10), bool); m[0:2, 0:5] = True
    ref = np.zeros((10, 10), bool); ref[0:2, 0:10] = True
    assert a_xai(m, ref, True) == 1.0
    assert a_xai(m, np.zeros((10, 10), bool), False) == 1.0
    assert a_xai(m, None, True) == 0.0

def test_composite_and_overrides():
    cfg = GateConfig(tau_gate=0.6)
    p = np.array([0.9, 0.05, 0.03, 0.02])            # confident glioma
    cam = np.zeros((30, 30)); cam[0:5, 0:5] = 1        # looks at the corner
    boxes = [(20, 20, 28, 28, 0.9)]                    # lesion elsewhere
    r = composite_gate(p, cam, boxes, None, 3, cfg)
    assert r.a_xai == 0.0 and r.override == "ungrounded_positive" and r.decision == "escalate"
    cam2 = np.zeros((30, 30)); cam2[20:28, 20:28] = 1
    r2 = composite_gate(p, cam2, boxes, None, 3, cfg)
    assert r2.decision == "accept" and r2.a_yolo > 0.5
    pn = np.array([0.02, 0.03, 0.05, 0.9])             # "no tumour" but YOLO strongly disagrees
    r3 = composite_gate(pn, cam2, boxes, None, 3, cfg)
    assert r3.override == "possible_missed_tumour"
    # ablation renormalises weights
    r4 = composite_gate(p, None, boxes, None, 3, GateConfig(use_yolo=False, use_xai=False))
    assert abs(r4.g - r4.g_base) < 1e-12

def test_cloud_route_and_tracker():
    cfg = GateConfig()
    p1 = np.array([.9, .05, .03, .02]); p2 = np.array([.02, .03, .05, .9])
    cam = np.zeros((10, 10)); cam[1:4, 1:4] = 1
    a = composite_gate(p1, cam, [(1, 1, 4, 4, .9)], None, 3, cfg)
    b = composite_gate(p2, cam, [(1, 1, 4, 4, .9)], None, 3, cfg)
    route, why = cloud_route(a, b, p1, p2, False, 0.75, 0.5)
    assert route == "clinician" and "disagreement" in why
    assert kl_div(p1, p1) < 1e-9
    t = AgreementTracker(3); [t.add(x) for x in (1, 1, 0, 0)]
    assert abs(t.value - 1 / 3) < 1e-9
