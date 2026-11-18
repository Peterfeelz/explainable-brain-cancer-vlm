"""v2.0: class-aware grounding terms, gate study, statistics and judge helpers."""
import json, numpy as np, pandas as pd, pytest
from bcd.gate import Grounding, box_features, xai_raw, composite_gate, GateConfig, GROUNDINGS
from bcd.config import load_config


def _loc_data(n=400, seed=0):
    """Synthetic lesions: class 0 left, class 1 top, class 2 centre-bottom, class 3 no box."""
    rng = np.random.default_rng(seed); out = []
    centres = {0: (0.25, 0.5), 1: (0.5, 0.2), 2: (0.5, 0.75)}
    for i in range(n):
        y = i % 4
        if y == 3:
            out.append(([], y)); continue
        cx, cy = centres[y]; cx += rng.normal(0, 0.05); cy += rng.normal(0, 0.05)
        b = (448 * (cx - .05), 448 * (cy - .05), 448 * (cx + .05), 448 * (cy + .05), float(rng.uniform(.4, .9)))
        out.append(([b], y))
    return out


def test_box_features():
    assert box_features([]).sum() == 0
    f = box_features([(0, 0, 44.8, 44.8, 0.5), (100, 100, 200, 200, 0.9)])
    assert f[0] == 1 and abs(f[1] - 0.9) < 1e-9 and f[-1] == 2


def test_location_model_learns_lesion_position():
    tr, te = _loc_data(400, 0), _loc_data(200, 1)
    g = Grounding(4).fit_location([b for b, _ in tr], [y for _, y in tr])
    acc = np.mean([g.location_probs(b).argmax() == y for b, y in te])
    assert acc > 0.9
    b, y = te[2]                                          # a class-2 lesion
    q = g.location_probs(b)
    assert q[2] > 0.8 and q[0] < 0.1                       # agreement high for the right class, low otherwise


def test_class_relative_xai_removes_class_bias():
    g = Grounding(4).fit_xai([0.9, 0.95, 1.0, 1.0, 1.0, 0.01, 0.02, 0.05, 0.1, 0.2], [3] * 5 + [0] * 5)
    # v1 would give every no-tumour prediction ~1 and every tumour prediction ~0; v2 ranks within class
    assert g.xai_rel(0.2, 0) == 0.9 and g.xai_rel(0.9, 3) == 0.1
    assert 0 <= g.xai_rel(0.03, 0) <= 1


def test_xai_raw_cases():
    m = np.zeros((10, 10), bool); m[2:4, 2:4] = True
    ref = np.zeros((10, 10), bool); ref[2:4, 2:3] = True
    assert xai_raw(m, ref, [], True) == 0.5
    assert xai_raw(m, None, [], True) == 0.0
    assert xai_raw(m, None, [], False) == 1.0                         # no lesion found, evidence not on a lesion
    assert xai_raw(m, None, [(2, 2, 4, 4, .9)], False) == 0.0         # evidence sits on a detected lesion


def test_composite_gate_v2_uses_grounding():
    tr = _loc_data(400)
    g = Grounding(4).fit_location([b for b, _ in tr], [y for _, y in tr]).fit_xai(list(np.linspace(0, 1, 20)), [0] * 20)
    GROUNDINGS["t"] = g
    cfg = GateConfig(w1=0.5, w2=0.3, w3=0.2, tau_gate=0.0, use_overrides=False, grounding="t")
    cam = np.zeros((448, 448)); cam[200:240, 90:130] = 1
    good = [(90, 200, 135, 245, 0.9)]                                # lesion on the left = class 0 territory
    r = composite_gate(np.array([.7, .1, .1, .1]), cam, good, None, 3, cfg)
    r_wrong = composite_gate(np.array([.1, .1, .7, .1]), cam, good, None, 3, cfg)   # says class 2 for a left lesion
    assert r.a_yolo > 0.8 and r_wrong.a_yolo < 0.2 and r.g > r_wrong.g
    assert r.extras["a_xai_raw"] is not None


def test_p_v3_runs_and_saves_gate(tmp_path):
    from bcd.mock import MockWorld, MockModel, MockTeacher
    from bcd.simulate import Simulator
    cfg = load_config(distill__round_labels=20, data__image_size=48)
    w = MockWorld(n=320, size=48)
    stream, val, test, trn = w.split(0, 160), w.split(160, 220), w.split(220, 260), w.split(260, 320)
    boxes = w.detector_boxes([p for p, _ in stream + val + test + trn])
    st, te = MockModel(w, 0.5, "s"), MockTeacher(w, 0.85, "t")
    sim = Simulator(cfg, stream, val, test, st, te, boxes, w.ref_mask, st.distill, st.evaluate, tmp_path,
                    cond="P_v3", log=lambda *a: None, train_boxes=[(boxes[p], y) for p, y in trn])
    s = sim.run()
    assert s["gate"]["grounding"] == "P_v3_seed0" and (tmp_path / "P_v3_seed0" / "grounding.pkl").exists()
    assert abs(s["gate"]["w1"] + s["gate"]["w2"] + s["gate"]["w3"] - 1) < 1e-9
    assert set(sim.val_terms) >= {"g_base", "a_yolo", "a_xai"}


class _RT(dict):
    pass


def _mock_rt(tmp_path, cfg):
    from bcd.mock import MockWorld, MockModel, MockTeacher
    w = MockWorld(n=320, size=48)
    stream, val, test, trn = w.split(0, 160), w.split(160, 220), w.split(220, 260), w.split(260, 320)
    boxes = w.detector_boxes([p for p, _ in stream + val + test + trn])
    st = MockModel(w, 0.5, "s"); st.reset = lambda: None; st.seed_state = None
    return dict(student=st, val=val, test=test, stream=stream, boxes=boxes, refs=w.ref_mask,
                train_boxes=[(boxes[p], y) for p, y in trn], teacher=MockTeacher(w, 0.85, "t")), w


def test_gate_study_v2_runs(tmp_path, monkeypatch):
    import bcd.runtime as R
    cfg = load_config(data__image_size=48)
    monkeypatch.setattr(R, "work", lambda c: tmp_path)
    rt, _ = _mock_rt(tmp_path, cfg)
    df = R.gate_study_v2(cfg, rt, folds=3, n_boot=50, log=lambda *a: None)
    assert len(df) == 8 and df.aurc.between(0, 1).all()
    info = json.loads((tmp_path / "gate_study_v2.json").read_text())
    assert 0 <= info["bootstrap_p_v2_not_better_than_gbase"] <= 1


def test_significance(tmp_path, monkeypatch):
    import bcd.runtime as R
    from bcd.simulate import Simulator
    cfg = load_config(data__image_size=48, distill__round_labels=20)
    monkeypatch.setattr(R, "work", lambda c: tmp_path)
    rt, w = _mock_rt(tmp_path, cfg)
    for cond in ("P_v2", "B5_v2"):
        st = rt["student"].__class__(w, 0.5, "s")
        sim = Simulator(cfg, rt["stream"], rt["val"], rt["test"], st, rt["teacher"], rt["boxes"], w.ref_mask,
                        st.distill, st.evaluate, tmp_path / "runs", cond=cond, log=lambda *a: None,
                        random_rate=0.3)
        sim.run()
        z = np.stack([st.logits(p) for p, _ in rt["test"]]); np.save(sim.out / "test_logits.npy", z)
    t2 = R.significance(cfg, rt, [("P_v2_seed0", "B5_v2_seed0")], n_boot=50, log=lambda *a: None)
    assert len(t2) == 1 and 0 <= t2.student_mcnemar_p[0] <= 1 and "system_diff_lo" in t2


def test_parse_scores():
    from bcd.runtime import _parse_scores
    s = _parse_scores('{"correctness": 7, "completeness": 6.5, "conciseness": 12}')
    assert s == {"correctness": 7.0, "completeness": 6.5, "conciseness": 10.0}
    assert np.isnan(_parse_scores("no numbers")["correctness"])
