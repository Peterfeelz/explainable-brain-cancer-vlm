"""v1.9: cumulative online distillation loop and significance-based promotion."""
import numpy as np, pytest
from bcd.train import mcnemar_worse_p, promotion_decision
from bcd.config import load_config


def test_mcnemar_values():
    cur = np.array([1] * 100, bool)
    assert mcnemar_worse_p(cur, cur) == 1.0                                  # identical predictions
    new = cur.copy(); new[:2] = False                                        # 2 lost, 0 gained
    assert abs(mcnemar_worse_p(cur, new) - 0.25) < 1e-12                     # P(X>=2 | n=2) = 1/4
    new = cur.copy(); new[:10] = False                                       # 10 lost, 0 gained
    assert mcnemar_worse_p(cur, new) < 0.01
    c2 = np.array([0] * 10 + [1] * 90, bool); n2 = np.array([1] * 10 + [1] * 90, bool)
    assert mcnemar_worse_p(c2, n2) == 1.0                                    # strictly better


def test_promotion_rule():
    d = load_config().distill
    m = lambda acc, sens: {"accuracy": acc, "tumour_sensitivity": sens}
    cur = np.ones(400, bool)
    small_dip = cur.copy(); small_dip[:2] = False      # v1 rule would reject this (-0.5 pp); v1.9 keeps it
    ok, p = promotion_decision(m(.72, .99), m(.715, .99), cur, small_dip, d)
    assert ok and p > 0.05
    big_dip = cur.copy(); big_dip[:15] = False
    assert not promotion_decision(m(.72, .99), m(.68, .99), cur, big_dip, d)[0]
    assert not promotion_decision(m(.72, .99), m(.75, .97), cur, cur, d)[0]   # sensitivity guard (-2 pp)


def test_cumulative_simulation_uses_all_labels(tmp_path):
    from bcd.mock import MockWorld, MockModel, MockTeacher
    from bcd.simulate import Simulator
    cfg = load_config(distill__round_labels=20, clinician__budget_per_100=10)
    w = MockWorld(n=300, size=48)
    stream, val, test = w.split(0, 200), w.split(200, 260), w.split(260, 300)
    seen = []

    class Rec(MockModel):
        def distill(self, buffer, seed=0, cumulative=False):
            seen.append((len(buffer), cumulative)); return {"promoted": True, "train_size": len(buffer)}
    st, te = Rec(w, 0.5, "s"), MockTeacher(w, 0.85, "t")
    boxes = w.detector_boxes([p for p, _ in stream + val + test])
    s = Simulator(cfg, stream, val, test, st, te, boxes, w.ref_mask, st.distill, st.evaluate, tmp_path,
                  cond="P_v2", log=lambda *a: None).run()
    import pandas as pd
    df = pd.read_csv(tmp_path / "P_v2_seed0" / "cases.csv")
    assert all(c for _, c in seen)                                   # cumulative flag passed
    assert sum(n for n, _ in seen) == int(df.buffered.sum())         # no label dropped at stream end (flush)
    assert all(n >= 20 for n, _ in seen[:-1])


from tests.test_tiny_qwen2vl import tiny_model_dir  # noqa: F401  (fixture)


def test_cumulative_round_tiny(tiny_model_dir):
    pytest.importorskip("transformers")
    import torch
    from tests import test_tiny_qwen2vl as T
    d = tiny_model_dir
    from bcd.vlm import VLM
    from bcd.runtime import StudentHandle
    classes = ["glioma", "meningioma", "pituitary", "no_tumour"]
    v = VLM(d, classes, dtype="float32", device="cpu", image_size=56,
            lora=dict(r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"]))
    ims = {f"p{i}": T._img(i) for i in range(10)}
    cfg = load_config(distill__cum_lr=1e-3, distill__cum_epochs=1)
    replay = [dict(path=f"p{i}", y_star=i % 4, z_cloud=None, plan=None, weight=1.0, teacher_wrong=False) for i in range(6, 8)]
    val = [(f"p{i}", i % 4) for i in range(8, 10)]
    sh = StudentHandle(cfg, v, ims.__getitem__, v.adapter_state(), replay, val)
    b1 = [dict(path=f"p{i}", y_star=i % 4, z_cloud=[2.0, 0, 0, 0], plan=None, weight=1.0, teacher_wrong=False) for i in range(3)]
    b2 = [dict(path=f"p{i}", y_star=i % 4, z_cloud=None, plan="refer to neurosurgery", weight=0.9, teacher_wrong=False) for i in range(3, 6)]
    r1 = sh.distill(b1, seed=1, cumulative=True)
    r2 = sh.distill(b2, seed=2, cumulative=True)
    assert len(sh.memory) == 6
    assert r1["train_size"] == 3 + 2 and r2["train_size"] == 6 + 2        # all labels so far + replay
    assert "promoted" in r2 and 0.0 <= r2["p_worse"] <= 1.0 and "correct" not in r2
    sh.reset(); assert sh.memory == [] and sh.cur_correct is None
