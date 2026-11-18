import numpy as np, pytest
from bcd.config import load_config
from bcd.mock import MockWorld, MockModel, MockTeacher
from bcd.simulate import Simulator, CONDITIONS

@pytest.mark.parametrize("cond", list(CONDITIONS))
def test_all_conditions_run(tmp_path, cond):
    cfg = load_config(distill__batch_size=16, clinician__budget_per_100=20)
    w = MockWorld(n=260, size=48)
    stream, val, test = w.split(0, 160), w.split(160, 220), w.split(220, 260)
    student, teacher = MockModel(w, 0.5, "s"), MockTeacher(w, 0.85, "t")
    boxes = w.detector_boxes([p for p, _ in stream + val + test])
    sim = Simulator(cfg, stream, val, test, student, teacher, boxes, w.ref_mask, student.distill,
                    student.evaluate, tmp_path, cond=cond, seed=0, random_rate=0.3, log=lambda *a: None)
    s = sim.run()
    assert (tmp_path / f"{cond}_seed0" / "summary.json").exists()
    if cond == "B0":
        assert s["cloud_calls"] == 0
    if cond == "B1":
        assert s["escalation_rate"] == 1.0
    if cond in ("P", "B3", "A1"):
        assert s["final"]["round"] >= 1 or s["cloud_calls"] < 16

def test_proposed_reviews_respect_budget(tmp_path):
    cfg = load_config(distill__batch_size=16, clinician__budget_per_100=10)
    w = MockWorld(n=300, size=48)
    stream, val, test = w.split(0, 200), w.split(200, 260), w.split(260, 300)
    student, teacher = MockModel(w, 0.5, "s"), MockTeacher(w, 0.85, "t")
    boxes = w.detector_boxes([p for p, _ in stream + val + test])
    s = Simulator(cfg, stream, val, test, student, teacher, boxes, w.ref_mask, student.distill,
                  student.evaluate, tmp_path, cond="P", log=lambda *a: None).run()
    assert s["reviews_per_100"] <= 10.0 + 1e-9
