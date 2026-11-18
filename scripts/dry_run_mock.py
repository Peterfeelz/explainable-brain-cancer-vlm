"""Dry run of every condition with CPU mocks — checks the pipeline end to end.
The numbers printed are from synthetic mocks and are NOT experimental results."""
import sys, tempfile, json
sys.path.insert(0, ".")
from bcd.config import load_config
from bcd.mock import MockWorld, MockModel, MockTeacher
from bcd.simulate import Simulator, CONDITIONS
from bcd.report import tables, figure_rounds
cfg = load_config(distill__batch_size=32)
w = MockWorld(n=1400, size=48)
stream, val, test = w.split(0, 1000), w.split(1000, 1200), w.split(1200, 1400)
boxes = w.detector_boxes([p for p, _ in stream + val + test])
out = tempfile.mkdtemp()
rr = None
for c in ["P"] + [k for k in CONDITIONS if k != "P"]:
    s = MockModel(w, 0.5, "s"); t = MockTeacher(w, 0.85, "t")
    r = Simulator(cfg, stream, val, test, s, t, boxes, w.ref_mask, s.distill, s.evaluate, out, cond=c,
                  random_rate=rr if c.split('_')[0] in ('B2','B3','B5') else None, log=lambda *a: None).run()
    if c == "P": rr = r["escalation_rate"]
    print(f"{c:3s} esc={r['escalation_rate']:.2f} reviews/100={r['reviews_per_100']:.1f} rounds={r['final']['round']} "
          f"student_f1={r['final']['student_f1']:.2f} system_acc={r.get('system_accuracy', float('nan')):.2f}")
t46, t48 = tables(out, out + "/results"); figure_rounds(out, out + "/fig.png")
print("tables + figure written to", out)
