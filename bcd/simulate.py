"""Deployment-stream simulation of the confidence-gated distillation loop (Sections 3.4, 3.7).

Conditions (Table 3.6):
  B0 local only | B1 cloud only | B2 softmax routing | B3 softmax routing + distillation (Deng-like)
  B4 offline full distillation | B5 random-selection distillation | P proposed
  A1 no A_YOLO | A2 no A_XAI | A3 no clinician | A4 no recalibration

The simulator talks to models through four small interfaces so that it can be dry-run on CPU
with mocks (bcd/mock.py) and run for real on GPU (bcd/runtime.py):
  student.logits(path) -> z            student.gradcam(path, T) -> (z, cam)
  teacher.cached(path) -> dict(z, cam, plan_fn)
  distill(buffer) -> dict(promoted, val)   (trains the student in place)
  evaluate(paths, y, T) -> (metrics, logits)
"""
from __future__ import annotations
import json, random, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np

from .gate import GateConfig, composite_gate, cloud_route, AgreementTracker, softmax, Grounding, GROUNDINGS
from .calibration import fit_temperature, update_threshold, fit_tau_for_fnr

CONDITIONS = {
    #       routing        distil  clinician recal  yolo  xai
    "B0": dict(routing="none",      distill=False, clinician=False, recal=False, yolo=False, xai=False),
    "B1": dict(routing="all",       distill=False, clinician=False, recal=False, yolo=False, xai=False),
    "B2": dict(routing="softmax",   distill=False, clinician=False, recal=False, yolo=False, xai=False),
    "B3": dict(routing="softmax",   distill=True,  clinician=False, recal=False, yolo=False, xai=False),
    "B4": dict(routing="offline",   distill=True,  clinician=False, recal=False, yolo=False, xai=False),
    "B5": dict(routing="random",    distill=True,  clinician=True,  recal=True,  yolo=True,  xai=True),
    "P":  dict(routing="composite", distill=True,  clinician=True,  recal=True,  yolo=True,  xai=True),
    "A1": dict(routing="composite", distill=True,  clinician=True,  recal=True,  yolo=False, xai=True),
    "A2": dict(routing="composite", distill=True,  clinician=True,  recal=True,  yolo=True,  xai=False),
    "A3": dict(routing="composite", distill=True,  clinician=False, recal=True,  yolo=True,  xai=True),
    "A4": dict(routing="composite", distill=True,  clinician=True,  recal=False, yolo=True,  xai=True),
}
# v1.9: the same conditions with the revised (cumulative) online distillation loop: every round trains on
# ALL verified labels collected so far (plus replay), rounds are triggered by new labels, the last labels
# are flushed at the end of the stream, and promotion uses a significance test (Section 3.4.9, revised).
for _k in ("P", "B3", "B5"):
    CONDITIONS[f"{_k}_v2"] = dict(CONDITIONS[_k], loop="cumulative")
# v2.0: cumulative loop + class-aware grounding terms (lesion-location agreement A_YOLO, class-relative A_XAI)
for _k in ("P", "A1", "A2", "A3", "A4"):
    CONDITIONS[f"{_k}_v3"] = dict(CONDITIONS[_k], loop="cumulative", gate="v2")


@dataclass
class CaseLog:
    idx: int
    path: str
    y_true: int
    y_local: int
    conf_local: float
    g: float | None
    a_yolo: float | None
    a_xai: float | None
    override: str | None
    decision: str
    route: str | None = None
    reasons: list = field(default_factory=list)
    y_cloud: int | None = None
    g_cloud: float | None = None
    y_final: int | None = None
    reviewed: bool = False
    buffered: bool = False
    round: int = 0
    tau_gate: float | None = None
    T: float | None = None


class Simulator:
    def __init__(self, cfg, stream, val, test_eval, student, teacher, detector_boxes, ref_masks,
                 distill_fn, evaluate_fn, out_dir, cond="P", seed=0, random_rate=None, log=print,
                 train_boxes=None):
        """stream/val/test_eval: lists of (path, y). detector_boxes: dict path->boxes.
        ref_masks: callable path->bool mask or None.
        random_rate: escalation rate of P; used by B5 (random selection) and, when given, to match the
        escalation budget of the softmax-routing baselines B2/B3."""
        self.cfg, self.cond_name, self.cond = cfg, cond, CONDITIONS[cond]
        self.cumulative = self.cond.get("loop") == "cumulative"
        self.stream, self.val, self.test_eval = stream, val, test_eval
        self.student, self.teacher, self.boxes, self.ref = student, teacher, detector_boxes, ref_masks
        self.distill_fn, self.evaluate_fn = distill_fn, evaluate_fn
        self.out = Path(out_dir) / f"{cond}_seed{seed}"; self.out.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed); self.seed = seed
        self.random_rate = random_rate
        self.log = log
        self.classes = list(cfg.classes); self.nt = self.classes.index(cfg.no_tumour_class)
        self.gcfg = GateConfig.from_cfg(cfg.gate, use_yolo=self.cond["yolo"], use_xai=self.cond["xai"])
        self.tau_cloud = cfg.gate.tau_cloud
        self.train_boxes = train_boxes or []      # v2.0: [(boxes, y)] of the training partitions (location model)

    # ----------------------------------------------------------- calibration on validation
    def _calibrate(self):
        vp, vy = [p for p, _ in self.val], np.array([y for _, y in self.val])
        m, z = self.evaluate_fn(vp, vy, 1.0)
        self.T = fit_temperature(z, vy) if self.cond["recal"] or self.cond["routing"] != "none" else 1.0
        p = softmax(z, self.T)
        correct = p.argmax(1) == vy
        pos = vy != self.nt
        if self.cond["routing"] == "softmax":
            if self.random_rate is not None:          # matched escalation budget (Section 3.7.2)
                self.tau = float(np.quantile(p.max(1), self.random_rate))
            else:
                self.tau = fit_tau_for_fnr(p.max(1), pos, correct, self.cfg.gate.target_fnr)
        elif self.cond["routing"] in ("composite", "random"):
            g_cfg = self.cfg.gate
            self.tau = None                                   # random routing (B5) does not use tau_gate
            if self.cond["routing"] == "composite":
                gr = None
                if self.cond.get("gate") == "v2":
                    key = f"{self.cond_name}_seed{self.seed}"
                    gr = Grounding(len(self.classes), float(self.cfg.data.image_size))
                    if self.train_boxes:
                        gr.fit_location([b for b, _ in self.train_boxes], [y for _, y in self.train_boxes], self.seed)
                    GROUNDINGS[key] = gr; self.gcfg.grounding = key
                gb, ay, ax, raw, yh = [], [], [], [], []
                for i, (path, y) in enumerate(self.val):
                    cam = self.student.gradcam(path, self.T)[1] if (self.gcfg.use_yolo or self.gcfg.use_xai) else None
                    r = composite_gate(p[i], cam, self.boxes.get(path, []), self.ref(path), self.nt,
                                       GateConfig(**{**asdict(self.gcfg), "tau_gate": 0.0, "use_overrides": False}))
                    gb.append(r.g_base); ay.append(0.0 if r.a_yolo is None else r.a_yolo)
                    ax.append(np.nan if r.a_xai is None else r.a_xai)
                    raw.append(r.extras.get("a_xai_raw")); yh.append(r.y_hat)
                if gr is not None and self.gcfg.use_xai:
                    # class-relative scaling fitted on validation, then applied to the same raw values
                    gr.fit_xai([0.0 if v is None else v for v in raw], yh)
                    ax = [gr.xai_rel(0.0 if v is None else v, c) for v, c in zip(raw, yh)]
                gb, ay, ax = np.array(gb), np.array(ay), np.array(ax, dtype=float)
                self.val_terms = dict(g_base=gb, a_yolo=ay, a_xai=ax, correct=correct, y_hat=np.array(yh))
                ax0 = np.nan_to_num(ax, nan=0.0)
                # gate weights selected on validation by minimum AURC (v1.6); ablated terms stay at 0
                if g_cfg.get("tune_weights", True):
                    from .metrics import risk_coverage
                    grid = np.round(np.arange(0, 0.51, 0.05), 2)
                    best = None
                    for w2 in (grid if self.gcfg.use_yolo else [0.0]):
                        for w3 in (grid if self.gcfg.use_xai else [0.0]):
                            w1 = 1.0 - w2 - w3
                            if w1 < 0.3:
                                continue
                            aurc = risk_coverage(w1 * gb + w2 * ay + w3 * ax0, correct)[2]
                            if best is None or aurc < best[0] - 1e-9:
                                best = (aurc, w1, w2, w3)
                    _, self.gcfg.w1, self.gcfg.w2, self.gcfg.w3 = best
                    self.log(f"[{self.cond_name}] gate weights (val AURC {best[0]:.4f}): "
                             f"w1={best[1]:.2f} w2={best[2]:.2f} w3={best[3]:.2f}")
                wsum = self.gcfg.w1 + (self.gcfg.w2 if self.gcfg.use_yolo else 0) + (self.gcfg.w3 if self.gcfg.use_xai else 0)
                gs = (self.gcfg.w1 * gb + (self.gcfg.w2 * ay if self.gcfg.use_yolo else 0)
                      + (self.gcfg.w3 * ax0 if self.gcfg.use_xai else 0)) / wsum
                if g_cfg.get("tau_mode", "budget") == "budget":
                    # tau_gate initialised to an escalation budget on validation, then adapted by Eq. 3.19-3.20
                    self.tau = float(np.quantile(gs, g_cfg.get("escalation_budget", 0.30)))
                else:
                    self.tau = fit_tau_for_fnr(gs, pos, correct, g_cfg.target_fnr)
                # Grounding-override threshold: q-th quantile of A_XAI among CORRECT positive validation cases
                ok = correct & pos & ~np.isnan(ax)
                if self.gcfg.use_xai and ok.sum() >= 10:
                    q = float(np.quantile(ax[ok], g_cfg.get("xai_min_quantile", 0.10)))
                    self.gcfg.tau_xai_min = min(self.gcfg.tau_xai_min, q)
                    self.log(f"[{self.cond_name}] tau_xai_min={self.gcfg.tau_xai_min:.4f}")
            # cloud-trust threshold tau_cloud from the teacher's validation outputs (v1.6): the smallest
            # threshold at which the teacher's error among trusted validation cases is <= cloud_error_target
            gc, okc = [], []
            ccfg = GateConfig(**{**asdict(self.gcfg), "tau_gate": 0.0, "use_overrides": False})
            for path, y in self.val:
                tc = self.teacher.cached(path); pc = softmax(tc["z"], self.teacher.T)
                rc = composite_gate(pc, tc.get("cam"), self.boxes.get(path, []), self.ref(path), self.nt, ccfg)
                gc.append(rc.g); okc.append(rc.y_hat == y)
            gc, okc = np.array(gc), np.array(okc)
            self.tau_cloud = fit_tau_for_fnr(gc, np.ones_like(okc, bool), okc, g_cfg.get("cloud_error_target", 0.05),
                                             grid=np.linspace(0.0, 0.98, 99))
            self.log(f"[{self.cond_name}] tau_cloud={self.tau_cloud:.3f} (teacher val acc among trusted "
                     f"{okc[gc >= self.tau_cloud].mean():.3f}, trusted share {(gc >= self.tau_cloud).mean():.3f})")
        else:
            self.tau = None
        self.gcfg.tau_gate = self.tau if self.tau is not None else self.gcfg.tau_gate
        self.val_metrics = m
        self.log(f"[{self.cond_name}] calibrated T={self.T:.3f} tau={self.tau}")

    # ----------------------------------------------------------- main loop
    def run(self):
        t0 = time.time(); self._t0 = t0
        cfg = self.cfg
        self._calibrate()
        if self.cond["routing"] == "offline":
            return self._run_offline(t0)
        tracker = AgreementTracker(cfg.clinician.window)
        buffer, audits, logs, rounds = [], [], [], []
        budget, reviews, cloud_calls, rnd = 0.0, 0, 0, 0
        rounds.append(self._round_eval(rnd, cloud_calls, reviews, 0))
        for i, (path, y) in enumerate(self.stream):
            budget += cfg.clinician.budget_per_100 / 100.0
            boxes = self.boxes.get(path, [])
            # Grad-CAM only when it can change the decision: a grounding term with non-zero weight or an
            # active grounding override (v1.7 speed-up; identical decisions otherwise)
            need_cam = self.cond["routing"] == "composite" and (
                (self.gcfg.use_yolo and self.gcfg.w2 > 0) or (self.gcfg.use_xai and (self.gcfg.w3 > 0 or self.gcfg.tau_xai_min > 0)))
            if need_cam:
                z, cam = self.student.gradcam(path, self.T)
            else:
                z, cam = self.student.logits(path), None
            p = softmax(z, self.T)
            r = composite_gate(p, cam, boxes, self.ref(path), self.nt, self.gcfg)
            routing = self.cond["routing"]
            if routing == "none":
                decision = "accept"
            elif routing == "all":
                decision = "escalate"
            elif routing == "softmax":
                decision = "accept" if p.max() >= self.tau else "escalate"
            elif routing == "random":
                decision = "escalate" if self.rng.random() < (self.random_rate or 0.2) else "accept"
            else:
                decision = r.decision
            c = CaseLog(i, path, int(y), r.y_hat, r.conf, r.g, r.a_yolo, r.a_xai, r.override, decision,
                        round=rnd, tau_gate=self.gcfg.tau_gate, T=self.T)
            if decision == "accept":
                c.y_final = r.y_hat
                if self.cond["recal"] and self.rng.random() < cfg.recalibration.audit_fraction:
                    audits.append(r.y_hat == y)
            else:
                cloud_calls += 1
                tc = self.teacher.cached(path)
                pc = softmax(tc["z"], self.teacher.T)
                rc = composite_gate(pc, tc.get("cam"), boxes, self.ref(path), self.nt,
                                    GateConfig(**{**asdict(self.gcfg), "tau_gate": 0.0}))
                c.y_cloud, c.g_cloud = rc.y_hat, rc.g
                positive = rc.y_hat != self.nt
                route, reasons = cloud_route(r, rc, p, pc, positive and cfg.clinician.review_plans,
                                             self.tau_cloud, cfg.gate.kl_delta)
                c.reasons = reasons
                ex = None
                if self.cond["clinician"] and route == "clinician" and budget >= 1.0:
                    budget -= 1.0; reviews += 1; c.reviewed = True; c.route = "clinician"
                    y_star = int(y)                                  # simulated clinician = ground truth
                    tracker.add(rc.y_hat == y_star)
                    c.y_final = y_star
                    ex = dict(path=path, y_star=y_star, z_cloud=list(map(float, tc["z"])),
                              weight=1.0, teacher_wrong=rc.y_hat != y_star)
                else:
                    c.route = "cloud"
                    c.y_final = rc.y_hat
                    trusted = rc.g >= self.tau_cloud or not self.cond["clinician"]
                    if self.cond["distill"] and trusted:
                        w = tracker.value if self.cond["clinician"] else 1.0
                        ex = dict(path=path, y_star=rc.y_hat, z_cloud=list(map(float, tc["z"])),
                                  weight=w, teacher_wrong=False)
                if ex is not None and self.cond["distill"]:
                    yname = self.classes[ex["y_star"]]
                    # plan supervision only for clinician-verified positive cases (Eq. 3.16: verified plans)
                    ex["plan"] = (tc["plan_fn"](yname) if (c.reviewed and yname != cfg.no_tumour_class) else None)
                    buffer.append(ex); c.buffered = True
            logs.append(c)
            trigger = cfg.distill.get("round_labels", 150) if self.cumulative else cfg.distill.batch_size
            last = i == len(self.stream) - 1
            if self.cond["distill"] and (len(buffer) >= trigger or (self.cumulative and last and buffer)):
                rnd += 1
                self.log(f"[{self.cond_name}] case {i+1}/{len(self.stream)}: distillation round {rnd} "
                         f"({len(buffer)} new labels)")
                if self.cumulative:
                    res = self.distill_fn(buffer, seed=self.seed * 1000 + rnd, cumulative=True)
                else:
                    res = self.distill_fn(buffer, seed=self.seed * 1000 + rnd)
                buffer = []
                if self.cond["recal"]:
                    self._recalibrate(audits); audits = []
                rounds.append(self._round_eval(rnd, cloud_calls, reviews, i + 1, res))
        rounds.append(self._round_eval(rnd, cloud_calls, reviews, len(self.stream), final=True))
        return self._finish(logs, rounds, t0, cloud_calls, reviews)

    def _recalibrate(self, audits):
        rc = self.cfg.recalibration
        vp, vy = [p for p, _ in self.val], np.array([y for _, y in self.val])
        _, z = self.evaluate_fn(vp, vy, 1.0)
        self.T = fit_temperature(z, vy)                                   # Eq. 3.21
        if audits:
            n_wrong = int(len(audits) - sum(audits))
            self.gcfg.tau_gate, far = update_threshold(self.gcfg.tau_gate, n_wrong, len(audits),
                                                       rc.far_target, rc.eta_tau, rc.tau_min, rc.tau_max)
            self.log(f"   recalibrated T={self.T:.3f}, FAR_obs={far:.3f}, tau_gate={self.gcfg.tau_gate:.3f}")

    def _round_eval(self, rnd, cloud_calls, reviews, n_seen, res=None, final=False):
        tp, ty = [p for p, _ in self.test_eval], np.array([y for _, y in self.test_eval])
        m, _ = self.evaluate_fn(tp, ty, self.T)
        row = dict(round=rnd, cases_seen=n_seen, cloud_calls=cloud_calls, reviews=reviews,
                   student_acc=m["accuracy"], student_f1=m["macro_f1"], student_sens=m["tumour_sensitivity"],
                   student_ece=m.get("ece"), tau_gate=self.gcfg.tau_gate, T=getattr(self, "T", 1.0),
                   promoted=None if res is None else res.get("promoted"),
                   train_size=None if res is None else res.get("train_size"),
                   p_worse=None if res is None else res.get("p_worse"),
                   minutes=(time.time() - self._t0) / 60 if hasattr(self, "_t0") else None, final=final)
        return row

    def _run_offline(self, t0):
        """B4: distil once on teacher labels for the whole stream, then accept everything."""
        buf = []
        for path, y in self.stream:
            tc = self.teacher.cached(path); pc = softmax(tc["z"], self.teacher.T); yc = int(pc.argmax())
            yname = self.classes[yc]
            buf.append(dict(path=path, y_star=yc, z_cloud=list(map(float, tc["z"])), weight=1.0, teacher_wrong=False,
                            plan=None))              # no clinician -> no verified plans
        rounds = [self._round_eval(0, 0, 0, 0)]
        self.distill_fn(buf, seed=self.seed * 1000)  # one offline distillation job on all stream cases
        rounds.append(self._round_eval(1, len(self.stream), 0, len(self.stream), final=True))
        return self._finish([], rounds, t0, len(self.stream), 0)

    def _finish(self, logs, rounds, t0, cloud_calls, reviews):
        import pandas as pd
        n = max(len(self.stream), 1)
        df = pd.DataFrame([asdict(c) for c in logs]) if logs else pd.DataFrame()
        df.to_csv(self.out / "cases.csv", index=False)
        pd.DataFrame(rounds).to_csv(self.out / "rounds.csv", index=False)
        summary = dict(condition=self.cond_name, seed=self.seed, n_stream=len(self.stream),
                       cloud_calls=cloud_calls, escalation_rate=cloud_calls / n,
                       reviews_per_100=100 * reviews / n, final=rounds[-1], minutes=(time.time() - t0) / 60)
        if len(df):
            sys_acc = float((df.y_final == df.y_true).mean())
            pos = df.y_true != self.nt
            summary.update(system_accuracy=sys_acc,
                           system_tumour_sensitivity=float((df.y_final[pos] != self.nt).mean()) if pos.any() else None,
                           local_accept_far=float((df[df.decision == "accept"].y_local !=
                                                   df[df.decision == "accept"].y_true).mean())
                           if (df.decision == "accept").any() else None)
        g = self.gcfg
        summary["gate"] = dict(w1=g.w1, w2=g.w2, w3=g.w3, tau_gate=g.tau_gate, tau_xai_min=g.tau_xai_min,
                               tau_cloud=self.tau_cloud, T=getattr(self, "T", 1.0), grounding=g.grounding,
                               use_yolo=g.use_yolo, use_xai=g.use_xai)
        if g.grounding and g.grounding in GROUNDINGS:            # kept for the demo application
            import pickle
            (self.out / "grounding.pkl").write_bytes(pickle.dumps(GROUNDINGS[g.grounding]))
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        self.log(f"[{self.cond_name}] done: {json.dumps({k: summary[k] for k in ('escalation_rate','reviews_per_100')})}")
        return summary
