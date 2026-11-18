"""GPU runtime: wires the real models into the simulator. Intended for Kaggle (2 x T4):
student on cuda:0, teacher on cuda:1."""
from __future__ import annotations
import json, os
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import load_config
from .data import load_image as _load_image, load_mask
from .gate import softmax
from .calibration import fit_temperature
from .train import train_examples, evaluate, distill_round, distill_round_cumulative
from .simulate import Simulator

GUIDE_DIR = Path(__file__).resolve().parent.parent / "guidelines"


def work(cfg) -> Path:
    p = Path(cfg.paths.work_dir); p.mkdir(parents=True, exist_ok=True); return p


def manifest(cfg) -> pd.DataFrame:
    return pd.read_csv(work(cfg) / "manifest.csv")


def make_loader(cfg):
    size = cfg.data.image_size

    @lru_cache(maxsize=4096)
    def load(path):
        return _load_image(path, size)
    return load


def part(man, name, order=False):
    d = man[man.partition == name]
    if order:
        d = d.sort_values("stream_order")
    return list(zip(d.path, d.y.astype(int)))


def guidance(class_name: str) -> str:
    f = GUIDE_DIR / f"{class_name}.txt"
    return f.read_text() if f.exists() else ""


# ------------------------------------------------------------------ detector cache
def detector_cache(cfg, man, weights) -> dict:
    out = work(cfg) / "yolo_boxes.json"
    if out.exists():
        return json.loads(out.read_text())
    from .detector import TumourDetector
    det = TumourDetector(weights, cfg.yolo.conf_keep, cfg.yolo.imgsz, device=0)
    load = make_loader(cfg)
    paths = man[man.partition.isin(["validation", "stream", "test"])].path.tolist()
    boxes = {p: det(load(p)) for p in paths}
    out.write_text(json.dumps(boxes))
    return boxes


def train_box_cache(cfg, man, weights) -> list:
    """v2.0: YOLO boxes on the TRAINING partitions (seed + teacher pool) with their ground-truth labels,
    used only to fit the lesion-location model of the A_YOLO term. Never validation, stream or test."""
    out = work(cfg) / "yolo_boxes_train.json"
    d = man[man.partition.isin(["seed", "teacher_pool"])]
    if out.exists():
        boxes = json.loads(out.read_text())
    else:
        from .detector import TumourDetector
        det = TumourDetector(weights, cfg.yolo.conf_keep, cfg.yolo.imgsz, device=0)
        load = make_loader(cfg)
        boxes = {p: det(load(p)) for p in d.path}
        out.write_text(json.dumps(boxes))
    return [(boxes[p], int(y)) for p, y in zip(d.path, d.y) if p in boxes]


def ref_mask_fn(cfg, man):
    mp = dict(zip(man.path, man.mask_path)) if "mask_path" in man else {}
    size = cfg.data.image_size

    @lru_cache(maxsize=4096)
    def f(path):
        return load_mask(mp.get(path), size)
    return f


# ------------------------------------------------------------------ models
def cam_choice(cfg) -> str | None:
    f = work(cfg) / "xai_choice.json"
    return json.loads(f.read_text())["variant"] if f.exists() else None


def _apply_cam_choice(cfg, vlm):
    v = cam_choice(cfg)
    if v:
        vlm.cam_variant = v
    return vlm


def load_student(cfg, adapter=None, device=0):
    from .vlm import VLM
    return _apply_cam_choice(cfg, VLM(cfg.models.student, cfg.classes, cfg.models.dtype, load_4bit=False,
               device=device, image_size=cfg.data.image_size, lora=dict(cfg.lora) if adapter is None else None,
               adapter_path=adapter))


def load_teacher(cfg, adapter=None, device=1, trainable=False):
    from .vlm import VLM
    lora = dict(cfg.lora) if (trainable and adapter is None) else None
    return _apply_cam_choice(cfg, VLM(cfg.models.teacher, cfg.classes, cfg.models.dtype,
               load_4bit=cfg.models.teacher_4bit, device=device, image_size=cfg.data.image_size, lora=lora,
               adapter_path=adapter))


def choose_cam_variant(cfg, adapter=None, n=60, device=0, log=print):
    """Pick the attribution variant whose maps best overlap the expert tumour masks on VALIDATION images
    (never test). Score = mean A_XAI (Eq. 3.9) with the gate's own top-mass mask. Also reports the
    pointing-game hit rate and the chance level (mean mask area). Writes xai_choice.json + table + figure."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .gate import cam_to_mask, a_xai
    man = manifest(cfg); load = make_loader(cfg); refm = ref_mask_fn(cfg, man)
    d = man[(man.partition == "validation") & man.mask_path.notna()].head(n)
    if adapter is None:
        adapter = str(work(cfg) / "adapters" / "student_seed")
    st = load_student(cfg, adapter=adapter, device=device)
    rows, example = [], None
    for path, y in zip(d.path, d.y.astype(int)):
        ref = refm(path).astype(bool)
        if ref.sum() == 0:
            continue
        z, maps = st.cam_variants(load(path))
        for k, m in maps.items():
            yy, xx = np.unravel_index(np.argmax(m), m.shape)
            rows.append(dict(variant=k, a_xai=a_xai(cam_to_mask(m, cfg.gate.cam_top_mass), ref, True),
                             pointing=float(ref[yy, xx]), chance=float(ref.mean())))
        if example is None:
            example = (path, ref, maps, int(np.argmax(z)), y)
    t = pd.DataFrame(rows).groupby("variant").mean().sort_values("a_xai", ascending=False)
    t["n_images"] = len(rows) // max(1, t.shape[0])
    best = t.index[0]
    (work(cfg) / "xai_choice.json").write_text(json.dumps({"variant": best, "table": t.round(4).to_dict()}, indent=2))
    t.to_csv(work(cfg) / "table_xai_variants.csv")
    path, ref, maps, yh, y = example
    ks = list(t.index)
    fig, ax = plt.subplots(1, len(ks) + 1, figsize=(3 * (len(ks) + 1), 3.2))
    ax[0].imshow(load(path)); ax[0].contour(ref, colors="lime", linewidths=1); ax[0].set_title("image + expert mask")
    for a, k in zip(ax[1:], ks):
        a.imshow(load(path)); a.imshow(maps[k], alpha=.45, cmap="jet"); a.contour(ref, colors="lime", linewidths=1)
        a.set_title(f"{k}\nA_XAI={t.loc[k, 'a_xai']:.2f}", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"true={cfg.classes[y]}  pred={cfg.classes[yh]}")
    fig.tight_layout(); fig.savefig(work(cfg) / "figure_xai_variants.png", dpi=120)
    log(t.round(3).to_string())
    log(f"chosen attribution variant: {best}")
    del st; torch.cuda.empty_cache()
    return best, t


def _balanced(cfg, ex):
    """Inverse-class-frequency weights, mean 1 (cfg.class_balanced)."""
    if not cfg.get("class_balanced", False):
        return ex
    ys = np.array([e["y_star"] for e in ex]); C = len(cfg.classes)
    cnt = np.bincount(ys, minlength=C).astype(float)
    w = len(ys) / (C * np.maximum(cnt, 1))
    for e in ex:
        e["weight"] = float(w[e["y_star"]])
    return ex


def seed_finetune(cfg, log=print) -> str:
    """Increment 2: fine-tune the student on the seed partition (labels only, no plans)."""
    man = manifest(cfg); load = make_loader(cfg)
    student = load_student(cfg)
    ex = _balanced(cfg, [dict(path=p, y_star=y, z_cloud=None, plan=None, weight=1.0, teacher_wrong=False)
                         for p, y in part(man, "seed")])
    s = cfg.seed_finetune
    train_examples(student, ex, load, list(cfg.classes), epochs=s.epochs, lr=s.lr, grad_accum=s.grad_accum, log=log)
    out = str(work(cfg) / "adapters" / "student_seed"); student.save_adapter(out)
    vp, vy = zip(*part(man, "validation"))
    m, _ = evaluate(student, list(vp), np.array(vy), load, no_tumour_idx=list(cfg.classes).index("no_tumour"))
    (work(cfg) / "student_seed_val.json").write_text(json.dumps(m, indent=2))
    log(f"seed student validation: {m}")
    return out


def teacher_adapt(cfg, log=print) -> str:
    """Increment 3: QLoRA adaptation of the cloud teacher on the teacher pool (central labelled data)."""
    man = manifest(cfg); load = make_loader(cfg)
    teacher = load_teacher(cfg, trainable=True, device=0)
    ex = _balanced(cfg, [dict(path=p, y_star=y, z_cloud=None, plan=None, weight=1.0, teacher_wrong=False)
                         for p, y in part(man, "teacher_pool")])
    s = cfg.teacher_adapt
    train_examples(teacher, ex, load, list(cfg.classes), epochs=s.epochs, lr=s.lr, batch_size=1,
                   grad_accum=s.grad_accum, log=log)
    out = str(work(cfg) / "adapters" / "teacher"); teacher.save_adapter(out)
    vp, vy = zip(*part(man, "validation"))
    m, _ = evaluate(teacher, list(vp), np.array(vy), load, no_tumour_idx=list(cfg.classes).index("no_tumour"), batch_size=4)
    (work(cfg) / "teacher_val.json").write_text(json.dumps(m, indent=2))
    log(f"teacher validation: {m}")
    return out


class TeacherCache:
    """Cloud teacher outputs (logits, Grad-CAM, plans), computed once and reused by every condition."""
    def __init__(self, cfg, teacher, load, cam_res=112):
        self.cfg, self.t, self.load, self.cam_res = cfg, teacher, load, cam_res
        self.dir = work(cfg) / "teacher_cache"; self.dir.mkdir(exist_ok=True)
        self.plan_file = self.dir / "plans.json"
        self.plans = json.loads(self.plan_file.read_text()) if self.plan_file.exists() else {}
        self.T = 1.0

    def _key(self, path):
        return Path(path).parent.name + "__" + Path(path).stem

    def precompute(self, paths, log=print):
        for i, p in enumerate(paths):
            f = self.dir / f"{self._key(p)}.npz"
            if f.exists():
                continue
            z, cam = self.t.gradcam(self.load(p))
            small = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((self.cam_res,) * 2, Image.BILINEAR))
            np.savez_compressed(f, z=z, cam=small)
            if (i + 1) % 100 == 0:
                log(f"  teacher cache {i + 1}/{len(paths)}")

    def fit_T(self, val):
        z = np.stack([self.cached(p)["z"] for p, _ in val]); y = np.array([y for _, y in val])
        self.T = fit_temperature(z, y)
        return self.T

    def cached(self, path):
        d = np.load(self.dir / f"{self._key(path)}.npz")
        size = self.cfg.data.image_size
        cam = np.asarray(Image.fromarray(d["cam"]).resize((size, size), Image.BILINEAR)).astype(np.float32) / 255.0
        return {"z": d["z"], "cam": cam, "plan_fn": lambda yname, p=path: self.plan(p, yname)}

    def plan(self, path, yname):
        k = f"{self._key(path)}|{yname}"
        if k not in self.plans:
            self.plans[k] = self.t.generate_plan(self.load(path), yname, guidance(yname),
                                                 self.cfg.models.max_new_tokens_plan)
            self.plan_file.write_text(json.dumps(self.plans))
        return self.plans[k]


class StudentHandle:
    def __init__(self, cfg, student, load, seed_adapter_state, replay_pool, val):
        self.cfg, self.s, self.load = cfg, student, load
        self.seed_state, self.replay, self.val = seed_adapter_state, replay_pool, val
        self.classes = list(cfg.classes); self.nt = self.classes.index("no_tumour")
        self.cur_val = None
        self.cur_correct = None
        self.memory = []          # v1.9: every verified label collected so far in the current run

    def reset(self):
        self.s.load_adapter_state(self.seed_state); self.cur_val = None
        self.cur_correct = None; self.memory = []

    def logits(self, path):
        return self.s.class_logits([self.load(path)])[0]

    def gradcam(self, path, T=1.0):
        return self.s.gradcam(self.load(path), T)

    def evaluate(self, paths, y, T=1.0):
        return evaluate(self.s, paths, y, self.load, T=T, no_tumour_idx=self.nt)

    def distill(self, buffer, seed=0, cumulative=False):
        vp, vy = [p for p, _ in self.val], np.array([y for _, y in self.val])
        if cumulative:
            self.memory.extend(buffer)
            res = distill_round_cumulative(self.s, self.memory, self.replay, vp, vy, self.load, self.classes,
                                           self.cfg, current_val=self.cur_val, current_correct=self.cur_correct,
                                           seed=seed)
            self.cur_val, self.cur_correct = res["val"], res["correct"]
            res = {k: v for k, v in res.items() if k != "correct"}
            return res
        res = distill_round(self.s, buffer, self.replay, vp, vy, self.load, self.classes, self.cfg,
                            current_val=self.cur_val, seed=seed)
        self.cur_val = res["val"]
        return res


def build_runtime(cfg, student_adapter, teacher_adapter, yolo_weights, log=print):
    man = manifest(cfg); load = make_loader(cfg)
    boxes = detector_cache(cfg, man, yolo_weights)
    train_boxes = train_box_cache(cfg, man, yolo_weights)
    refs = ref_mask_fn(cfg, man)
    student = load_student(cfg, adapter=student_adapter, device=0)
    teacher = load_teacher(cfg, adapter=teacher_adapter, device=1 if torch.cuda.device_count() > 1 else 0)
    tc = TeacherCache(cfg, teacher, load)
    stream, val = part(man, "stream", order=True), part(man, "validation")
    test = part(man, "test")
    log("precomputing teacher outputs for validation + stream ...")
    tc.precompute([p for p, _ in val + stream], log=log)
    tc.fit_T(val)
    replay = [dict(path=p, y_star=y, z_cloud=None, plan=None, weight=1.0, teacher_wrong=False) for p, y in part(man, "seed")]
    sh = StudentHandle(cfg, student, load, student.adapter_state(), replay, val)
    rng = np.random.default_rng(cfg.seed)
    k = min(cfg.data.round_eval_size, len(test))
    test_eval = [test[i] for i in sorted(rng.choice(len(test), k, replace=False))]
    return dict(man=man, load=load, boxes=boxes, refs=refs, teacher=tc, student=sh,
                stream=stream, val=val, test=test, test_eval=test_eval, train_boxes=train_boxes)


def teacher_test_eval(cfg, rt, log=print) -> dict:
    """Cloud teacher on the full test set (logits only; temperature from validation). Cached to JSON."""
    f = work(cfg) / "teacher_test.json"
    if f.exists():
        return json.loads(f.read_text())
    tc = rt["teacher"]; nt = list(cfg.classes).index("no_tumour")
    tp, ty = [p for p, _ in rt["test"]], np.array([y for _, y in rt["test"]])
    m, _ = evaluate(tc.t, tp, ty, rt["load"], T=tc.T, no_tumour_idx=nt, batch_size=4)
    f.write_text(json.dumps(m, indent=2)); log(f"teacher test: {m}")
    return m


def run_condition(cfg, rt, cond, seed=0, random_rate=None, log=print):
    rt["student"].reset()
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
    stream = list(rt["stream"])
    if seed > 0:      # v2.0: repeated runs see the deployment stream in a different (seeded) order
        import random as _r
        _r.Random(seed).shuffle(stream)
    sim = Simulator(cfg, stream, rt["val"], rt["test_eval"], rt["student"], rt["teacher"], rt["boxes"],
                    rt["refs"], rt["student"].distill, rt["student"].evaluate, work(cfg) / "runs",
                    cond=cond, seed=seed, random_rate=random_rate, log=log, train_boxes=rt.get("train_boxes"))
    summary = sim.run()
    # full test-set evaluation of the final student
    tp, ty = [p for p, _ in rt["test"]], np.array([y for _, y in rt["test"]])
    m, z = rt["student"].evaluate(tp, ty, sim.T)
    summary["final_student_test"] = m
    if torch.cuda.is_available():                       # NFR2: peak GPU memory of this run (GB)
        summary["peak_gpu_gb"] = {f"gpu{d}": round(torch.cuda.max_memory_allocated(d) / 1e9, 2)
                                  for d in range(torch.cuda.device_count())}
    (sim.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.save(sim.out / "test_logits.npy", z)
    rt["student"].s.save_adapter(str(sim.out / "adapter"))
    return summary


def gate_quality(cfg, rt, log=print) -> pd.DataFrame:
    """Table 4.7: compare gate variants on the validation set with the seed student."""
    from .gate import composite_gate, GateConfig
    from .metrics import risk_coverage, far_at_escalation
    st = rt["student"]; st.reset()
    vp, vy = [p for p, _ in rt["val"]], np.array([y for _, y in rt["val"]])
    _, z = st.evaluate(vp, vy, 1.0); T = fit_temperature(z, vy); p = softmax(z, T)
    nt = list(cfg.classes).index("no_tumour")
    rows = {k: [] for k in ["softmax", "g_base", "g_base+yolo", "g_base+xai", "composite"]}
    over, comp = [], []
    for i, path in enumerate(vp):
        _, cam = st.gradcam(path, T)
        b, ref = rt["boxes"].get(path, []), rt["refs"](path)
        base = GateConfig.from_cfg(cfg.gate, tau_gate=0.0)
        rows["softmax"].append(p[i].max())
        rows["g_base"].append(composite_gate(p[i], None, b, ref, nt, GateConfig(**{**base.__dict__, "use_yolo": False, "use_xai": False})).g)
        rows["g_base+yolo"].append(composite_gate(p[i], cam, b, ref, nt, GateConfig(**{**base.__dict__, "use_xai": False})).g)
        rows["g_base+xai"].append(composite_gate(p[i], cam, b, ref, nt, GateConfig(**{**base.__dict__, "use_yolo": False})).g)
        r = composite_gate(p[i], cam, b, ref, nt, base); rows["composite"].append(r.g); over.append(r.override is not None)
        comp.append((r.g_base, r.a_yolo or 0.0, r.a_xai or 0.0))
    correct = p.argmax(1) == vy; pos = vy != nt
    # validation-tuned weights (same grid as the simulator, v1.6); in-sample on validation
    comp = np.array(comp); best = None
    for w2 in np.round(np.arange(0, 0.51, 0.05), 2):
        for w3 in np.round(np.arange(0, 0.51, 0.05), 2):
            w1 = 1 - w2 - w3
            if w1 < 0.3:
                continue
            a = risk_coverage(comp @ np.array([w1, w2, w3]), correct)[2]
            if best is None or a < best[0] - 1e-9:
                best = (a, w1, w2, w3)
    tuned_name = f"composite_tuned(w={best[1]:.2f},{best[2]:.2f},{best[3]:.2f})"
    rows[tuned_name] = list(comp @ np.array(best[1:]))
    out = []
    for k, s in rows.items():
        s = np.array(s)
        if k == "composite":
            s = np.where(over, -1.0, s)              # overrides always escalate
        _, _, aurc = risk_coverage(s, correct)
        acc_idx = np.argsort(-s)[:int(round(0.8 * len(s)))]
        missed = int(((p.argmax(1)[acc_idx] == nt) & pos[acc_idx]).sum())
        out.append(dict(gate=k, aurc=aurc, far_at_20pct_escalation=far_at_escalation(s, correct, 0.2),
                        missed_tumours_accepted=missed))
    df = pd.DataFrame(out); df.to_csv(work(cfg) / "gate_quality.csv", index=False)
    return df


def results_dump(cfg, log=print):
    """Print every finished run compactly (so results can be read from the notebook output alone)."""
    rd = work(cfg) / "runs"
    for f in sorted(rd.glob("*/summary.json")):
        s = json.loads(f.read_text())
        ft = s.get("final_student_test", {})
        keys = ("escalation_rate", "cloud_calls", "reviews_per_100", "system_accuracy",
                "system_tumour_sensitivity", "local_accept_far", "minutes", "peak_gpu_gb", "gate")
        log(f"== {f.parent.name}: " + ", ".join(f"{k}={s.get(k)}" for k in keys))
        log("   final student on test: " + json.dumps({k: round(v, 4) for k, v in ft.items() if isinstance(v, float)}))
        rf = f.parent / "rounds.csv"
        if rf.exists():
            log(pd.read_csv(rf).round(4).to_string(index=False))
    for name in ("teacher_test.json", "table_gate_quality.csv", "xai_choice.json"):
        f = work(cfg) / name
        if f.exists():
            log(f"== {name}\n" + f.read_text()[:3000])


# ------------------------------------------------------------------ v2.0 gate study (RQ1, validation only)
def _aurc_far(score, correct, frac=0.2):
    from .metrics import risk_coverage, far_at_escalation
    return risk_coverage(score, correct)[2], far_at_escalation(score, correct, frac)


def gate_study_v2(cfg, rt, folds=5, n_boot=2000, log=print) -> pd.DataFrame:
    """Compare the original (v1) and class-aware (v2) grounding terms on the VALIDATION set with the seed
    student. Weights are chosen by k-fold cross-validation (chosen on k-1 folds, scored on the held-out fold),
    so the AURC of the tuned gate is out-of-sample. A paired bootstrap gives the probability that the tuned
    v2 gate is NOT better than G_base. Writes table_gate_study_v2.csv."""
    from .gate import (composite_gate, GateConfig, Grounding, xai_raw, cam_to_mask, boxes_to_mask)
    from .metrics import risk_coverage
    st = rt["student"]; st.reset()
    vp, vy = [p for p, _ in rt["val"]], np.array([y for _, y in rt["val"]])
    _, z = st.evaluate(vp, vy, 1.0); T = fit_temperature(z, vy); p = softmax(z, T)
    nt = list(cfg.classes).index("no_tumour"); yh = p.argmax(1); correct = yh == vy
    gr = Grounding(len(cfg.classes), float(cfg.data.image_size))
    tb = rt["train_boxes"]; gr.fit_location([b for b, _ in tb], [y for _, y in tb], 0)
    base = GateConfig.from_cfg(cfg.gate, tau_gate=0.0, use_overrides=False)
    gb, ay1, ax1, ay2, raw = [], [], [], [], []
    for i, path in enumerate(vp):
        _, cam = st.gradcam(path, T)
        b, ref = rt["boxes"].get(path, []), rt["refs"](path)
        r = composite_gate(p[i], cam, b, ref, nt, base)
        gb.append(r.g_base); ay1.append(r.a_yolo); ax1.append(r.a_xai)
        ay2.append(float(gr.location_probs(b)[yh[i]]))
        m = cam_to_mask(cam, cfg.gate.cam_top_mass)
        rr = ref if ref is not None else (boxes_to_mask(b, m.shape) if b else None)
        raw.append(xai_raw(m, rr, b, yh[i] != nt))
    gb, ay1, ax1, ay2, raw = map(np.array, (gb, ay1, ax1, ay2, raw))
    rng = np.random.default_rng(0); fold = rng.permutation(len(vy)) % folds
    # class-relative A_XAI: percentile among the OTHER folds' predictions of the same class (out-of-sample)
    ax2 = np.zeros(len(vy))
    for k in range(folds):
        g = Grounding(len(cfg.classes)).fit_xai(raw[fold != k], yh[fold != k])
        ax2[fold == k] = [g.xai_rel(v, c) for v, c in zip(raw[fold == k], yh[fold == k])]
    grid = np.round(np.arange(0, 0.51, 0.05), 2)

    def tune(idx, a, b_):
        best = None
        for w2 in grid:
            for w3 in grid:
                w1 = 1 - w2 - w3
                if w1 < 0.3:
                    continue
                s = w1 * gb[idx] + w2 * a[idx] + w3 * b_[idx]
                aurc = risk_coverage(s, correct[idx])[2]
                if best is None or aurc < best[0] - 1e-9:
                    best = (aurc, w1, w2, w3)
        return best

    def cv_score(a, b_):
        out = np.zeros(len(vy)); ws = []
        for k in range(folds):
            _, w1, w2, w3 = tune(np.where(fold != k)[0], a, b_)
            ws.append((w1, w2, w3)); i = fold == k
            out[i] = w1 * gb[i] + w2 * a[i] + w3 * b_[i]
        return out, np.mean(ws, axis=0)

    v1_cv, w_v1 = cv_score(ay1, ax1)
    v2_cv, w_v2 = cv_score(ay2, ax2)
    full = tune(np.arange(len(vy)), ay2, ax2)
    rows = {"softmax confidence": p.max(1), "G_base": gb,
            "G_base + A_YOLO v1 (0.25)": 0.75 * gb + 0.25 * ay1, "G_base + A_XAI v1 (0.25)": 0.75 * gb + 0.25 * ax1,
            "G_base + A_YOLO v2 (0.25)": 0.75 * gb + 0.25 * ay2, "G_base + A_XAI v2 (0.25)": 0.75 * gb + 0.25 * ax2,
            f"v1 tuned, {folds}-fold CV (mean w={np.round(w_v1, 2).tolist()})": v1_cv,
            f"v2 tuned, {folds}-fold CV (mean w={np.round(w_v2, 2).tolist()})": v2_cv}
    out = []
    for k, sc in rows.items():
        a, f = _aurc_far(sc, correct)
        out.append(dict(gate=k, aurc=round(a, 4), far_at_20pct_escalation=round(f, 4)))
    # paired bootstrap: P(AURC_v2cv >= AURC_Gbase)
    worse = 0
    for _ in range(n_boot):
        i = rng.integers(0, len(vy), len(vy))
        worse += risk_coverage(v2_cv[i], correct[i])[2] >= risk_coverage(gb[i], correct[i])[2]
    df = pd.DataFrame(out)
    df.to_csv(work(cfg) / "table_gate_study_v2.csv", index=False)
    info = dict(weights_full_validation=dict(w1=full[1], w2=full[2], w3=full[3], aurc_in_sample=full[0]),
                bootstrap_p_v2_not_better_than_gbase=worse / n_boot,
                location_model_train_images=len(tb),
                location_model_val_accuracy=float((np.array([gr.location_probs(rt["boxes"].get(q, [])).argmax()
                                                             for q in vp]) == vy).mean()))
    (work(cfg) / "gate_study_v2.json").write_text(json.dumps(info, indent=2, default=float))
    log(df.to_string(index=False)); log(json.dumps(info, indent=2, default=float))
    return df


def adapter_state_from_dir(path):
    """Read a saved LoRA adapter folder as a state dict usable by VLM.load_adapter_state."""
    from safetensors.torch import load_file
    f = Path(path) / "adapter_model.safetensors"
    return load_file(str(f)) if f.exists() else __import__("torch").load(str(Path(path) / "adapter_model.bin"))


def best_run(cfg, prefer=("P_v3_seed0", "P_v2_seed0", "P_seed0")):
    for r in prefer:
        if (work(cfg) / "runs" / r / "summary.json").exists():
            return r
    return None


# ------------------------------------------------------------------ plan quality (LLM-as-judge, Appendix A.2)
JUDGE_PROMPT = ("You are an experienced neuro-oncologist evaluating an AI-drafted treatment plan against a "
                "reference. Score each criterion from 0 to 10: Correctness (clinically accurate, consistent with "
                "the diagnosis), Completeness (covers findings, investigations, management and referral), "
                "Conciseness (free of irrelevant or repeated content). Return only JSON: "
                '{"correctness": x, "completeness": y, "conciseness": z}\n'
                "Diagnosis: {dx}\nReference plan: {ref}\nAI plan: {cand}")


def _parse_scores(text):
    import re
    out = {}
    for k in ("correctness", "completeness", "conciseness"):
        m = re.search(k + r'"?\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.I)
        out[k] = min(10.0, float(m.group(1))) if m else np.nan
    return out


def plan_quality(cfg, rt, adapters: dict, n=30, log=print) -> pd.DataFrame:
    """Draft plans for n test tumour images from each student adapter and from the teacher, scored 0-10 by the
    teacher used as a text-only judge against the class guideline note (the only reference available).
    Caveat: the judge is the 7B teacher itself (self-preference possible) and the reference is a placeholder
    note, not a clinician-approved plan."""
    nt = "no_tumour"; load = rt["load"]; tc = rt["teacher"]; st = rt["student"]
    test = [(q, y) for q, y in rt["test"] if cfg.classes[y] != nt]
    idx = np.random.default_rng(1).choice(len(test), min(n, len(test)), replace=False)
    sample = [test[i] for i in sorted(idx)]
    rows = []
    sources = dict(adapters); sources["cloud teacher"] = None
    for name, ad in sources.items():
        if ad is not None:
            st.s.load_adapter_state(ad)
        for q, y in sample:
            dx = cfg.classes[y]
            model = tc.t if ad is None else st.s
            plan = model.generate_plan(load(q), dx, guidance(dx), cfg.models.max_new_tokens_plan)
            txt = tc.t.generate_text(JUDGE_PROMPT.replace("{dx}", dx).replace("{ref}", guidance(dx)[:1500])
                                     .replace("{cand}", plan), max_new_tokens=60)
            rows.append(dict(source=name, path=q, diagnosis=dx, plan=plan, judge=txt, **_parse_scores(txt)))
        log(f"  plans scored: {name}")
    st.reset()
    df = pd.DataFrame(rows); df.to_csv(work(cfg) / "plan_quality.csv", index=False)
    summ = df.groupby("source")[["correctness", "completeness", "conciseness"]].agg(["mean", "std"]).round(2)
    summ.to_csv(work(cfg) / "table_plan_quality.csv")
    log(summ.to_string())
    return summ


# ------------------------------------------------------------------ statistics across runs (single-run validity)
def significance(cfg, rt, pairs, n_boot=2000, log=print) -> pd.DataFrame:
    """For each finished run: 95% bootstrap CI of system accuracy (stream) and of final-student test accuracy.
    For each pair (a, b): exact two-sided McNemar test on the SAME 909 test images (final students), and a paired
    bootstrap CI of the system-accuracy difference on the stream (same cases, matched by path)."""
    from math import comb
    rd = work(cfg) / "runs"; ty = np.array([y for _, y in rt["test"]]); rng = np.random.default_rng(0)

    def ci(x):
        x = np.asarray(x, float); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
        return round(100 * x.mean(), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)

    def mcnemar2(a, b):
        b_, c_ = int((a & ~b).sum()), int((~a & b).sum()); n = b_ + c_
        if n == 0:
            return 1.0
        k = min(b_, c_)
        return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))

    runs, rows = {}, []
    for d in sorted(rd.iterdir()):
        if not (d / "summary.json").exists() or not (d / "test_logits.npy").exists():
            continue
        z = np.load(d / "test_logits.npy"); stu = z.argmax(1) == ty
        cases = pd.read_csv(d / "cases.csv") if (d / "cases.csv").exists() and (d / "cases.csv").stat().st_size > 5 else None
        sysc = None if cases is None or not len(cases) else (cases.y_final == cases.y_true).values
        runs[d.name] = (stu, cases)
        r = dict(run=d.name)
        r["student_test_acc"], r["student_lo"], r["student_hi"] = ci(stu)
        if sysc is not None:
            r["system_acc"], r["system_lo"], r["system_hi"] = ci(sysc)
        rows.append(r)
    t1 = pd.DataFrame(rows); log(t1.to_string(index=False))
    out = []
    for a, b in pairs:
        if a not in runs or b not in runs:
            continue
        sa, ca = runs[a]; sb, cb = runs[b]
        row = dict(a=a, b=b, student_diff_pts=round(100 * (sa.mean() - sb.mean()), 1),
                   student_mcnemar_p=round(mcnemar2(sa, sb), 4))
        if ca is not None and cb is not None and len(ca) and len(cb):
            m = ca[["path", "y_final", "y_true"]].merge(cb[["path", "y_final"]], on="path", suffixes=("_a", "_b"))
            da = (m.y_final_a == m.y_true).values.astype(float); db = (m.y_final_b == m.y_true).values.astype(float)
            diff = da - db; bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
            row.update(system_diff_pts=round(100 * diff.mean(), 1), system_diff_lo=round(100 * np.percentile(bs, 2.5), 1),
                       system_diff_hi=round(100 * np.percentile(bs, 97.5), 1))
        out.append(row)
    t2 = pd.DataFrame(out); log(t2.to_string(index=False))
    t1.to_csv(work(cfg) / "table_run_ci.csv", index=False); t2.to_csv(work(cfg) / "table_pairwise_tests.csv", index=False)
    return t2


def pack_results(cfg, out="/kaggle/working/results_small.zip"):
    """Small zip of every result file (no model weights) for download."""
    import zipfile
    W = work(cfg)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        skip = {"teacher_cache", "adapters", "adapter", "yolo_runs", "yolo", "figshare"}
        for f in W.rglob("*"):
            if f.is_file() and f.suffix in (".csv", ".json", ".npy", ".png") and not (skip & set(f.parts)) \
                    and f.stat().st_size < 20e6:
                z.write(f, f.relative_to(W))
    return out
