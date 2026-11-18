"""Untrained vs trained VLMs (v2.1): task performance, calibration, attribution pictures and training summary.

All models are evaluated on the same 412 validation and 909 test images. 'Untrained' means the original
Qwen2-VL weights (LoRA adapters present but with their B matrices set to zero, which is mathematically the
base model). Temperatures are fitted on validation only.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from .gate import softmax
from .calibration import fit_temperature
from .metrics import classification_metrics


def _zero_lora(state):
    import torch
    return {k: (torch.zeros_like(v) if "lora_B" in k else v) for k, v in state.items()}


def calibration_scores(z, y, T):
    p = softmax(z, T); y = np.asarray(y); n = len(y)
    onehot = np.eye(p.shape[1])[y]
    brier = float(((p - onehot) ** 2).sum(1).mean())
    nll = float(-np.log(p[np.arange(n), y] + 1e-12).mean())
    conf, pred = p.max(1), p.argmax(1)
    bins = np.linspace(0, 1, 11); ece = 0.0; rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc = float((pred[m] == y[m]).mean()); cf = float(conf[m].mean())
            ece += m.mean() * abs(acc - cf); rel.append((lo, hi, cf, acc, int(m.sum())))
    return dict(brier=brier, nll=nll, ece10=float(ece)), rel


def untrained_vs_trained(cfg, rt, out_dir, run_adapters: dict, log=print):
    """run_adapters: name -> adapter directory of trained students to include (e.g. P_v2, P_v3, B4)."""
    import torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .runtime import adapter_state_from_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = list(cfg.classes); nt = classes.index("no_tumour"); load = rt["load"]
    vp, vy = [p for p, _ in rt["val"]], np.array([y for _, y in rt["val"]])
    tp, ty = [p for p, _ in rt["test"]], np.array([y for _, y in rt["test"]])
    st, te = rt["student"].s, rt["teacher"].t
    seed_state, teacher_state = st.adapter_state(), te.adapter_state()
    models = [("Qwen2-VL-2B, untrained", st, _zero_lora(seed_state)),
              ("2B seed student (412 images)", st, seed_state)]
    for name, ad in run_adapters.items():
        models.append((f"2B student after {name}", st, adapter_state_from_dir(ad)))
    models += [("Qwen2-VL-7B, untrained", te, _zero_lora(teacher_state)),
               ("7B cloud teacher (1,441 images)", te, teacher_state)]
    rows, rels, Z = [], {}, {}
    for name, m, state in models:
        m.load_adapter_state(state)
        bs = 8 if m is st else 4
        zv = m.class_logits([load(p) for p in vp], batch_size=bs)
        zt = m.class_logits([load(p) for p in tp], batch_size=bs)
        T = float(fit_temperature(zv, vy))
        Z[name] = (zt, T)
        for tlabel, Tu in (("raw (T = 1)", 1.0), (f"calibrated (T = {T:.2f})", T)):
            p = softmax(zt, Tu)
            mt = classification_metrics(ty, p.argmax(1), p, nt)
            cs, rel = calibration_scores(zt, ty, Tu)
            rows.append(dict(model=name, scaling=tlabel, accuracy=round(100 * mt["accuracy"], 1),
                             macro_f1=round(mt["macro_f1"], 3), sensitivity=round(mt["tumour_sensitivity"], 3),
                             specificity=round(mt["specificity"], 3), auc=round(mt["auroc"], 3),
                             ece=round(cs["ece10"], 3), brier=round(cs["brier"], 3), nll=round(cs["nll"], 3),
                             pred_no_tumour_share=round(float((p.argmax(1) == nt).mean()), 3)))
            if Tu != 1.0:
                rels[name] = rel
        log(f"  evaluated: {name}")
    df = pd.DataFrame(rows); df.to_csv(out_dir / "table_untrained_vs_trained.csv", index=False)
    log(df.to_string(index=False))
    # reliability diagrams (calibrated)
    n = len(rels); fig, axs = plt.subplots(1, n, figsize=(3.0 * n, 3.2), dpi=160, sharey=True)
    for ax, (name, rel) in zip(np.atleast_1d(axs), rels.items()):
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        if rel:
            cf = [r[2] for r in rel]; ac = [r[3] for r in rel]
            ax.bar([r[0] + 0.05 for r in rel], ac, width=0.09, color="#2F5597", alpha=.8, label="accuracy")
            ax.plot(cf, ac, "o", color="#C55A11", ms=3)
        ax.set_title(name.replace(", ", ",\n").replace(" student after ", " student\nafter "), fontsize=7.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("confidence", fontsize=7)
    np.atleast_1d(axs)[0].set_ylabel("accuracy", fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "figure_reliability.png", facecolor="white"); plt.close(fig)
    # restore deployed states
    st.load_adapter_state(seed_state); te.load_adapter_state(teacher_state)
    return df, Z


def attribution_pictures(cfg, rt, out_dir, final_adapter, final_name, log=print):
    """Grad-CAM + YOLO boxes for one test image per class, from untrained and trained models."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .runtime import adapter_state_from_dir
    classes = list(cfg.classes); load = rt["load"]
    st, te = rt["student"].s, rt["teacher"].t
    seed_state, teacher_state = st.adapter_state(), te.adapter_state()
    picks = []
    for k, c in enumerate(classes):
        cand = [p for p, y in rt["test"] if y == k and (c == "no_tumour" or rt["boxes"].get(p))]
        cand = cand or [p for p, y in rt["test"] if y == k]
        picks.append((cand[0], k))
    cols = [("2B untrained", st, _zero_lora(seed_state)), ("2B seed student", st, seed_state),
            (f"2B after {final_name}", st, adapter_state_from_dir(final_adapter)),
            ("7B untrained", te, _zero_lora(teacher_state)), ("7B teacher", te, teacher_state)]
    fig, axs = plt.subplots(len(picks), len(cols), figsize=(2.6 * len(cols), 2.75 * len(picks)), dpi=150)
    for j, (cname, m, state) in enumerate(cols):
        m.load_adapter_state(state)
        for i, (path, y) in enumerate(picks):
            im = load(path); z, cam = m.gradcam(im); p = softmax(z, 1.0); k = int(p.argmax())
            ax = axs[i][j]; ax.imshow(im); ax.imshow(cam, cmap="jet", alpha=.40)
            for b in rt["boxes"].get(path, []):
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color="lime", lw=1.4))
            ok = "✓" if k == y else "✗"
            ax.set_title(f"{classes[k]} {p[k]:.2f} {ok}", fontsize=8, color="black" if k == y else "#C00000")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.text(0.5, 1.25, cname, transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"true: {classes[y]}", fontsize=9)
        log(f"  pictures: {cname}")
    fig.tight_layout(); fig.savefig(out_dir / "figure_untrained_vs_trained_gradcam.png", facecolor="white"); plt.close(fig)
    st.load_adapter_state(seed_state); te.load_adapter_state(teacher_state)


def training_summary(cfg, rt, out_dir, log=print):
    """Parameters trained, data used, and accuracy on the training data vs validation vs test (generalisation gap)."""
    from .train import evaluate
    from .runtime import manifest, part
    classes = list(cfg.classes); nt = classes.index("no_tumour"); load = rt["load"]
    man = manifest(cfg); st, te = rt["student"].s, rt["teacher"].t
    rows = []
    for name, m, train_part, tcount in (("2B seed student", st, "seed", None), ("7B cloud teacher", te, "teacher_pool", None)):
        tr = part(man, train_part)
        res = {}
        for lab, data in (("train", tr), ("validation", rt["val"]), ("test", rt["test"])):
            paths, ys = [p for p, _ in data], np.array([y for _, y in data])
            mt, _ = evaluate(m, paths, ys, load, T=1.0, no_tumour_idx=nt, batch_size=8 if m is st else 4)
            res[lab] = round(100 * mt["accuracy"], 1)
        tot, lora = m.parameter_count(), m.lora_parameter_count()
        rows.append(dict(model=name, training_images=len(tr), total_parameters_M=round(tot / 1e6, 1),
                         trained_lora_parameters_M=round(lora / 1e6, 2), trained_share_pct=round(100 * lora / tot, 2),
                         train_acc=res["train"], val_acc=res["validation"], test_acc=res["test"]))
        log(f"  training summary: {name}")
    df = pd.DataFrame(rows); df.to_csv(out_dir / "table_training_summary.csv", index=False); log(df.to_string(index=False))
    return df
