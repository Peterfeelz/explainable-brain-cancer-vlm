"""Layer 6 demo application (Gradio): regional diagnosis, specialist review queue and administrator view.

Runs inside a Kaggle notebook on the trained models from the experiments and opens a public link
(share=True) for screenshots or a live demonstration. Research prototype - not for clinical use.
"""
from __future__ import annotations
import json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from .data import load_image
from .gate import GateConfig, composite_gate, softmax, cloud_route
from .runtime import work, manifest, make_loader, part, load_student, load_teacher, guidance, TeacherCache

NICE = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary tumour", "no_tumour": "No tumour"}


def _system_state(cfg, run="P_seed0"):
    """Gate settings the proposed system ended its deployment run with (weights, thresholds, T, grounding)."""
    import pickle
    from .gate import GROUNDINGS
    W = work(cfg)
    rounds = pd.read_csv(W / "runs" / run / "rounds.csv")
    last = rounds.iloc[-1]
    gcfg = GateConfig.from_cfg(cfg.gate)
    s = json.loads((W / "runs" / run / "summary.json").read_text()).get("gate")
    if s:                                               # v2.0 runs store their validation-selected gate
        gcfg.w1, gcfg.w2, gcfg.w3 = s["w1"], s["w2"], s["w3"]
        gcfg.tau_xai_min = s["tau_xai_min"]
        gp = W / "runs" / run / "grounding.pkl"
        if s.get("grounding") and gp.exists():
            GROUNDINGS[s["grounding"]] = pickle.loads(gp.read_bytes()); gcfg.grounding = s["grounding"]
    else:                                               # v1 run: weights selected on validation were (1, 0, 0)
        gcfg.w1, gcfg.w2, gcfg.w3 = 1.0, 0.0, 0.0
        gcfg.tau_xai_min = 0.0
    gcfg.tau_gate = float(last.tau_gate)
    return gcfg, float(last["T"])


def build_demo(cfg, run=None, with_teacher=True, log=print):
    import gradio as gr
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from .detector import TumourDetector

    import gc, shutil, traceback
    gc.collect(); torch.cuda.empty_cache()
    for d in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(d)
        log(f"GPU {d}: {free / 2**30:.1f} GB free of {total / 2**30:.1f} GB")
        if free / 2**30 < 9:
            log(f"WARNING: GPU {d} is already partly full (models from an earlier run?). "
                "If the app runs out of memory: Run -> Restart, then run the cell again.")
    W = work(cfg)
    if run is None:
        from .runtime import best_run
        run = best_run(cfg)
    log(f"deployment run shown in the app: {run}")
    classes = list(cfg.classes); nt = classes.index("no_tumour")
    gcfg, T = _system_state(cfg, run)
    log(f"gate: tau_gate={gcfg.tau_gate:.3f}, T={T:.3f}, weights=({gcfg.w1},{gcfg.w2},{gcfg.w3})")
    student = load_student(cfg, adapter=str(W / "runs" / run / "adapter"), device=0)
    det = TumourDetector(str(W / "yolo_runs" / "tumour" / "weights" / "best.pt"), cfg.yolo.conf_keep,
                         cfg.yolo.imgsz, device=0)
    teacher, T_t = None, 1.0
    if with_teacher:
        teacher = load_teacher(cfg, adapter=str(W / "adapters" / "teacher"),
                               device=1 if torch.cuda.device_count() > 1 else 0)
        tc = TeacherCache(cfg, teacher, make_loader(cfg))
        T_t = tc.fit_T(part(manifest(cfg), "validation"))
    log(f"models loaded (teacher T={T_t:.3f})")

    def to_rgb(fig):
        """Render a matplotlib figure to an RGB array (avoids gr.Plot, which re-encodes figures as WebP)."""
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return arr

    def overlay(im, cam, boxes, title):
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=110)
        ax.imshow(im); ax.imshow(cam, cmap="jet", alpha=.40)
        for b in boxes:
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color="lime", lw=2))
            ax.text(b[0], b[1] - 4, f"YOLO {b[4]:.2f}", color="lime", fontsize=8)
        ax.axis("off"); fig.tight_layout()   # prediction is shown in the text, not over the panel label
        return to_rgb(fig)

    from .explain import explain as _explain, mismatch as _mismatch, mismatch_prompt as _mprompt, SELF_EXPLAIN
    from .gate import GROUNDINGS as _GR

    def analyse(img, draft_plan, model_text=False):
        try:
            return _analyse(img, draft_plan, model_text)
        except Exception as e:
            traceback.print_exc()
            return None, None, f"**Error:** {type(e).__name__}: {e}", {}, {}
        finally:
            gc.collect(); torch.cuda.empty_cache()

    def _analyse(img, draft_plan, model_text=False):
        if img is None:
            return None, None, "Upload an MRI slice first.", {}, {}
        # input validation (S11): size and colour checks before any model is called
        w_, h_ = img.size
        if min(w_, h_) < 64:
            return None, None, "**Rejected:** image is too small (minimum 64 × 64 pixels).", {}, {}
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        colourfulness = float(np.mean(np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])))
        if colourfulness > 25:
            return None, None, ("**Rejected:** this looks like a colour photograph, not a greyscale MRI slice. "
                                "Upload a brain MRI image."), {}, {}
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            img.save(f.name); im = load_image(f.name, cfg.data.image_size)
        boxes = det(im)
        z, cam = student.gradcam(im, T)
        p = softmax(z, T)
        r = composite_gate(p, cam, boxes, None, nt, gcfg)
        local = {NICE[c]: float(p[i]) for i, c in enumerate(classes)}
        fig_local = overlay(im, cam, boxes, f"Local student: {NICE[classes[r.y_hat]]}")
        lines = [f"### Local result: **{NICE[classes[r.y_hat]]}** (confidence {r.conf:.2f})",
                 f"Gate score G = {r.g:.3f} (threshold τ_gate = {gcfg.tau_gate:.3f}); "
                 f"normalised entropy {r.h_norm:.2f}; YOLO boxes: {len(boxes)}",
                 "Gate terms: G_base {:.3f} · A_YOLO {} · A_XAI {} · weights ({:.2f}, {:.2f}, {:.2f})".format(
                     r.g_base, "—" if r.a_yolo is None else f"{r.a_yolo:.3f}",
                     "—" if r.a_xai is None else f"{r.a_xai:.3f}", gcfg.w1, gcfg.w2, gcfg.w3)]
        fig_cloud, cloud = None, {}
        pc_used, route_used, reasons_used, extra = None, None, None, []
        grd = _GR.get(gcfg.grounding) if gcfg.grounding else None
        if r.decision == "accept":
            lines.append("#### ✅ Accepted locally. No cloud call needed.")
            plan_model, y_plan = student, r.y_hat
        else:
            why = "possible missed tumour (detector sees a lesion)" if r.override == "possible_missed_tumour" else \
                  "confidence below the gate threshold"
            lines.append(f"#### ⬆️ Escalated to the cloud model: {why}.")
            plan_model, y_plan = student, r.y_hat
            if teacher is not None:
                try:
                    zt, camt = teacher.gradcam(im)
                except torch.OutOfMemoryError:            # fall back to logits without a heat map
                    gc.collect(); torch.cuda.empty_cache()
                    zt, camt = teacher.class_logits([im])[0], np.zeros((cfg.data.image_size,) * 2)
                    lines.append("*(Cloud heat map skipped: GPU memory full.)*")
                pc = softmax(zt, T_t)
                rc = composite_gate(pc, camt, boxes, None, nt, GateConfig(**{**gcfg.__dict__, "tau_gate": 0.0}))
                cloud = {NICE[c]: float(pc[i]) for i, c in enumerate(classes)}
                fig_cloud = overlay(im, camt, boxes, f"Cloud teacher: {NICE[classes[rc.y_hat]]}")
                route, reasons = cloud_route(r, rc, p, pc, rc.y_hat != nt and cfg.clinician.review_plans,
                                             0.0, cfg.gate.kl_delta)
                pc_used, route_used, reasons_used = pc, route, reasons
                if _mismatch(classes, p, r, boxes, grd):
                    q = _mprompt(classes, p, boxes, grd, float(cfg.data.image_size))
                    try:
                        ans = teacher.generate_with_image(im, q, 90)
                    except torch.OutOfMemoryError:
                        gc.collect(); torch.cuda.empty_cache(); ans = "(not generated: GPU memory full)"
                    extra.append("#### Bounding-box mismatch: prompt sent to the cloud model\n\n> " + q +
                                 "\n\n**Cloud model's answer:** " + ans)
                lines.append(f"Cloud result: **{NICE[classes[rc.y_hat]]}** (confidence {pc.max():.2f}).")
                if route == "clinician":
                    lines.append("#### 👩‍⚕️ Routed to specialist review: " + ", ".join(reasons).replace("_", " ") + ".")
                else:
                    lines.append("Cloud answer accepted and added to the distillation buffer.")
                plan_model, y_plan = teacher, rc.y_hat
        lines.append("#### Explanation\n\n" + _explain(classes, p, r, boxes, grd, pc_used, route_used, reasons_used,
                                                       (gcfg.w1, gcfg.w2, gcfg.w3), gcfg.tau_gate,
                                                       float(cfg.data.image_size)))
        lines.extend(extra)
        if model_text:
            lines.append("#### The models' own descriptions (generated text; may not reflect how they decided)")
            lines.append("**Local model:** " + student.generate_with_image(im, SELF_EXPLAIN.format(dx=NICE[classes[r.y_hat]]), 80))
            if pc_used is not None:
                yc = int(np.argmax(pc_used))
                lines.append("**Cloud model:** " + teacher.generate_with_image(im, SELF_EXPLAIN.format(dx=NICE[classes[yc]]), 80))
        if draft_plan and y_plan != nt:
            name = classes[y_plan]
            plan = plan_model.generate_plan(im, name, guidance(name), 160)
            lines.append("**Draft treatment plan (for clinician review only):**\n\n" + plan)
        lines.append("\n*Research prototype. Not for clinical use. A qualified clinician must review every result.*")
        return fig_local, fig_cloud, "\n\n".join(lines), local, cloud

    # specialist queue and administrator data from the deployment run
    cases = pd.read_csv(W / "runs" / run / "cases.csv")
    q = cases[cases.route == "clinician"].copy()
    q["case"] = q.path.map(lambda s: Path(s).name)
    for col in ("y_local", "y_cloud", "y_final", "y_true"):
        if col in q:
            q[col] = q[col].map(lambda v: NICE[classes[int(v)]] if pd.notna(v) else "")
    queue = q[["idx", "case", "y_local", "y_cloud", "g", "g_cloud", "reasons", "y_final"]].round(3).head(200)
    summ = []
    for f in sorted((W / "runs").glob("*/summary.json")):
        s = json.loads(f.read_text())
        summ.append(dict(condition=f.parent.name.replace("_seed0", ""),
                         escalation_pct=round(100 * s["escalation_rate"], 1),
                         reviews_per_100=round(s["reviews_per_100"], 1),
                         system_accuracy_pct=None if s.get("system_accuracy") is None else round(100 * s["system_accuracy"], 1),
                         student_test_accuracy_pct=round(100 * s["final_student_test"]["accuracy"], 1)))
    summ = pd.DataFrame(summ)
    rounds = pd.read_csv(W / "runs" / run / "rounds.csv")
    fig_r, ax = plt.subplots(figsize=(6, 3), dpi=110)
    ax.step(rounds.cases_seen, rounds.student_f1, where="post", color="#2a78d6", lw=2, label="student macro-F1")
    ax.step(rounds.cases_seen, rounds.tau_gate, where="post", color="#eb6834", lw=2, label="τ_gate")
    ax.set_xlabel("stream cases processed"); ax.legend(frameon=False); ax.grid(alpha=.3); fig_r.tight_layout()
    chart = to_rgb(fig_r)

    # example images are copied next to the app: Gradio refuses files from /kaggle/input
    test = part(manifest(cfg), "test")
    ex_dir = W / "demo_examples"; ex_dir.mkdir(exist_ok=True)
    examples = []
    for k, c in enumerate(classes):
        hit = [p for p, y in test if y == k]
        if hit:
            dst = ex_dir / f"{k + 1}_{c}{Path(hit[0]).suffix}"
            shutil.copy(hit[0], dst)
            examples.append([str(dst)])

    with gr.Blocks(title="Brain MRI decision support (research prototype)") as demo:
        gr.Markdown("# Brain MRI decision support: confidence-gated local–cloud VLM\n"
                    "Research prototype. Not for clinical use.")
        with gr.Tab("Regional diagnosis"):
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(type="pil", label="Upload a brain MRI slice", height=320, format="png")
                    plan = gr.Checkbox(label="Draft a treatment plan (slower)", value=False)
                    mtext = gr.Checkbox(label="Ask each model to describe its evidence (slower)", value=False)
                    btn = gr.Button("Analyse", variant="primary")
                    if examples:
                        gr.Examples(examples, inputs=[img], label="Test-set examples")
                with gr.Column(scale=2):
                    out_md = gr.Markdown()
                    with gr.Row():
                        out_local = gr.Image(label="Local student: Grad-CAM + YOLO", type="numpy", format="png",
                                             interactive=False)
                        out_cloud = gr.Image(label="Cloud teacher (if escalated)", type="numpy", format="png",
                                             interactive=False)
                    with gr.Row():
                        lab_local = gr.Label(label="Local class probabilities")
                        lab_cloud = gr.Label(label="Cloud class probabilities")
            btn.click(analyse, [img, plan, mtext], [out_local, out_cloud, out_md, lab_local, lab_cloud])
        with gr.Tab("Specialist review"):
            gr.Markdown(f"Cases routed to the specialist during the deployment run ({len(q)} reviews). "
                        "Select a case, enter the confirmed diagnosis and, for a tumour, the corrected plan. "
                        "Saved reviews become verified labels for the next distillation round (FR7).")
            gr.Dataframe(queue, wrap=True)
            rev_file = W / "specialist_reviews.csv"
            with gr.Row():
                case_in = gr.Dropdown(choices=list(queue["case"]), label="Case", allow_custom_value=False)
                label_in = gr.Radio(choices=[NICE[c] for c in classes], label="Confirmed diagnosis")
            plan_in = gr.Textbox(label="Corrected / approved treatment plan (tumour cases)", lines=4)
            save_btn = gr.Button("Save review", variant="primary")
            saved = gr.Dataframe(label="Saved specialist reviews")

            def save_review(case, label, plan):
                if not case or not label:
                    return gr.update(), "Select a case and a diagnosis first."
                row = pd.DataFrame([dict(case=case, label=label, plan=(plan or "").strip(),
                                         time=pd.Timestamp.now().isoformat(timespec="seconds"))])
                old = pd.read_csv(rev_file) if rev_file.exists() else pd.DataFrame()
                allr = pd.concat([old[old.case != case] if len(old) else old, row], ignore_index=True)
                allr.to_csv(rev_file, index=False)
                return allr, f"Saved review for {case}: {label}."
            msg = gr.Markdown()
            save_btn.click(save_review, [case_in, label_in, plan_in], [saved, msg])
        with gr.Tab("Administrator"):
            gr.Markdown("Results of every condition on the 1,853-case stream, and the proposed system's "
                        "student F1 and gate threshold over time.")
            gr.Dataframe(summ)
            gr.Image(chart, label="Proposed system over the stream", type="numpy", format="png", interactive=False)
    return demo
