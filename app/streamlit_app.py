"""Layer 6 application: regional diagnosis, specialist review and administrator views.

Run (on a machine with a GPU and the trained adapters):
    streamlit run app/streamlit_app.py -- --work /path/to/bcd --adapter /path/to/student_adapter
The specialist queue and administrator charts read the simulation outputs in <work>/runs.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bcd.config import load_config
from bcd.gate import GateConfig, composite_gate, softmax

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/kaggle/working/bcd")
ap.add_argument("--adapter", default=None)
ap.add_argument("--yolo", default=None)
args, _ = ap.parse_known_args()
cfg = load_config(); cfg.paths.work_dir = args.work
W = Path(args.work)

st.set_page_config(page_title="Brain MRI decision support (research prototype)", layout="wide")
st.warning("Research prototype — not for clinical use. All outputs are drafts for review by a qualified clinician.")
view = st.sidebar.radio("View", ["Regional diagnosis", "Specialist review", "Administrator"])


@st.cache_resource
def models():
    from bcd.runtime import load_student
    from bcd.detector import TumourDetector
    s = load_student(cfg, adapter=args.adapter, device=0)
    d = TumourDetector(args.yolo, cfg.yolo.conf_keep, cfg.yolo.imgsz, device=0) if args.yolo else None
    return s, d


if view == "Regional diagnosis":
    f = st.file_uploader("Upload a brain MRI slice", type=["png", "jpg", "jpeg"])
    if f and args.adapter:
        from PIL import Image
        im = Image.open(f).convert("L").resize((cfg.data.image_size,) * 2).convert("RGB")
        s, d = models()
        z, cam = s.gradcam(im)
        boxes = d(im) if d else []
        T = 1.0
        p = softmax(z, T)
        r = composite_gate(p, cam, boxes, None, cfg.classes.index("no_tumour"), GateConfig.from_cfg(cfg.gate))
        c1, c2 = st.columns(2)
        c1.image(im, caption="Input")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(); ax.imshow(im); ax.imshow(cam, cmap="jet", alpha=.45)
        for b in boxes:
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color="lime", lw=2))
        ax.axis("off"); c2.pyplot(fig)
        st.subheader(f"Prediction: {cfg.classes[r.y_hat]}  (confidence {r.conf:.2f})")
        st.table(pd.DataFrame([dict(G_base=r.g_base, A_YOLO=r.a_yolo, A_XAI=r.a_xai, G=r.g, override=r.override)]))
        if r.decision == "accept":
            st.success("Accepted locally")
            if r.y_hat != cfg.classes.index("no_tumour"):
                from bcd.runtime import guidance
                st.markdown("**Draft treatment plan (for clinician review)**")
                st.write(s.generate_plan(im, cfg.classes[r.y_hat], guidance(cfg.classes[r.y_hat])))
        else:
            st.error("Escalated — awaiting cloud model / specialist review")
    elif f:
        st.info("Start the app with --adapter to enable inference.")

elif view == "Specialist review":
    runs = sorted((W / "runs").glob("*/cases.csv"))
    if not runs:
        st.info("No simulation runs found.")
    else:
        run = st.selectbox("Run", runs, format_func=lambda p: p.parent.name)
        df = pd.read_csv(run)
        q = df[df.route == "clinician"].copy()
        q["priority"] = np.where(q.reasons.astype(str).str.contains("disagreement"), 0, 1)
        st.dataframe(q.sort_values("priority")[["idx", "path", "y_local", "y_cloud", "g", "g_cloud", "reasons", "y_final"]])

else:
    runs = sorted((W / "runs").glob("*/rounds.csv"))
    for r in runs:
        d = pd.read_csv(r)
        st.markdown(f"**{r.parent.name}**")
        st.line_chart(d.set_index("round")[["student_f1", "tau_gate"]])
        s = json.loads((r.parent / "summary.json").read_text())
        st.json({k: s[k] for k in ("escalation_rate", "reviews_per_100", "system_accuracy") if k in s})
