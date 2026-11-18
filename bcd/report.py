"""Aggregate simulation outputs into the Chapter 4 tables and Figure 4.2."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "P", "A1", "A2", "A3", "A4"]
NAMES = {"B0": "B0 Local only", "B1": "B1 Cloud only", "B2": "B2 Softmax routing",
         "B3": "B3 Softmax + distillation", "B4": "B4 Offline full distillation",
         "B5": "B5 Random-selection distillation", "P": "P Proposed", "A1": "A1 without A_YOLO",
         "A2": "A2 without A_XAI", "A3": "A3 without clinician", "A4": "A4 without recalibration"}


def load_runs(run_dir) -> pd.DataFrame:
    rows = []
    for f in Path(run_dir).glob("*_seed*/summary.json"):
        s = json.loads(f.read_text())
        t = s.get("final_student_test", {})
        rows.append(dict(condition=s["condition"], seed=s["seed"],
                         system_accuracy=s.get("system_accuracy"), student_accuracy=t.get("accuracy"),
                         student_macro_f1=t.get("macro_f1"), student_sensitivity=t.get("tumour_sensitivity"),
                         student_ece=t.get("ece"), escalation_rate=s.get("escalation_rate"),
                         cloud_cost_ratio=s.get("escalation_rate"), reviews_per_100=s.get("reviews_per_100"),
                         local_accept_far=s.get("local_accept_far"), minutes=s.get("minutes")))
    return pd.DataFrame(rows)


def mean_sd(df, cols):
    g = df.groupby("condition")[cols]
    m, s, n = g.mean(), g.std(), g.size()
    out = pd.DataFrame(index=m.index)
    for c in cols:
        out[c] = [f"{m.loc[i, c]:.3f}" + (f" ± {s.loc[i, c]:.3f}" if n[i] > 1 and not np.isnan(s.loc[i, c]) else "")
                  if pd.notna(m.loc[i, c]) else "—" for i in m.index]
    out["runs"] = n
    out = out.reindex([c for c in ORDER if c in out.index]); out.index = [NAMES[i] for i in out.index]
    return out


def tables(run_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_runs(run_dir)
    if df.empty:
        raise FileNotFoundError(f"no runs in {run_dir}")
    t46 = mean_sd(df[df.condition.isin(ORDER[:7])], ["student_accuracy", "student_macro_f1", "student_sensitivity",
                                                     "student_ece", "system_accuracy", "escalation_rate",
                                                     "cloud_cost_ratio", "reviews_per_100"])
    t48 = mean_sd(df[df.condition.isin(["P", "A1", "A2", "A3", "A4"])],
                  ["student_macro_f1", "student_sensitivity", "local_accept_far", "escalation_rate"])
    t46.to_csv(out / "table_4_6_overall.csv"); t48.to_csv(out / "table_4_8_ablation.csv")
    df.to_csv(out / "all_runs.csv", index=False)
    return t46, t48


def figure_rounds(run_dir, out_png, conds=("P", "B3", "B5")):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for c in conds:
        fs = sorted(Path(run_dir).glob(f"{c}_seed*/rounds.csv"))
        if not fs:
            continue
        r = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(fs)])
        r = r[~r["final"].astype(bool)] if "final" in r else r
        g = r.groupby("round")
        ax[0].plot(g.student_f1.mean().index, g.student_f1.mean().values, marker="o", label=NAMES[c])
        esc = (g.cloud_calls.mean() / g.cases_seen.mean().clip(lower=1))
        ax[1].plot(esc.index, esc.values, marker="o", label=NAMES[c])
    ax[0].set(xlabel="Distillation round", ylabel="Student macro-F1 (test subset)")
    ax[1].set(xlabel="Distillation round", ylabel="Cumulative escalation rate")
    for a in ax:
        a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    return out_png
