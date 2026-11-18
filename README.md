# Explainable Agentic Brain Cancer Diagnosis with Online VLM Distillation

Source code for the Master of Information Technology (MIT899) project:

**"Design and Implementation of an Explainable Agentic Brain Cancer Diagnosis Application in Low-Resource African Settings Using Online VLM Distillation"**
Peter Ademola Arowoogun (249074233), Department of Computer Science, University of Lagos.
Supervisor: Dr E. P. Fasina. November 2026.

> **Research prototype. Not for clinical use.** Treatment-plan text is a draft for clinician review only.

## What the system does

A small vision-language model (Qwen2-VL-2B) runs on the facility's own computer and sees every brain MRI slice. A **confidence gate** decides whether its answer can be trusted:

```
G = 0.60 · confidence + 0.35 · lesion-location fit (YOLOv8) + 0.05 · attention (Grad-CAM)
```

- If G is at or above the threshold, the local answer is given, with a plain-language explanation.
- If G is below the threshold, the case is escalated to a larger cloud model (Qwen2-VL-7B).
- If the two models disagree, the case is routed to a clinician.

Verified answers become training labels. The local model is retrained in batches while it is in use (**online distillation**). An update is promoted only after a McNemar test and a tumour-sensitivity guard. The gate threshold is recalibrated from audits of accepted cases.

## Main results

| Measured on the 909 held-out test images | Accuracy | Tumour sensitivity | Sent to the cloud |
|---|---|---|---|
| Local model alone | 76.8% | 98.8% | 0% |
| Cloud model alone | 88.7% | 97.1% | 100% |
| **Proposed system** (local + cloud + clinician) | **88.6%** | **99.2%** | **25%** |

- On the 1,853-case deployment stream, the system reached 93.3% accuracy with 22% escalation, 78% fewer cloud calls than cloud-only operation.
- Online distillation raised the local model from 69.5% to 84.0% on the test set.

## Repository layout

| Path | Contents |
|---|---|
| `bcd/data.py` | Dataset scan, perceptual-hash duplicate removal, figshare mask matching, partitions |
| `bcd/detector.py` | YOLOv8n tumour detector (train, validate, predict) |
| `bcd/vlm.py` | Qwen2-VL student and teacher: class probabilities, Grad-CAM, plan generation, training |
| `bcd/gate.py` | Composite confidence gate, lesion-location term, overrides, cloud routing |
| `bcd/losses.py` | Reliability-weighted distillation loss |
| `bcd/calibration.py` | Temperature scaling, false-accept-rate controller, threshold fitting |
| `bcd/train.py` | LoRA training; cumulative distillation rounds with McNemar promotion and rollback |
| `bcd/simulate.py` | Deployment-stream simulation for all conditions (B0–B5, P, P_v2, P_v3, ablations) |
| `bcd/runtime.py` | Connects the real models to the simulator (teacher cache, student handle) |
| `bcd/explain.py` | Plain-language explanation of each gate decision and the bounding-box mismatch prompt |
| `bcd/demo.py` | Gradio application: regional diagnosis, specialist review and administrator views |
| `bcd/compare.py` | Untrained vs trained models: accuracy, calibration and attribution maps |
| `bcd/metrics.py` | Accuracy, macro-F1, sensitivity, calibration error, risk–coverage metrics |
| `bcd/report.py` | Builds the results tables and figures from run outputs |
| `bcd/config.py` | Loads the settings in `configs/default.yaml` |
| `bcd/mock.py` | CPU mocks used only by the tests |
| `app/streamlit_app.py` | Earlier Streamlit version of the user interface |
| `configs/default.yaml` | All settings (models, LoRA, gate, distillation, clinician budget) |
| `guidelines/` | Short guideline notes used as context for draft treatment plans |
| `notebooks/01–08` | Kaggle notebooks for each stage: data and YOLO, seed student, teacher, simulation, results, demo, final iterations, untrained vs trained |
| `scripts/` | Kaggle launcher, notebook builder and a CPU dry run with mocks |
| `RUN_ON_KAGGLE.md` | Step-by-step guide to running the experiments on Kaggle |
| `tests/` | Unit tests (CPU) |

## Data

The images are not included in this repository. Both datasets are public and de-identified:

- Brain Tumor MRI Dataset (Nickparvar, 2021): https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Figshare brain tumour dataset (Cheng et al., 2015), 3,064 T1-weighted slices with expert tumour masks.

After duplicate removal, the 5,027 images were split as follows:

| Partition | Images |
|---|---|
| Teacher pool | 1,441 |
| Seed | 412 |
| Validation | 412 |
| Deployment stream | 1,853 |
| Test | 909 |

## How to run (Kaggle, 2 × T4 GPUs)

1. Upload this folder to Kaggle as a dataset.
2. Create a notebook. Add as inputs this code, the MRI dataset and the figshare dataset.
3. Turn on **GPU T4 x2** and **Internet**.
4. Run the notebooks in `notebooks/` in order (01 to 08), attaching each notebook's output as an input to the next one. `RUN_ON_KAGGLE.md` gives the details.

The free Kaggle quota (about 30 GPU-hours per week) allowed one run per condition.

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

## Licence and citation

For academic use. If you use this code, please cite the dissertation above.
