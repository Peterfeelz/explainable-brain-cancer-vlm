"""Generate the Kaggle notebooks in notebooks/."""
import nbformat as nbf
from pathlib import Path

SETUP = r'''# === STEP 0: SETUP — run this cell first, every time =========================
# Before running: in the right-hand panel
#   * "Session options" -> Accelerator = GPU T4 x2, Internet = On
#   * "Add Input" -> your code dataset (the uploaded bcd-project.zip)
#   * "Add Input" -> search "brain tumor mri dataset" by masoudnickparvar
#   * (notebooks 02-05) "Add Input" -> "Your Work" -> the previous notebook(s)
import os, sys, glob, shutil, subprocess

def find_dir(test, root='/kaggle/input', max_depth=6):
    # walk the attached inputs and return the first folder that passes test(folder)
    root_depth = root.rstrip('/').count('/')
    for d, subdirs, files in os.walk(root):
        if d.count('/') - root_depth > max_depth:
            subdirs[:] = []
            continue
        if test(d, subdirs, files):
            return d
    return None

# 1) the project code: the folder that contains bcd/__init__.py
code_dir = find_dir(lambda d, s, f: os.path.isfile(os.path.join(d, 'bcd', '__init__.py')))
if code_dir is None:
    raise SystemExit('Code not found. Click "Add Input" and attach the dataset you created from bcd-project.zip.')
shutil.copytree(code_dir, '/kaggle/working/bcd-project', dirs_exist_ok=True)
sys.path.insert(0, '/kaggle/working/bcd-project')
print('code found at:', code_dir)

# 2) install the extra libraries (takes ~1-2 minutes)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.45,<4.52', 'peft>=0.12',
                'bitsandbytes>=0.43', 'accelerate>=0.33', 'ultralytics', 'imagehash', 'h5py'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'])  # old torchao on Kaggle breaks peft

# 3) the brain MRI dataset: the folder that has Training/ and Testing/ inside
from bcd.config import load_config
cfg = load_config('/kaggle/working/bcd-project/configs/default.yaml')
mri = find_dir(lambda d, s, f: 'Training' in s and 'Testing' in s)
if mri is None:
    raise SystemExit('MRI dataset not found. Click "Add Input" and attach masoudnickparvar/brain-tumor-mri-dataset.')
cfg.paths.kaggle_root = mri
print('MRI dataset found at:', mri)

# 4) results of earlier notebooks (a folder named bcd that contains manifest.csv)
os.makedirs(cfg.paths.work_dir, exist_ok=True)
for prev in glob.glob('/kaggle/input/**/bcd/manifest.csv', recursive=True):
    src = os.path.dirname(prev)
    if not src.startswith('/kaggle/working'):
        shutil.copytree(src, cfg.paths.work_dir, dirs_exist_ok=True)
        print('restored earlier results from:', src)
print('GPUs available:', __import__('torch').cuda.device_count())
print('Setup complete. Work folder:', cfg.paths.work_dir)
'''

NB = {
"01_data_and_yolo.ipynb": [
("md", "# Increment 1 — Data pipeline and YOLO tumour detector\nCleans and partitions the data (Section 3.5) and trains the YOLO detector (Step 1). Produces Table 4.5. Then **Save Version** so the output can be attached to notebook 02."),
("code", SETUP),
("code", "# Figshare .mat files (Cheng et al., 2015) are needed for YOLO and the A_XAI reference masks\nfrom bcd.data import find_figshare_dir\ncfg.paths.figshare_root = find_figshare_dir('/kaggle/input')\nn_mat = len(glob.glob(f'{cfg.paths.figshare_root}/**/*.mat', recursive=True)) if cfg.paths.figshare_root else 0\nprint('figshare folder:', cfg.paths.figshare_root, '| .mat files:', n_mat)\nif n_mat < 3000:\n    raise SystemExit('Figshare DATASET not attached. Add Input -> Datasets tab -> search \"figshare brain tumor dataset\" -> add the dataset with 3064 .mat files (a dataset, not a notebook).')\n# To save GPU time you can cap the stream, e.g. cfg.data.stream_limit = 1500"),

("code", "from bcd.data import prepare_all\nman = prepare_all(cfg)\nimport pandas as pd\nprint(pd.read_csv(f'{cfg.paths.work_dir}/partition_summary.csv', index_col=0))\nprint('duplicates removed:', len(pd.read_csv(f'{cfg.paths.work_dir}/duplicates_removed.csv')))\nprint('images with reference masks:', man.mask_path.notna().sum())"),
("code", "from bcd.detector import TumourDetector\nweights = TumourDetector.train(f'{cfg.paths.work_dir}/yolo/data.yaml', cfg.yolo.weights, cfg.yolo.epochs,\n                               cfg.yolo.imgsz, cfg.yolo.batch, project=f'{cfg.paths.work_dir}/yolo_runs', device=0)\ndet = TumourDetector(weights, cfg.yolo.conf_keep, cfg.yolo.imgsz, device=0)\nt45 = det.validate(f'{cfg.paths.work_dir}/yolo/data.yaml')\nimport json; json.dump(t45, open(f'{cfg.paths.work_dir}/table_4_5_yolo.json','w'), indent=2)\nt45  # -> Table 4.5 ('All' row; single tumour class)"),
],
"02_student_seed.ipynb": [
("md", "# Increment 2 — Seed fine-tuning of the local student (Qwen2-VL-2B, LoRA)\nAttach the output of notebook 01. Produces the seed student adapter and its validation metrics (B0 starting point)."),
("code", SETUP),
("code", "from bcd.runtime import seed_finetune\nadapter = seed_finetune(cfg)\nprint(open(f'{cfg.paths.work_dir}/student_seed_val.json').read())"),
("code", "# quick check: Grad-CAM on one validation image\nimport matplotlib.pyplot as plt, numpy as np\nfrom bcd.runtime import load_student, manifest, make_loader, part\nst = load_student(cfg, adapter=adapter); load = make_loader(cfg)\np, y = part(manifest(cfg), 'validation')[0]\nz, cam = st.gradcam(load(p))\nplt.imshow(load(p)); plt.imshow(cam, alpha=.45, cmap='jet'); plt.title(f'true={cfg.classes[y]} pred={cfg.classes[int(np.argmax(z))]}'); plt.axis('off');"),
],
"03_teacher_adapt.ipynb": [
("md", "# Increment 3 — Cloud teacher (Qwen2-VL-7B, 4-bit QLoRA) adapted on the teacher pool\nAttach outputs of 01 and 02. Produces the teacher adapter and its validation metrics."),
("code", SETUP),
("code", "# Choose the attribution (XAI) variant on validation images with expert masks (student, GPU 1)\nfrom bcd.runtime import choose_cam_variant\nbest, t_xai = choose_cam_variant(cfg, device=1)\nfrom IPython.display import Image as _Im, display; display(_Im(f'{cfg.paths.work_dir}/figure_xai_variants.png'))"),
("code", "from bcd.runtime import teacher_adapt\nt_adapter = teacher_adapt(cfg)\nprint(open(f'{cfg.paths.work_dir}/teacher_val.json').read())"),
],
"04_simulation.ipynb": [
("md", "# Increments 4–5 — Gate quality and deployment-stream simulation\nAttach outputs of 01–03. Runs Table 4.7 and the conditions of Table 3.6. Each condition takes roughly 1–2 h on T4 for ~1,500 stream cases: run a few conditions per session and Save Version; later sessions attach this notebook's output and continue."),
("code", SETUP),
("code", "import glob, json\nfrom bcd.runtime import build_runtime, run_condition, gate_quality\nW = cfg.paths.work_dir\nyolo = glob.glob(f'{W}/yolo_runs/tumour/weights/best.pt')[0]\nrt = build_runtime(cfg, f'{W}/adapters/student_seed', f'{W}/adapters/teacher', yolo)\nprint(len(rt['stream']), 'stream cases;', 'teacher T =', rt['teacher'].T)\nfrom bcd.runtime import teacher_test_eval\nprint('teacher on test:', teacher_test_eval(cfg, rt))"),
("code", "import pandas as pd\nif os.path.exists(f'{W}/table_gate_quality.csv') and 'gate_quality' not in globals().get('FORCE', []):\n    gq = pd.read_csv(f'{W}/table_gate_quality.csv')\nelse:\n    gq = gate_quality(cfg, rt); gq.to_csv(f'{W}/table_gate_quality.csv', index=False)\nprint(gq.to_string())   # Table 4.10"),
("code", "# Choose what to run in this session. P must run before B2, B3 and B5 (they use P's escalation rate).\nimport shutil\nfor cond in globals().get('FORCE', []):   # re-run these: the old run is moved to runs_archive (kept for the report)\n    if os.path.isdir(f'{W}/runs/{cond}'):\n        os.makedirs(f'{W}/runs_archive', exist_ok=True)\n        shutil.rmtree(f'{W}/runs_archive/{cond}', ignore_errors=True)\n        shutil.move(f'{W}/runs/{cond}', f'{W}/runs_archive/{cond}'); print('archived old run', cond)\nTODO = globals().get('RUN_LIST') or [('P', 0), ('B0', 0), ('B1', 0), ('B2', 0)]   # P first: B2/B3/B5 use its escalation rate\nimport traceback\nFAILED = []\nfor cond, seed in TODO:\n    if os.path.exists(f'{W}/runs/{cond}_seed{seed}/summary.json'):\n        print('skip', cond, seed); continue\n    try:\n        rr = None\n        if cond.split('_')[0] in ('B2', 'B3', 'B5'):   # matched escalation budget = the proposed system's rate (P, or P_v2 for _v2 runs)\n            ref = 'P_v2' if cond.endswith('_v2') else 'P'\n            rr = json.load(open(f'{W}/runs/{ref}_seed{seed}/summary.json'))['escalation_rate']\n        s = run_condition(cfg, rt, cond, seed, random_rate=rr)\n        print(cond, seed, {k: s[k] for k in ('escalation_rate', 'reviews_per_100')}, s['final_student_test'])\n    except Exception:\n        traceback.print_exc(); FAILED.append(cond)   # keep going: the other runs still finish and are saved\n        print('RUN FAILED:', cond, '- the other runs continue')\nprint('failed runs:', FAILED or 'none')\nfrom bcd.runtime import results_dump\nresults_dump(cfg)"),
("md", "Suggested sessions (seed 0): `P B0 B1 B2` → `B3 B4 B5` → `A1 A2` → `A3 A4`; extra seeds for P, B3, B5 only if GPU hours remain."),
],
"06_demo.ipynb": [
("md", "# Demo application (Layer 6)\nRun INTERACTIVELY (Shift+Enter), not with Save & Run All. Attach the output of the last step-04 notebook. Opens a public Gradio link for screenshots or a live demonstration."),
("code", SETUP),
("code", "import subprocess, sys\ntry:\n    import gradio  # Kaggle's preinstalled Gradio\nexcept Exception as e:\n    print('preinstalled gradio failed to import:', e, '- installing gradio 5.50 (keeps Pillow)')\n    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'gradio==5.50.0', 'gradio-client==1.14.0'], check=True)\n    for m in [m for m in sys.modules if m.startswith(('gradio', 'gradio_client'))]:\n        del sys.modules[m]\nimport gradio; print('gradio', gradio.__version__)\nimport gc, torch\nif 'demo' in globals():\n    try: demo.close()\n    except Exception: pass\n    del demo\nsys.last_type = sys.last_value = sys.last_traceback = None   # release models held by an earlier failed run\ngc.collect(); torch.cuda.empty_cache()\nfrom bcd.demo import build_demo\ndemo = build_demo(cfg)\ndemo.launch(share=True, prevent_thread_lock=True, show_error=True, allowed_paths=[cfg.paths.work_dir])\nprint('Open the public link above (it ends with .gradio.live). Keep this notebook running while you use it.')"),
],
"07_final_v2.ipynb": [
("md", "# Final runs (v2.0): class-aware grounding (RQ1), ablations, plan quality and statistics\nAttach the output of the last notebook that ran step 04 (it holds the models, teacher cache and earlier runs). Run with **Save Version -> Save & Run All** (about 6-9 h)."),
("code", SETUP),
("code", "import glob, json, os\nfrom bcd.runtime import build_runtime, run_condition, gate_study_v2, results_dump, plan_quality, adapter_state_from_dir, significance, pack_results, work\nW = cfg.paths.work_dir\nyolo = glob.glob(f'{W}/yolo_runs/tumour/weights/best.pt')[0]\nrt = build_runtime(cfg, f'{W}/adapters/student_seed', f'{W}/adapters/teacher', yolo)\nprint(len(rt['stream']), 'stream cases;', len(rt['train_boxes']), 'training images with YOLO boxes for the location model')"),
("code", "# 1) RQ1 on validation: original vs class-aware grounding terms, weights chosen by 5-fold cross-validation\ngs = gate_study_v2(cfg, rt)"),
("code", "# 2) proposed system with the new gate + cumulative online loop; then the ablations of each grounding term\nimport traceback\ndef run(cond, seed=0):\n    if os.path.exists(f'{W}/runs/{cond}_seed{seed}/summary.json'):\n        print('skip', cond, seed); return json.load(open(f'{W}/runs/{cond}_seed{seed}/summary.json'))\n    try:\n        s = run_condition(cfg, rt, cond, seed)\n        print(cond, seed, s['gate'], s['final_student_test']); return s\n    except Exception:\n        traceback.print_exc(); print('RUN FAILED:', cond, seed); return None\nsP = run('P_v3', 0)\nw = (sP or {}).get('gate', {})\nprint('P_v3 gate weights selected on validation:', w)\nif w.get('w2', 0) > 0: run('A1_v3', 0)      # without A_YOLO: shows what the YOLO term contributes\nif w.get('w3', 0) > 0: run('A2_v3', 0)      # without A_XAI: shows what the Grad-CAM term contributes\nif w.get('w2', 0) == 0 and w.get('w3', 0) == 0: run('P_v3', 1)   # no grounding weight: repeat run instead"),
("code", "# 3) treatment-plan quality (Appendix A.2 judge): seed student vs final student vs teacher, 30 test tumour images\nfrom bcd.runtime import best_run\nbr = best_run(cfg)\nprint('final student from run:', br)\ntry:\n    plan_quality(cfg, rt, {'seed student': rt['student'].seed_state, f'final student ({br})': adapter_state_from_dir(f'{W}/runs/{br}/adapter')}, n=30)\nexcept Exception:\n    traceback.print_exc(); print('PLAN QUALITY FAILED - the other results are kept')"),
("code", "# 4) statistics: CIs for every run and paired tests (same test images / same stream cases)\npairs = [('P_v3_seed0', 'P_v2_seed0'), ('P_v3_seed0', 'P_seed0'), ('P_v2_seed0', 'P_seed0'), ('P_v3_seed0', 'B2_seed0'),\n         ('P_v2_seed0', 'B5_v2_seed0'), ('P_v2_seed0', 'B3_v2_seed0'), ('P_v3_seed0', 'A1_v3_seed0'), ('P_v3_seed0', 'A2_v3_seed0'),\n         ('P_v3_seed0', 'P_v3_seed1'), ('P_v2_seed0', 'B4_seed0')]\nsignificance(cfg, rt, pairs)\nresults_dump(cfg)\nfor f in ('gate_study_v2.json', 'table_gate_study_v2.csv', 'table_plan_quality.csv'):\n    if os.path.exists(f'{W}/{f}'): print('==', f); print(open(f'{W}/{f}').read())\nprint('download:', pack_results(cfg))"),
],
"08_untrained_vs_trained.ipynb": [
("md", "# Untrained vs trained VLMs: task performance, calibration, pictures and training (v2.1)\nAttach the output of the notebook that ran step 07. Run with **Save Version -> Save & Run All** (about 1 h)."),
("code", SETUP),
("code", "import glob, os, traceback\nfrom pathlib import Path\nfrom bcd.runtime import build_runtime, best_run, pack_results, work\nfrom bcd.compare import untrained_vs_trained, attribution_pictures, training_summary\nW = cfg.paths.work_dir\nyolo = glob.glob(f'{W}/yolo_runs/tumour/weights/best.pt')[0]\nrt = build_runtime(cfg, f'{W}/adapters/student_seed', f'{W}/adapters/teacher', yolo)\nout = Path(W) / 'comparison'\nruns = {k: f'{W}/runs/{k}_seed0/adapter' for k in ('P_v2', 'P_v3', 'A2_v3', 'B4') if os.path.isdir(f'{W}/runs/{k}_seed0/adapter')}\nprint('trained students included:', list(runs))"),
("code", "# (i) task performance and calibration of untrained and trained models on the 909 test images\ntry:\n    df, Z = untrained_vs_trained(cfg, rt, out, runs)\nexcept Exception:\n    traceback.print_exc(); print('COMPARISON FAILED')\nfrom IPython.display import Image as _Im, display\nif (out / 'figure_reliability.png').exists(): display(_Im(str(out / 'figure_reliability.png')))"),
("code", "# (ii) pictures: Grad-CAM + YOLO for one test image per class, untrained vs trained\ntry:\n    br = best_run(cfg)\n    attribution_pictures(cfg, rt, out, f'{W}/runs/{br}/adapter', br.replace('_seed0', ''))\n    display(_Im(str(out / 'figure_untrained_vs_trained_gradcam.png')))\nexcept Exception:\n    traceback.print_exc(); print('PICTURES FAILED')"),
("code", "# (iii) training: parameters trained and accuracy on training data vs validation vs test\ntry:\n    training_summary(cfg, rt, out)\nexcept Exception:\n    traceback.print_exc(); print('TRAINING SUMMARY FAILED')\nfor f in sorted(out.glob('*.csv')):\n    print('==', f.name); print(open(f).read())\nprint('download:', pack_results(cfg))"),
],
"05_results.ipynb": [
("md", "# Results — Tables 4.6, 4.8, 4.10 and Figure 4.2\nAttach the output of notebook 04 (all runs)."),
("code", SETUP),
("code", "from bcd.report import tables, figure_rounds\nW = cfg.paths.work_dir\nt46, t48 = tables(f'{W}/runs', f'{W}/results')\nt46"),
("code", "t48"),
("code", "figure_rounds(f'{W}/runs', f'{W}/results/figure_4_2.png')\nfrom IPython.display import Image; Image(f'{W}/results/figure_4_2.png')"),
("code", "# Table 4.10: latency and memory on this GPU\nimport time, torch, numpy as np, glob\nfrom bcd.runtime import load_student, load_teacher, manifest, make_loader, part\nfrom bcd.detector import TumourDetector\nload = make_loader(cfg); paths = [p for p, _ in part(manifest(cfg), 'test')[:30]]\ndet = TumourDetector(glob.glob(f'{W}/yolo_runs/tumour/weights/best.pt')[0], device=0)\nst = load_student(cfg, adapter=f'{W}/runs/P_seed0/adapter', device=0)\ntorch.cuda.reset_peak_memory_stats(0); t = []\nfor p in paths:\n    s = time.time(); im = load(p); det(im); st.gradcam(im); t.append(time.time() - s)\nprint('regional pipeline s/case', np.mean(t), 'peak GB', torch.cuda.max_memory_allocated(0) / 1e9)\ntc = load_teacher(cfg, adapter=f'{W}/adapters/teacher', device=1); torch.cuda.reset_peak_memory_stats(1); t = []\nfor p in paths[:10]:\n    s = time.time(); tc.gradcam(load(p)); t.append(time.time() - s)\nprint('teacher s/case (excl. network)', np.mean(t), 'peak GB', torch.cuda.max_memory_allocated(1) / 1e9)"),
],
}

out = Path(__file__).resolve().parent.parent / "notebooks"
for name, cells in NB.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, out / name)
print("wrote", len(NB), "notebooks")
