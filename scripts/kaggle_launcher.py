STEP = "08+06"       # v2.1: untrained vs trained (~1 h), then the demo app opens (run with Shift+Enter)
RUN_LIST = []
FORCE = []

import os, re, sys, glob, json, shutil, subprocess
def ver(c):
    m = re.search(r'__version__\s*=\s*"([\d.]+)"', open(os.path.join(c, "bcd", "__init__.py")).read())
    return tuple(int(x) for x in m.group(1).split(".")) if m else (0,)
cands = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/bcd/__init__.py", recursive=True)]
assert cands, "BCD project not found - click Add Input and add it"
code = max(cands, key=lambda c: (ver(c), "/datasets/" in c))
assert ver(code) >= (2, 1, 0), "Old code version attached - upload the new zip as a New Version of Bcd Project, then update the input"
for m in [m for m in sys.modules if m == "bcd" or m.startswith("bcd.")]:
    del sys.modules[m]
shutil.rmtree("/kaggle/working/bcd-project", ignore_errors=True)
shutil.copytree(code, "/kaggle/working/bcd-project")
sys.path.insert(0, "/kaggle/working/bcd-project")
print("code found at:", code, "| version", ".".join(map(str, ver(code))))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.45,<4.52", "peft>=0.12",
                "bitsandbytes>=0.43", "accelerate>=0.33", "ultralytics", "imagehash", "h5py"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
for m in [m for m in sys.modules if m.startswith(("peft", "torchao"))]:
    del sys.modules[m]
from bcd.config import load_config
cfg = load_config("/kaggle/working/bcd-project/configs/default.yaml")
def find_dir(test):
    for d, s, f in os.walk("/kaggle/input"):
        if test(d, s, f):
            return d
mri = find_dir(lambda d, s, f: "Training" in s and "Testing" in s)
assert mri, "Brain tumor MRI dataset not found - click Add Input and add it"
cfg.paths.kaggle_root = mri
shutil.rmtree(f"{cfg.paths.work_dir}/figshare", ignore_errors=True)
os.makedirs(cfg.paths.work_dir, exist_ok=True)
for prev in sorted(glob.glob("/kaggle/input/**/bcd/manifest.csv", recursive=True)):
    shutil.copytree(os.path.dirname(prev), cfg.paths.work_dir, dirs_exist_ok=True)
    print("restored earlier results from:", os.path.dirname(prev))
print("MRI dataset found at:", mri)
import gc
for one in STEP.split("+"):
    nb = json.load(open(glob.glob(f"/kaggle/working/bcd-project/notebooks/{one}_*.ipynb")[0]))
    print("=" * 20, "STEP", one, "=" * 20)
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] != "code" or "STEP 0: SETUP" in src or "--- Setup (run first)" in src:
            continue
        r = get_ipython().run_cell(src)
        if not r.success:
            raise RuntimeError("A step failed - see the error above")
    for k in ("st", "student", "teacher", "tc", "det", "rt", "Z", "df"):
        globals().pop(k, None)
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    print("STEP", one, "finished")
