# Running on Kaggle without importing notebooks

1. Upload bcd-project.zip as a Kaggle Dataset (or New Version of your existing one).
2. On that dataset's page click  Code -> New Notebook  (the dataset is attached automatically).
3. Right panel: Add Input -> "brain tumor mri dataset" (Masoud Nickparvar); for step 01 also a
   figshare brain tumor dataset with .mat files; for steps 02-05 also Your Work -> earlier notebooks.
   Session options: Accelerator = GPU T4 x2, Internet = On.
4. Paste this into the first cell and press Shift+Enter (change STEP to 02, 03, 04 or 05 later):

```python
STEP = "01"          # 01, 02, 03, 04 or 05
# RUN_LIST = [('P', 0), ('B0', 0)]   # step 04 only: which conditions to run this session
import glob
nb = glob.glob(f"/kaggle/input/**/notebooks/{STEP}_*.ipynb", recursive=True)
assert nb, "Attach the bcd-project dataset (Add Input)"
print("running", nb[0])
get_ipython().run_line_magic("run", f'-i "{nb[0]}"')
```

5. When it finishes: Save Version -> Save & Run All (Commit).
