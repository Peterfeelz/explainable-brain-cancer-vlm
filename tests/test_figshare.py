import numpy as np, h5py
from bcd.data import convert_figshare, find_figshare_dir

def _cj(path, label, pid):
    with h5py.File(path, "w") as h:
        g = h.create_group("cjdata")
        img = np.random.default_rng(label).integers(0, 4000, (512, 512)).astype(np.int16)
        m = np.zeros((512, 512), np.uint8); m[100:200, 150:300] = 1
        g["image"] = img.T; g["tumorMask"] = m.T; g["label"] = np.array([[label]], float)
        g["PID"] = np.array([[ord(c)] for c in pid], np.uint16)

def test_convert_with_cvind(tmp_path):
    d = tmp_path / "in" / "dataset" / "data"; d.mkdir(parents=True)
    for i in range(1, 5):
        _cj(d / f"{i}.mat", (i % 3) + 1, f"P{i}")
    with h5py.File(tmp_path / "in" / "dataset" / "cvind.mat", "w") as h:
        h["cvind"] = np.arange(4)
    root = find_figshare_dir(tmp_path / "in")
    assert root.endswith("data")
    df = convert_figshare(tmp_path / "in" / "dataset", tmp_path / "out", 64)   # parent folder incl. cvind.mat
    assert len(df) == 4 and set(df.fs_label) <= {"meningioma", "glioma", "pituitary"}
    assert df.pid.iloc[0] == "P1"
