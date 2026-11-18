"""Data pipeline — Increment 1 (Section 3.5).

* scan the Kaggle 4-class dataset (Nickparvar, 2021)
* remove near-duplicates with perceptual hashing
* load figshare .mat files (Cheng et al., 2015) for tumour masks
* match Kaggle images to figshare masks (reference region R_ref for A_XAI)
* make stratified partitions (teacher pool / seed / validation / stream; Kaggle Testing = test)
* export a YOLO detection dataset that does not overlap validation, stream or test
"""
from __future__ import annotations
import json, os, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

KAGGLE_DIRS = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary", "notumor": "no_tumour"}
FIGSHARE_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}   # cjdata.label coding
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------ images
def load_image(path, size: int = 448) -> Image.Image:
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return im.convert("RGB")


def load_mask(path, size: int = 448) -> np.ndarray | None:
    if path is None or (isinstance(path, float) and np.isnan(path)) or not os.path.exists(str(path)):
        return None
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(m) > 127


def phash64(path_or_img) -> int:
    import imagehash
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    return int(str(imagehash.phash(im.convert("L"), hash_size=8)), 16)


def _popcount64(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.uint64)
    bits = np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1)
    return bits.sum(-1).astype(np.int64)


def nearest_hamming(query: np.ndarray, ref: np.ndarray, chunk: int = 512):
    """For each query hash return (index of nearest ref hash, distance)."""
    query = np.asarray(query, np.uint64); ref = np.asarray(ref, np.uint64)
    idx = np.empty(len(query), int); dist = np.empty(len(query), int)
    for s in range(0, len(query), chunk):
        q = query[s:s + chunk, None] ^ ref[None, :]
        d = _popcount64(q)
        idx[s:s + chunk] = d.argmin(1); dist[s:s + chunk] = d.min(1)
    return idx, dist


# ------------------------------------------------------------------ Kaggle
def scan_kaggle(root) -> pd.DataFrame:
    rows = []
    for src in ("Training", "Testing"):
        for d, lab in KAGGLE_DIRS.items():
            folder = Path(root) / src / d
            if not folder.exists():
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    rows.append({"path": str(f), "label": lab, "source": src})
    if not rows:
        raise FileNotFoundError(f"No images found under {root}/Training|Testing/<class>/")
    return pd.DataFrame(rows)


def dedupe(df: pd.DataFrame, hamming: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop near-duplicate Training images, and Testing images that duplicate Training ones.
    Returns (kept, removed)."""
    df = df.copy()
    if "phash" not in df:
        df["phash"] = [phash64(p) for p in df.path]
    hashes = df.phash.to_numpy(np.uint64)
    keep = np.ones(len(df), bool)
    tr = np.where(df.source.to_numpy() == "Training")[0]
    te = np.where(df.source.to_numpy() == "Testing")[0]
    seen = []
    for i in tr:                                   # greedy within Training
        if seen:
            _, d = nearest_hamming(hashes[[i]], hashes[seen])
            if d[0] <= hamming:
                keep[i] = False; continue
        seen.append(i)
    if len(te) and seen:                           # Testing vs kept Training
        _, d = nearest_hamming(hashes[te], hashes[seen])
        keep[te[d <= hamming]] = False
    return df[keep].reset_index(drop=True), df[~keep].reset_index(drop=True)


# ------------------------------------------------------------------ figshare
def _read_cjdata(path):
    """Read one figshare cjdata file (MATLAB v7.3/HDF5, with a scipy fallback for older formats).
    Returns (image, mask, label, pid) or None if the file is not a cjdata image (e.g. cvind.mat)."""
    try:
        import h5py
        with h5py.File(path, "r") as h:
            if "cjdata" not in h:
                return None
            cj = h["cjdata"]
            img = np.array(cj["image"]).T.astype(np.float32)
            mask = np.array(cj["tumorMask"]).T.astype(bool)
            label = int(np.array(cj["label"]).ravel()[0])
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel())
            return img, mask, label, pid
    except OSError:                       # not HDF5 -> try classic MATLAB format
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "cjdata" not in m:
            return None
        cj = m["cjdata"]
        return (np.asarray(cj.image, np.float32), np.asarray(cj.tumorMask).astype(bool),
                int(np.ravel(cj.label)[0]), str(cj.PID))


def find_figshare_dir(root="/kaggle/input"):
    """Folder that holds the figshare cjdata .mat files (ignores cvind.mat and other extras)."""
    from collections import Counter
    c = Counter(str(p.parent) for p in Path(root).rglob("*.mat") if p.stem.lower() != "cvind")
    return c.most_common(1)[0][0] if c else None


def convert_figshare(mat_dir, out_dir, size: int = 448) -> pd.DataFrame:
    """Convert figshare cjdata .mat files to PNG image + mask pairs."""
    out = Path(out_dir); (out / "images").mkdir(parents=True, exist_ok=True); (out / "masks").mkdir(exist_ok=True)
    rows, skipped = [], 0
    for f in sorted(Path(mat_dir).rglob("*.mat")):
        r = _read_cjdata(f)
        if r is None:
            skipped += 1; continue
        img, mask, lab, pid = r
        label = FIGSHARE_LABELS[lab]
        img = (255 * (img - img.min()) / max(img.max() - img.min(), 1e-6)).astype(np.uint8)
        ip, mp = out / "images" / f"{f.stem}.png", out / "masks" / f"{f.stem}.png"
        Image.fromarray(img).resize((size, size), Image.BILINEAR).save(ip)
        Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.NEAREST).save(mp)
        rows.append({"fs_id": f.stem, "fs_image": str(ip), "fs_mask": str(mp), "fs_label": label, "pid": pid})
    if not rows:
        raise FileNotFoundError(f"No figshare cjdata .mat files found under {mat_dir}")
    df = pd.DataFrame(rows)
    df["phash"] = [phash64(p) for p in df.fs_image]
    df.to_csv(out / "figshare_index.csv", index=False)
    print(f"figshare: converted {len(df)} images (skipped {skipped} non-image .mat files)")
    return df


def attach_masks(kag: pd.DataFrame, fs: pd.DataFrame, hamming: int = 4) -> pd.DataFrame:
    """Give each Kaggle image the figshare mask of its near-duplicate (if any)."""
    kag = kag.copy(); kag["mask_path"] = None; kag["fs_id"] = None
    if fs is None or len(fs) == 0:
        return kag
    idx, d = nearest_hamming(kag.phash.to_numpy(np.uint64), fs.phash.to_numpy(np.uint64))
    hit = d <= hamming
    kag.loc[hit, "mask_path"] = fs.fs_mask.to_numpy()[idx[hit]]
    kag.loc[hit, "fs_id"] = fs.fs_id.to_numpy()[idx[hit]]
    return kag


# ------------------------------------------------------------------ partitions
def make_partitions(df: pd.DataFrame, shares: dict, seed: int = 42, stream_limit=None, test_limit=None) -> pd.DataFrame:
    """Stratified split of the Training images; Testing images become 'test'."""
    rng = random.Random(seed)
    df = df.copy(); df["partition"] = None
    df.loc[df.source == "Testing", "partition"] = "test"
    names = ["teacher_pool", "seed", "validation", "stream"]
    fr = np.array([shares[n] for n in names], float); fr = fr / fr.sum()
    for lab, g in df[df.source == "Training"].groupby("label"):
        ids = list(g.index); rng.shuffle(ids)
        cuts = np.round(np.cumsum(fr) * len(ids)).astype(int)
        start = 0
        for n, c in zip(names, cuts):
            df.loc[ids[start:c], "partition"] = n; start = c
    # stream order is fixed (shuffled once) and optionally capped
    stream = df[df.partition == "stream"].sample(frac=1.0, random_state=seed)
    if stream_limit:
        drop = stream.index[stream_limit:]; df.loc[drop, "partition"] = "unused"; stream = stream.iloc[:stream_limit]
    df["stream_order"] = -1
    df.loc[stream.index, "stream_order"] = np.arange(len(stream))
    if test_limit:
        t = df[df.partition == "test"].sample(frac=1.0, random_state=seed)
        df.loc[t.index[test_limit:], "partition"] = "unused"
    return df


def export_yolo(fs: pd.DataFrame, kag: pd.DataFrame, out_dir, val_frac: float = 0.2, seed: int = 42) -> str:
    """Single-class ('tumour') YOLO dataset from figshare masks, excluding any figshare image whose
    Kaggle twin is in validation, stream or test (prevents detector leakage). Split by patient."""
    out = Path(out_dir)
    blocked = set(kag[kag.partition.isin(["validation", "stream", "test"])].fs_id.dropna())
    fs = fs[~fs.fs_id.isin(blocked)].copy()
    pids = sorted(fs.pid.unique()); random.Random(seed).shuffle(pids)
    val_p = set(pids[:max(1, int(len(pids) * val_frac))])
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True); (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    from .gate import bbox_of_mask
    n = {"train": 0, "val": 0}
    for _, r in fs.iterrows():
        split = "val" if r.pid in val_p else "train"
        m = np.asarray(Image.open(r.fs_mask)) > 127
        bb = bbox_of_mask(m)
        if bb is None:
            continue
        H, W = m.shape
        x1, y1, x2, y2 = bb
        dst = out / "images" / split / f"{r.fs_id}.png"
        if not dst.exists():
            os.symlink(os.path.abspath(r.fs_image), dst)
        (out / "labels" / split / f"{r.fs_id}.txt").write_text(
            f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        n[split] += 1
    yaml_path = out / "data.yaml"
    yaml_path.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: tumour\n")
    (out / "counts.json").write_text(json.dumps(n))
    return str(yaml_path)


def prepare_all(cfg) -> pd.DataFrame:
    """Run the whole Increment-1 data preparation and write manifest.csv to work_dir."""
    work = Path(cfg.paths.work_dir); work.mkdir(parents=True, exist_ok=True)
    kag = scan_kaggle(cfg.paths.kaggle_root)
    kag["phash"] = [phash64(p) for p in kag.path]
    kept, removed = dedupe(kag, cfg.data.dedupe_hamming)
    removed.to_csv(work / "duplicates_removed.csv", index=False)
    fs = None
    if cfg.paths.figshare_root:
        fs_csv = work / "figshare" / "figshare_index.csv"
        fs = pd.read_csv(fs_csv) if fs_csv.exists() else convert_figshare(cfg.paths.figshare_root, work / "figshare", cfg.data.image_size)
    kept = attach_masks(kept, fs, cfg.data.dedupe_hamming)
    shares = {k: cfg.data[k] for k in ("teacher_pool", "seed", "validation", "stream")}
    man = make_partitions(kept, shares, cfg.seed, cfg.data.stream_limit, cfg.data.test_limit)
    classes = list(cfg.classes)
    man["y"] = man.label.map({c: i for i, c in enumerate(classes)})
    man.to_csv(work / "manifest.csv", index=False)
    if fs is not None:
        export_yolo(fs, man, work / "yolo", seed=cfg.seed)
    summary = man.groupby(["partition", "label"]).size().unstack(fill_value=0)
    summary.to_csv(work / "partition_summary.csv")
    return man
