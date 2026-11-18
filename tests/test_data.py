import numpy as np, pandas as pd
from PIL import Image
from bcd.data import scan_kaggle, dedupe, make_partitions, export_yolo, attach_masks, phash64, nearest_hamming

def _img(path, seed, shape=(64, 64)):
    rng = np.random.default_rng(seed)
    a = (rng.random(shape) * 255).astype(np.uint8)
    Image.fromarray(a).resize((128, 128)).save(path)

def test_pipeline(tmp_path):
    root = tmp_path / "kaggle"
    k = 0
    for src, n in (("Training", 10), ("Testing", 3)):
        for d in ("glioma", "meningioma", "pituitary", "notumor"):
            (root / src / d).mkdir(parents=True)
            for i in range(n):
                _img(root / src / d / f"{d}_{i}.jpg", k); k += 1
    # exact duplicate inside Training and Testing->Training duplicate
    Image.open(root / "Training/glioma/glioma_0.jpg").save(root / "Training/glioma/glioma_dup.jpg")
    Image.open(root / "Training/glioma/glioma_1.jpg").save(root / "Testing/glioma/glioma_leak.jpg")
    df = scan_kaggle(root)
    kept, removed = dedupe(df, 4)
    assert len(removed) == 2
    man = make_partitions(kept, {"teacher_pool": .35, "seed": .1, "validation": .1, "stream": .45}, seed=1)
    assert set(man.partition) <= {"teacher_pool", "seed", "validation", "stream", "test"}
    assert (man.partition == "test").sum() == 12
    s = man[man.partition == "stream"].stream_order
    assert sorted(s) == list(range(len(s)))
    # figshare-like set that shares one image with the stream
    fsd = tmp_path / "fs"; (fsd / "images").mkdir(parents=True); (fsd / "masks").mkdir()
    rows = []
    stream_img = man[man.partition == "stream"].path.iloc[0]
    for i in range(6):
        ip = fsd / "images" / f"{i}.png"
        if i == 0:
            Image.open(stream_img).save(ip)
        else:
            _img(ip, 1000 + i)
        m = np.zeros((128, 128), np.uint8); m[30:60, 40:80] = 255
        Image.fromarray(m).save(fsd / "masks" / f"{i}.png")
        rows.append(dict(fs_id=str(i), fs_image=str(ip), fs_mask=str(fsd / "masks" / f"{i}.png"),
                         fs_label="glioma", pid=f"p{i % 3}", phash=phash64(ip)))
    fs = pd.DataFrame(rows)
    man = attach_masks(man, fs, 4)
    assert man.mask_path.notna().sum() >= 1
    y = export_yolo(fs, man, tmp_path / "yolo")
    labels = list((tmp_path / "yolo" / "labels").rglob("*.txt"))
    assert len(labels) == 5                            # the stream twin is excluded
    line = labels[0].read_text().split()
    assert line[0] == "0" and abs(float(line[3]) - 40 / 128) < 1e-3
