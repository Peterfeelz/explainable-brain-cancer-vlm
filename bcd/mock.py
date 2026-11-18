"""CPU mocks of the detector, student, teacher and trainer — used for tests and dry runs only.
They produce synthetic scores; numbers from mocks are NOT results and must never be reported."""
from __future__ import annotations
import hashlib
import numpy as np


def _h(path, salt=""):
    return int(hashlib.md5((salt + str(path)).encode()).hexdigest()[:8], 16)


class MockWorld:
    """Synthetic cases: each path has a true label, a lesion box and a 'difficulty'."""
    def __init__(self, n=400, n_classes=4, no_tumour=3, size=64, seed=0):
        rng = np.random.default_rng(seed)
        self.size, self.no_tumour, self.C = size, no_tumour, n_classes
        self.cases = {}
        for i in range(n):
            y = int(rng.integers(0, n_classes))
            x1, y1 = rng.integers(5, size - 25, 2)
            box = (float(x1), float(y1), float(x1 + 18), float(y1 + 18))
            self.cases[f"case_{i:04d}.png"] = dict(y=y, box=box if y != no_tumour else None,
                                                   difficulty=float(rng.uniform(0, 1)))

    def split(self, a, b):
        keys = sorted(self.cases)[a:b]
        return [(k, self.cases[k]["y"]) for k in keys]

    def ref_mask(self, path):
        c = self.cases[path]
        m = np.zeros((self.size, self.size), bool)
        if c["box"] is not None:
            x1, y1, x2, y2 = map(int, c["box"]); m[y1:y2, x1:x2] = True
        return m

    def detector_boxes(self, paths, miss=0.1, fp=0.05, seed=0):
        rng = np.random.default_rng(seed); out = {}
        for p in paths:
            c = self.cases[p]; b = []
            if c["box"] is not None and rng.random() > miss:
                b.append((*c["box"], float(rng.uniform(0.4, 0.95))))
            if c["box"] is None and rng.random() < fp:
                b.append((10.0, 10.0, 28.0, 28.0, float(rng.uniform(0.3, 0.7))))
            out[p] = b
        return out


class MockModel:
    """Skill in [0,1]: probability of a correct, well-grounded prediction rises with skill."""
    def __init__(self, world, skill=0.55, salt="s"):
        self.w, self.skill, self.salt, self.T = world, skill, salt, 1.0

    def _draw(self, path):
        c = self.w.cases[path]
        r = np.random.default_rng(_h(path, self.salt + f"{self.skill:.3f}"))
        correct = r.random() < self.skill * (1.2 - 0.5 * c["difficulty"])
        y = c["y"] if correct else int((c["y"] + r.integers(1, self.w.C)) % self.w.C)
        z = r.normal(0, 1, self.w.C); z[y] += 2.5 if correct else r.uniform(0.5, 2.5)
        return z, y, correct, r

    def logits(self, path):
        return self._draw(path)[0]

    def gradcam(self, path, T=1.0):
        z, y, correct, r = self._draw(path)
        cam = np.zeros((self.w.size, self.w.size))
        c = self.w.cases[path]
        if c["box"] is not None and correct:
            x1, y1, x2, y2 = map(int, c["box"]); cam[y1:y2, x1:x2] = 1.0
        else:
            x0, y0 = r.integers(0, self.w.size - 16, 2); cam[y0:y0 + 16, x0:x0 + 16] = 1.0
        return z, cam

    def evaluate(self, paths, y, T=1.0):
        from .metrics import classification_metrics
        from .gate import softmax
        z = np.stack([self.logits(p) for p in paths]); pr = softmax(z, T)
        return classification_metrics(y, pr.argmax(1), pr, self.w.no_tumour), z

    def distill(self, buffer, seed=0, cumulative=False):
        verified = sum(1 for e in buffer if e["weight"] >= 1.0)
        self.skill = min(0.95, self.skill + 0.002 * verified + 0.0005 * len(buffer))
        return {"promoted": True}


class MockTeacher(MockModel):
    def cached(self, path):
        z, cam = self.gradcam(path)
        return {"z": z, "cam": cam, "plan_fn": lambda yname: f"(mock plan for {yname})"}
