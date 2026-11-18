"""Composite confidence gate — Steps 1-8 of Section 3.4 (Equations 3.1-3.14).

Pure NumPy so it can be unit-tested without a GPU. Boxes are (x1, y1, x2, y2, score)
in pixel coordinates of the square working image; masks are boolean HxW arrays.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------- Step 2 (Eq. 3.2-3.4)
def softmax(z, T: float = 1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def norm_entropy(p):
    p = np.asarray(p, dtype=np.float64)
    C = p.shape[-1]
    return float(-(p * np.log(p + EPS)).sum(-1) / np.log(C))


# ---------------------------------------------------------------- Step 3 (Eq. 3.5)
def g_base(p, alpha: float):
    return float(alpha * np.max(p) + (1 - alpha) * (1 - norm_entropy(p)))


# ---------------------------------------------------------------- Step 4 helpers
def cam_to_mask(cam: np.ndarray, top_mass: float = 0.10) -> np.ndarray:
    """Binary mask keeping the smallest set of pixels holding `top_mass` of attribution mass."""
    cam = np.clip(np.asarray(cam, dtype=np.float64), 0, None)
    if cam.max() <= 0:
        return np.zeros(cam.shape, dtype=bool)
    cam = cam / cam.max()
    flat = np.sort(cam.ravel())[::-1]
    cum = np.cumsum(flat)
    k = int(np.searchsorted(cum, top_mass * cum[-1])) + 1
    thr = flat[min(k, len(flat)) - 1]
    return cam >= thr


def bbox_of_mask(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def boxes_to_mask(boxes, shape) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
    return m


# ---------------------------------------------------------------- Step 4 (Eq. 3.8)
def a_yolo(mask: np.ndarray, boxes, positive: bool) -> float:
    if not boxes:
        return 0.0 if positive else 1.0          # nothing corroborates a positive / both agree on negative
    if not positive:
        return 0.0                               # detector sees a lesion the VLM does not
    mb = bbox_of_mask(mask)
    return max(iou(mb, b[:4]) for b in boxes)


# ---------------------------------------------------------------- Step 5 (Eq. 3.9)
def a_xai(mask: np.ndarray, ref: np.ndarray | None, positive: bool) -> float:
    if ref is None or ref.sum() == 0:
        return 0.0 if positive else 1.0
    if mask.sum() == 0:
        return 0.0
    return float((mask & ref).sum() / mask.sum())


# ---------------------------------------------------------------- v2.0 class-aware grounding terms
GROUNDINGS: dict = {}            # key -> Grounding (kept outside GateConfig so configs stay cheap to copy)


def box_features(boxes, size: float = 448.0) -> np.ndarray:
    """Features of the YOLO evidence: [has_box, top score, centre x, centre y, width, height, n_boxes]
    (coordinates of the highest-scoring box, normalised to [0, 1])."""
    if not boxes:
        return np.zeros(7)
    b = max(boxes, key=lambda r: r[4])
    x1, y1, x2, y2, s = [float(v) for v in b[:5]]
    return np.array([1.0, s, (x1 + x2) / 2 / size, (y1 + y2) / 2 / size, (x2 - x1) / size, (y2 - y1) / size,
                     float(len(boxes))])


def xai_raw(mask: np.ndarray, ref, boxes, positive: bool) -> float:
    """Raw attribution-grounding value before class-relative scaling.
    Tumour prediction: share of the attribution mask inside the reference region (expert mask, else YOLO boxes;
    0 if there is no reference). No-tumour prediction: share of the attribution mask OUTSIDE any YOLO box
    (1 if YOLO found nothing), i.e. the evidence does not sit on a detected lesion."""
    if positive:
        if ref is None or ref.sum() == 0 or mask.sum() == 0:
            return 0.0
        return float((mask & ref).sum() / mask.sum())
    if not boxes or mask.sum() == 0:
        return 1.0
    bm = boxes_to_mask(boxes, mask.shape)
    return float(1.0 - (mask & bm).sum() / mask.sum())


class Grounding:
    """v2.0 grounding (Section 3.4.4-3.4.5, revised after RQ1):
    A_YOLO = lesion-location agreement: probability, under a classifier trained on the YOLO boxes of the
             TRAINING partitions (seed + teacher pool, ground-truth labels), that a lesion found where YOLO
             found it (or no lesion) belongs to the predicted class. Targets the main error: tumour-type confusion.
    A_XAI  = class-relative grounding: percentile of the raw attribution value among VALIDATION predictions of
             the same class, so the term no longer rewards one class (the v1 terms equalled 1 for 'no tumour')."""
    def __init__(self, n_classes: int, size: float = 448.0):
        self.C, self.size = n_classes, size
        self.loc = None
        self.ref = {}

    def fit_location(self, boxes_list, y, seed: int = 0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        X = np.stack([box_features(b, self.size) for b in boxes_list]); y = np.asarray(y)
        self.loc = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=200,
                                                  l2_regularization=1.0, random_state=seed).fit(X, y)
        return self

    def location_probs(self, boxes) -> np.ndarray:
        if self.loc is None:
            return np.full(self.C, 1.0 / self.C)
        q = np.zeros(self.C)
        q[self.loc.classes_] = self.loc.predict_proba(box_features(boxes, self.size)[None])[0]
        return q

    def fit_xai(self, raw, y_hat):
        raw, y_hat = np.asarray(raw, float), np.asarray(y_hat)
        self.ref = {int(c): np.sort(raw[y_hat == c]) for c in np.unique(y_hat) if (y_hat == c).sum() >= 5}
        return self

    def xai_rel(self, raw: float, y: int) -> float:
        r = self.ref.get(int(y))
        if r is None or len(r) == 0:
            return float(raw)
        # mid-rank percentile (ties share the middle), in [0, 1]
        lo, hi = np.searchsorted(r, raw, "left"), np.searchsorted(r, raw, "right")
        return float((lo + hi) / 2 / len(r))


# ---------------------------------------------------------------- Step 6 (Eq. 3.10-3.11)
@dataclass
class GateConfig:
    alpha: float = 0.6
    w1: float = 0.5
    w2: float = 0.25
    w3: float = 0.25
    tau_gate: float = 0.70
    tau_xai_min: float = 0.2
    tau_yolo_high: float = 0.6
    cam_top_mass: float = 0.10
    use_yolo: bool = True          # ablation A1
    use_xai: bool = True           # ablation A2
    use_overrides: bool = True
    grounding: str | None = None   # v2.0: key into GROUNDINGS; None = original (v1) terms

    @classmethod
    def from_cfg(cls, g, **kw):
        base = {k: g[k] for k in ("alpha", "w1", "w2", "w3", "tau_gate", "tau_xai_min",
                                  "tau_yolo_high", "cam_top_mass") if k in g}
        base.update(kw)
        return cls(**base)


@dataclass
class GateResult:
    y_hat: int
    conf: float
    h_norm: float
    g_base: float
    a_yolo: float | None
    a_xai: float | None
    g: float
    override: str | None
    decision: str                 # "accept" | "escalate"
    extras: dict = field(default_factory=dict)


def composite_gate(p, cam, boxes, ref_mask, no_tumour_idx: int, cfg: GateConfig) -> GateResult:
    """p: calibrated class probabilities; cam: HxW attribution (or None); boxes: YOLO boxes
    kept at tau_YOLO; ref_mask: annotated tumour mask or None (falls back to YOLO boxes)."""
    p = np.asarray(p, dtype=np.float64)
    y = int(np.argmax(p))
    positive = y != no_tumour_idx
    gb = g_base(p, cfg.alpha)
    h = norm_entropy(p)
    ay = ax = None
    terms, weights = [gb], [cfg.w1]
    mask = None
    if cam is not None and (cfg.use_yolo or cfg.use_xai):
        mask = cam_to_mask(cam, cfg.cam_top_mass)
    gr = GROUNDINGS.get(cfg.grounding) if cfg.grounding else None
    ax_raw = None
    if cfg.use_yolo:
        if gr is not None:
            ay = float(gr.location_probs(boxes)[y])
        else:
            ay = a_yolo(mask if mask is not None else np.zeros((1, 1), bool), boxes, positive)
        terms.append(ay); weights.append(cfg.w2)
    if cfg.use_xai:
        ref = ref_mask
        m0 = mask if mask is not None else np.zeros((1, 1), bool)
        if ref is None and boxes and mask is not None:
            ref = boxes_to_mask(boxes, mask.shape)
        if gr is not None:
            ax_raw = xai_raw(m0, ref, boxes, positive)
            ax = gr.xai_rel(ax_raw, y)
        else:
            ax = a_xai(m0, ref, positive)
        terms.append(ax); weights.append(cfg.w3)
    weights = np.asarray(weights) / np.sum(weights)      # renormalise when a term is ablated
    g = float(np.dot(weights, terms))
    override = None
    if cfg.use_overrides:
        if positive and cfg.use_xai and ax is not None and ax < cfg.tau_xai_min:
            override = "ungrounded_positive"
        elif (not positive) and boxes and max(b[4] for b in boxes) >= cfg.tau_yolo_high:
            override = "possible_missed_tumour"
    decision = "escalate" if (g < cfg.tau_gate or override) else "accept"
    return GateResult(y, float(p[y]), h, gb, ay, ax, g, override, decision,
                      extras={"mask_px": int(mask.sum()) if mask is not None else 0, "a_xai_raw": ax_raw})


# ---------------------------------------------------------------- Step 7 (Eq. 3.12)
def kl_div(p, q) -> float:
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    return float((p * (np.log(p + EPS) - np.log(q + EPS))).sum())


def cloud_route(local: GateResult, cloud: GateResult, p_local, p_cloud, positive_needs_plan_review: bool,
                tau_cloud: float, kl_delta: float) -> tuple[str, list[str]]:
    """Return ("clinician"|"pseudo_label", reasons) per Section 3.4.7."""
    reasons = []
    if cloud.g < tau_cloud:
        reasons.append("low_cloud_trust")
    if local.y_hat != cloud.y_hat and kl_div(p_cloud, p_local) > kl_delta:
        reasons.append("disagreement")
    if positive_needs_plan_review:
        reasons.append("plan_review")
    return ("clinician" if reasons else "pseudo_label"), reasons


# ---------------------------------------------------------------- Step 8 (Eq. 3.13-3.14)
class AgreementTracker:
    def __init__(self, window: int = 50, prior: float = 0.8):
        self.window, self.prior, self.a = window, prior, []

    def add(self, agrees: bool):
        self.a.append(1.0 if agrees else 0.0)
        self.a = self.a[-self.window:]

    @property
    def value(self) -> float:
        return float(np.mean(self.a)) if self.a else self.prior
