"""Plain-language explanations of a gate decision, and the bounding-box mismatch prompt (v2.1).

The explanation is built only from quantities the system computed (class probabilities, YOLO boxes,
the lesion-location model, the attribution overlap, the gate terms and the routing rules), so every sentence
can be traced to a number shown on screen. The models' own free-text descriptions are shown separately and
labelled as such, because a model's self-description need not reflect how it actually decided.
"""
from __future__ import annotations
import numpy as np

NICE = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary tumour", "no_tumour": "no tumour"}
TYPICAL = {"glioma": "inside the brain tissue", "meningioma": "at the surface of the brain, next to the skull or dura",
           "pituitary": "in the midline at the skull base (pituitary region)", "no_tumour": "—"}


def top_box(boxes):
    return max(boxes, key=lambda b: b[4]) if boxes else None


def region_name(box, size: float = 448.0) -> str:
    """Rough anatomical description of where a box lies on an axial/coronal slice."""
    x1, y1, x2, y2 = [float(v) / size for v in box[:4]]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    vert = "upper" if cy < 0.4 else ("lower" if cy > 0.62 else "middle")
    horiz = "left side of the image" if cx < 0.42 else ("right side of the image" if cx > 0.58 else "midline")
    r = np.hypot(cx - 0.5, cy - 0.5)
    depth = "near the brain surface" if r > 0.30 else ("deep, inside the brain tissue" if r < 0.18 else "within the brain")
    if horiz == "midline" and vert == "lower":
        return "in the lower midline (skull-base / sella region)"
    return f"in the {vert} part, {horiz}, {depth}"


def explain(classes, p_local, gate, boxes, grounding=None, p_cloud=None, route=None, reasons=None,
            weights=None, tau=None, size: float = 448.0) -> str:
    """Markdown explanation of one case."""
    nt = classes.index("no_tumour")
    order = np.argsort(-np.asarray(p_local))
    y, y2 = int(order[0]), int(order[1])
    c1, c2 = classes[y], classes[y2]
    L = []
    pl = np.asarray(p_local)
    if pl[y] >= 0.85:
        L.append(f"**Local model:** confident it is **{NICE[c1]}** ({pl[y]:.0%}).")
    else:
        L.append(f"**Local model:** leans to **{NICE[c1]}** ({pl[y]:.0%}) but also considers {NICE[c2]} ({pl[y2]:.0%}), "
                 f"so it is uncertain.")
    b = top_box(boxes)
    if b is None:
        L.append("**Detector (YOLO):** found no lesion on this slice"
                 + (", which agrees with 'no tumour'." if y == nt else ", so nothing independent supports a tumour diagnosis."))
    else:
        where = region_name(b, size)
        s = f"**Detector (YOLO):** found a possible lesion {where} (detector score {b[4]:.2f})."
        if grounding is not None and getattr(grounding, "loc", None) is not None:
            q = grounding.location_probs(boxes)
            k = int(np.argmax(q))
            s += f" Lesions found in this position were most often **{NICE[classes[k]]}** in the training images ({q[k]:.0%})."
            if y == nt:
                s += " The local model's 'no tumour' answer conflicts with the detected lesion."
            elif q[y] < 0.3:
                s += (f" A {NICE[c1]} would usually lie {TYPICAL[c1]}, so the location **does not support** the local "
                      f"answer (location agreement {q[y]:.2f}).")
            else:
                s += f" The location **supports** the local answer (location agreement {q[y]:.2f})."
        L.append(s)
    if gate.a_xai is not None and gate.extras.get("a_xai_raw") is not None:
        raw = gate.extras["a_xai_raw"]
        if y != nt:
            L.append(f"**Where the local model looked (Grad-CAM):** {raw:.0%} of its strongest attention lies on the lesion "
                     f"region; compared with other {NICE[c1]} predictions this is at the {gate.a_xai:.0%} level. "
                     f"Attention maps of this model are diffuse, so this is shown for the clinician rather than relied on.")
        else:
            L.append(f"**Where the local model looked (Grad-CAM):** {raw:.0%} of its strongest attention lies away from any detected lesion.")
    if weights is not None and tau is not None:
        L.append(f"**Gate:** confidence term {gate.g_base:.2f}, location term "
                 f"{'—' if gate.a_yolo is None else f'{gate.a_yolo:.2f}'}, attention term "
                 f"{'—' if gate.a_xai is None else f'{gate.a_xai:.2f}'}; weighted score G = {gate.g:.2f} against the "
                 f"threshold {tau:.2f} → **{'accepted locally' if gate.decision == 'accept' else 'sent to the cloud model'}**"
                 + (f" (safety rule: {gate.override.replace('_', ' ')})" if gate.override else "") + ".")
    if p_cloud is not None:
        pc = np.asarray(p_cloud); yc = int(np.argmax(pc))
        s = f"**Cloud model:** says **{NICE[classes[yc]]}** ({pc[yc]:.0%})."
        if yc == nt and b is not None:
            s += " It reports no tumour although the detector found a lesion."
        if yc != y:
            s += " It **disagrees** with the local model."
        else:
            s += " It agrees with the local model."
        L.append(s)
    if route == "clinician":
        why = ", ".join(r.replace("_", " ") for r in (reasons or []))
        L.append(f"**Decision:** a specialist must review this case ({why}).")
    elif route:
        L.append("**Decision:** the cloud answer is accepted and kept as a training label for the local model.")
    return "\n\n".join(L)


def mismatch(classes, p_local, gate, boxes, grounding=None, threshold: float = 0.3) -> bool:
    """True when the detector's evidence and the local prediction disagree."""
    nt = classes.index("no_tumour"); y = int(np.argmax(p_local))
    if gate.override == "possible_missed_tumour":
        return True
    if y != nt and not boxes:
        return True
    return y != nt and gate.a_yolo is not None and gate.a_yolo < threshold and grounding is not None


def mismatch_prompt(classes, p_local, boxes, grounding=None, size: float = 448.0) -> str:
    """Prompt sent with the image to the cloud model when the bounding box and the local prediction disagree."""
    nt = classes.index("no_tumour"); y = int(np.argmax(p_local)); c = classes[y]
    b = top_box(boxes)
    if b is None:
        return (f"A smaller model suggested '{NICE[c]}' for this brain MRI slice, but a lesion detector found no lesion. "
                "Look carefully: is a tumour visible? If so, where is it and what type is it most likely to be "
                "(glioma, meningioma, pituitary tumour)? Answer in at most three sentences.")
    x1, y1, x2, y2 = [int(v) for v in b[:4]]
    where = region_name(b, size)
    if y == nt:
        q = ("A smaller model answered 'no tumour', but a lesion detector marked a possible lesion "
             f"{where} (box from x={x1}, y={y1} to x={x2}, y={y2} in a {int(size)}×{int(size)} image). ")
    else:
        q = (f"A smaller model suggested '{NICE[c]}', but a lesion detector marked the lesion {where} "
             f"(box from x={x1}, y={y1} to x={x2}, y={y2} in a {int(size)}×{int(size)} image), "
             f"where a {NICE[c]} is unusual. ")
    return q + ("Examine the marked region. Is there a tumour there, and which type does its location and appearance "
                "suggest (glioma, meningioma, pituitary tumour, or none)? Answer in at most three sentences.")


SELF_EXPLAIN = ("In two sentences, describe the image features that support your diagnosis of '{dx}' for this brain "
                "MRI slice, and where in the image they are.")
