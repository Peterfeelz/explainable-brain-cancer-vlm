"""YOLO tumour detector — Step 1 (Eq. 3.1), Increment 1."""
from __future__ import annotations
from pathlib import Path


class TumourDetector:
    def __init__(self, weights: str, conf_keep: float = 0.25, imgsz: int = 448, device=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf_keep, self.imgsz, self.device = conf_keep, imgsz, device

    @staticmethod
    def train(data_yaml: str, base_weights="yolov8n.pt", epochs=100, imgsz=448, batch=32,
              project="/kaggle/working/bcd/yolo_runs", device=0) -> str:
        from ultralytics import YOLO
        m = YOLO(base_weights)
        m.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, project=project,
                name="tumour", exist_ok=True, device=device, patience=20, verbose=False)
        best = Path(project) / "tumour" / "weights" / "best.pt"
        return str(best)

    def validate(self, data_yaml: str) -> dict:
        r = self.model.val(data=data_yaml, imgsz=self.imgsz, device=self.device, verbose=False)
        return {"precision": float(r.box.mp), "recall": float(r.box.mr),
                "map50": float(r.box.map50), "map50_95": float(r.box.map)}

    def __call__(self, image):
        """image: PIL image at working size. Returns list of (x1,y1,x2,y2,score) with score>=tau_YOLO,
        where score = objectness x class prob as reported by YOLO (Eq. 3.1)."""
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf_keep, device=self.device, verbose=False)[0]
        out = []
        for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s)))
        return out
