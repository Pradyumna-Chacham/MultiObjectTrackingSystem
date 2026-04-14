from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from src.detectors.base import BaseDetector
from src.schemas import Detection


class UltralyticsAdapter(BaseDetector):
    def __init__(
        self,
        weights: str,
        conf_threshold: float,
        iou_threshold: float,
        classes: list[int],
        device: str,
    ) -> None:
        self.weights = str(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.device = device
        self.model: YOLO | None = None

    @property
    def name(self) -> str:
        return "ultralytics"

    def load(self) -> None:
        self.model = YOLO(self.weights)
        self.model.to(self.device)

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("Detector not loaded. Call load() first.")

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes if self.classes else None,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            names = result.names
            boxes = result.boxes

            if boxes is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cls_id in zip(xyxy, confs, clss):
                detections.append(
                    Detection(
                        frame_index=frame_index,
                        bbox=[float(v) for v in bbox.tolist()],
                        class_id=int(cls_id),
                        class_name=str(names[int(cls_id)]),
                        confidence=float(conf),
                    )
                )

        return detections
