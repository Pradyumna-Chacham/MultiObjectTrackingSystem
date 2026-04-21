from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import RTDETR


class RTDETRDetector:
    """
    RT-DETR detector wrapper.

    Returns detections in normalized internal format:
    {
        "bbox": [x1, y1, x2, y2],
        "score": float,
        "class_id": int,
        "label": str,
    }
    """

    def __init__(
        self,
        model_path: str = "rtdetr-l.pt",
        conf_threshold: float = 0.25,
        device: str = "cpu",
        classes: list[int] | None = None,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.classes = classes

        self.model = RTDETR(model_path)
        self.names = self.model.names if hasattr(self.model, "names") else {}

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detection dicts
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []

        for box, score, class_id in zip(xyxy, conf, cls):
            x1, y1, x2, y2 = map(float, box.tolist())
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": float(score),
                    "class_id": int(class_id),
                    "label": self.names.get(int(class_id), str(class_id)),
                }
            )

        return detections