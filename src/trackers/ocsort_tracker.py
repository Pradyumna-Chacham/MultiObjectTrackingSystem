from __future__ import annotations

from typing import Any

import numpy as np
from boxmot.trackers.ocsort.ocsort import OcSort

from src.schemas import Detection, Track


class OCSORTTracker:
    def __init__(
        self,
        det_thresh: float = 0.25,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        per_class: bool = False,
        min_conf: float = 0.1,
        delta_t: int = 3,
        inertia: float = 0.2,
        use_byte: bool = False,
        Q_xy_scaling: float = 0.01,
        Q_s_scaling: float = 0.0001,
    ) -> None:
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.per_class = per_class
        self.min_conf = min_conf
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_byte = use_byte
        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling

        self._build_tracker()

    def _build_tracker(self) -> None:
        try:
            self.tracker = OcSort(
                det_thresh=self.conf_threshold,
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold,
                delta_t=self.delta_t,
                asso_func=self.asso_func,
                inertia=self.inertia,
                use_byte=self.use_byte,
            )
        except TypeError:
            self.tracker = OcSort(
                det_thresh=self.conf_threshold,
                max_age=self.max_age,
                min_hits=self.min_hits,
                asso_threshold=self.iou_threshold,
                delta_t=self.delta_t,
                asso_func=self.asso_func,
                inertia=self.inertia,
                use_byte=self.use_byte,
            )

    def reset(self) -> None:
        self._build_tracker()

    def _extract_detection_fields(self, det: Any) -> tuple[float, float, float, float, float, int, str, int]:
        """
        Supports your Detection dataclass and dict-like detections.
        Returns:
        x1, y1, x2, y2, confidence, class_id, class_name, frame_index
        """
        if isinstance(det, Detection):
            x1, y1, x2, y2 = map(float, det.bbox)
            return (
                x1,
                y1,
                x2,
                y2,
                float(det.confidence),
                int(det.class_id),
                str(det.class_name),
                int(det.frame_index),
            )

        if isinstance(det, dict):
            bbox = det["bbox"]
            return (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                float(det.get("confidence", det.get("score", 1.0))),
                int(det.get("class_id", 0)),
                str(det.get("class_name", det.get("label", "object"))),
                int(det.get("frame_index", 0)),
            )

        # fallback for generic objects
        bbox = getattr(det, "bbox", getattr(det, "xyxy", None))
        if bbox is None:
            raise AttributeError(f"Unsupported detection object: {type(det)}")

        confidence = getattr(det, "confidence", getattr(det, "score", getattr(det, "conf", 1.0)))
        class_id = getattr(det, "class_id", getattr(det, "cls", 0))
        class_name = getattr(det, "class_name", getattr(det, "label", str(class_id)))
        frame_index = getattr(det, "frame_index", 0)

        return (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
            float(confidence),
            int(class_id),
            str(class_name),
            int(frame_index),
        )

    def update(
        self,
        detections: list[Any],
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        if detections:
            parsed = [self._extract_detection_fields(det) for det in detections]
            dets = np.array([list(p[:6]) for p in parsed], dtype=np.float32)
            meta = parsed
        else:
            dets = np.empty((0, 6), dtype=np.float32)
            meta = []

        tracks = self.tracker.update(dets, frame)

        results: list[Track] = []
        if tracks is None or len(tracks) == 0:
            return results

        # Build lightweight metadata lookup by class_id from current detections
        # This is not perfect association metadata, but good enough to keep pipeline compatible.
        class_info: dict[int, tuple[str, int]] = {}
        for _, _, _, _, _, class_id, class_name, frame_index in meta:
            class_info[class_id] = (class_name, frame_index)

        for trk in tracks:
            x1, y1, x2, y2 = map(float, trk[:4])
            track_id = int(trk[4])
            confidence = float(trk[5]) if len(trk) > 5 else 1.0
            class_id = int(trk[6]) if len(trk) > 6 else 0

            class_name, frame_index = class_info.get(class_id, (str(class_id), 0))

            results.append(
                Track(
                    track_id=track_id,
                    frame_index=frame_index,
                    bbox=[x1, y1, x2, y2],
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    state="moving",
                    trajectory=[],
                    first_frame=frame_index,
                    last_frame=frame_index,
                    age=1,
                )
            )

        return results