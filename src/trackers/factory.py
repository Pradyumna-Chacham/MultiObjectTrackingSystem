from __future__ import annotations

from src.config import AppConfig
from src.trackers.base import BaseTracker


def get_tracker(cfg: AppConfig) -> BaseTracker:
    name = cfg.tracker["name"].lower()

    if name == "deepsort":
        from src.trackers.deepsort_tracker import DeepSortTracker

        return DeepSortTracker(
            max_age=int(cfg.tracker["max_age"]),
            n_init=int(cfg.tracker["n_init"]),
            max_iou_distance=float(cfg.tracker["max_iou_distance"]),
            embedder=str(cfg.tracker["embedder"]),
        )

    if name == "ocsort":
        from src.trackers.ocsort_tracker import OCSORTTracker

        return OCSORTTracker(
            det_thresh=float(cfg.tracker["det_thresh"]),
            max_age=int(cfg.tracker["max_age"]),
            min_hits=int(cfg.tracker["min_hits"]),
            iou_threshold=float(cfg.tracker["iou_threshold"]),
            per_class=bool(cfg.tracker.get("per_class", False)),
        )

    raise ValueError(f"Unsupported tracker: {name}")