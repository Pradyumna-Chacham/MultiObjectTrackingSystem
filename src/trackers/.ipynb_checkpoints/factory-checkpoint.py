from __future__ import annotations

from src.config import AppConfig
from src.trackers.base import BaseTracker
from src.trackers.deepsort_tracker import DeepSortTracker


def get_tracker(cfg: AppConfig) -> BaseTracker:
    name = cfg.tracker["name"].lower()

    if name == "deepsort":
        return DeepSortTracker(
            max_age=int(cfg.tracker["max_age"]),
            n_init=int(cfg.tracker["n_init"]),
            embedder=str(cfg.tracker["embedder"]),
        )

    raise ValueError(f"Unsupported tracker: {name}")