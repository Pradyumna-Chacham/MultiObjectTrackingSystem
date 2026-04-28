from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class AppConfig:
    raw: dict[str, Any]

    @property
    def system(self) -> dict[str, Any]:
        return self.raw.get("system", {})

    @property
    def device(self) -> str:
        requested = self.raw.get("device", "auto")
        if requested != "auto":
            return requested

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def detector(self) -> dict[str, Any]:
        return self.raw["detector"]

    @property
    def tracker(self) -> dict[str, Any]:
        return self.raw["tracker"]

    @property
    def events(self) -> dict[str, Any]:
        return self.raw.get("events", {})

    @property
    def captioning(self) -> dict[str, Any]:
        return self.raw.get("captioning", {})

    @property
    def visualization(self) -> dict[str, Any]:
        return self.raw.get("visualization", {})

    @property
    def output(self) -> dict[str, Any]:
        return self.raw.get("output", {})

    @property
    def demo(self) -> dict[str, Any]:
        return self.raw.get("demo", {})

    @property
    def appearance(self) -> dict[str, Any]:
        """
        Optional appearance-extraction config.

        This block is intentionally optional so the core tracking pipeline
        still works even when appearance extraction is not configured or not run.
        """
        defaults: dict[str, Any] = {
            "enabled": False,
            "sample_count": 5,
            "min_crop_width": 30,
            "min_crop_height": 60,
            "center_crop_ratio": 0.70,
            "kmeans_k": 3,
            "cluster_distance_threshold": 30.0,
            "low_confidence_threshold": 0.50,
            "skin_hue_min": 0,
            "skin_hue_max": 25,
            "skin_sat_min": 20,
            "skin_sat_max": 255,
            "skin_val_min": 40,
            "skin_val_max": 255,
            "canonical_color_map": {
                "navy": "blue",
                "sky_blue": "blue",
                "pink": "red",
                "orange": "orange",
                "brown": "brown",
                "black": "black",
                "white": "white",
                "gray": "gray",
                "red": "red",
                "blue": "blue",
                "green": "green",
                "yellow": "yellow",
            },
        }

        user_cfg = self.raw.get("appearance", {})
        merged = dict(defaults)
        merged.update(user_cfg)

        canonical_map = dict(defaults["canonical_color_map"])
        canonical_map.update(user_cfg.get("canonical_color_map", {}))
        merged["canonical_color_map"] = canonical_map

        return merged

    @property
    def seed(self) -> int:
        return int(self.system.get("seed", 42))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = AppConfig(raw=raw)
    set_seed(cfg.seed)
    return cfg