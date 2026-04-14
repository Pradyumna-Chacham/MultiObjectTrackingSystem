from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.schemas import Detection, Track


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError