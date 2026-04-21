from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from statistics import mean
from typing import Any
import json
import math


@dataclass
class BuiltTrack:
    track_id: int
    class_id: int
    class_name: str

    first_frame: int
    last_frame: int
    duration_frames: int
    duration_seconds: float

    frame_indices: list[int]
    bboxes: list[list[float]]
    centers: list[list[float]]

    avg_confidence: float
    min_confidence: float
    max_confidence: float

    start_bbox: list[float]
    end_bbox: list[float]
    start_center: list[float]
    end_center: list[float]

    displacement_px: float
    total_path_px: float
    avg_speed_px_per_frame: float

    direction: str
    entry_side: str
    exit_side: str

    num_observations: int
    is_short_lived: bool
    is_fragmented: bool


class TrackBuilder:
    """
    Consolidates frame-level track snapshots into per-track histories.

    Expected input record shape matches your current tracks.json entries:
    {
      "track_id": int,
      "frame_index": int,
      "bbox": [x1, y1, x2, y2],
      "class_id": int,
      "class_name": str,
      "confidence": float,
      ...
    }
    """

    def __init__(
        self,
        fps: float,
        frame_width: int | None = None,
        frame_height: int | None = None,
        short_lived_threshold_frames: int = 10,
        fragmented_gap_threshold_frames: int = 5,
        edge_margin_ratio: float = 0.1,
    ) -> None:
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.short_lived_threshold_frames = short_lived_threshold_frames
        self.fragmented_gap_threshold_frames = fragmented_gap_threshold_frames
        self.edge_margin_ratio = edge_margin_ratio

    def build(self, track_snapshots: list[dict[str, Any]]) -> list[BuiltTrack]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in track_snapshots:
            if "track_id" not in item:
                continue
            grouped[int(item["track_id"])].append(item)

        built_tracks: list[BuiltTrack] = []
        for track_id, items in grouped.items():
            items.sort(key=lambda x: int(x["frame_index"]))
            built_tracks.append(self._build_one(track_id, items))

        built_tracks.sort(key=lambda t: (t.first_frame, t.track_id))
        return built_tracks

    def to_json_records(self, built_tracks: list[BuiltTrack]) -> list[dict[str, Any]]:
        return [asdict(track) for track in built_tracks]

    def save_json(self, built_tracks: list[BuiltTrack], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_records(built_tracks), f, indent=2)

    def _build_one(self, track_id: int, items: list[dict[str, Any]]) -> BuiltTrack:
        frame_indices = [int(x["frame_index"]) for x in items]
        bboxes = [list(map(float, x["bbox"])) for x in items]
        centers = [self._bbox_center(b) for b in bboxes]
        confidences = [float(x.get("confidence", 1.0)) for x in items]

        class_id = int(items[0].get("class_id", 0))
        class_name = str(items[0].get("class_name", str(class_id)))

        first_frame = frame_indices[0]
        last_frame = frame_indices[-1]
        duration_frames = last_frame - first_frame + 1
        duration_seconds = duration_frames / self.fps if self.fps > 0 else 0.0

        start_bbox = bboxes[0]
        end_bbox = bboxes[-1]
        start_center = centers[0]
        end_center = centers[-1]

        displacement_px = self._distance(start_center, end_center)
        total_path_px = self._path_length(centers)
        avg_speed_px_per_frame = total_path_px / max(len(centers) - 1, 1)

        direction = self._estimate_direction(start_center, end_center)
        entry_side = self._estimate_side(start_center)
        exit_side = self._estimate_side(end_center)

        num_observations = len(items)
        is_short_lived = num_observations < self.short_lived_threshold_frames
        is_fragmented = self._is_fragmented(frame_indices)

        return BuiltTrack(
            track_id=track_id,
            class_id=class_id,
            class_name=class_name,
            first_frame=first_frame,
            last_frame=last_frame,
            duration_frames=duration_frames,
            duration_seconds=duration_seconds,
            frame_indices=frame_indices,
            bboxes=bboxes,
            centers=centers,
            avg_confidence=mean(confidences) if confidences else 0.0,
            min_confidence=min(confidences) if confidences else 0.0,
            max_confidence=max(confidences) if confidences else 0.0,
            start_bbox=start_bbox,
            end_bbox=end_bbox,
            start_center=start_center,
            end_center=end_center,
            displacement_px=displacement_px,
            total_path_px=total_path_px,
            avg_speed_px_per_frame=avg_speed_px_per_frame,
            direction=direction,
            entry_side=entry_side,
            exit_side=exit_side,
            num_observations=num_observations,
            is_short_lived=is_short_lived,
            is_fragmented=is_fragmented,
        )

    def _bbox_center(self, bbox: list[float]) -> list[float]:
        x1, y1, x2, y2 = bbox
        return [0.5 * (x1 + x2), 0.5 * (y1 + y2)]

    def _distance(self, a: list[float], b: list[float]) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _path_length(self, centers: list[list[float]]) -> float:
        if len(centers) < 2:
            return 0.0
        return sum(self._distance(centers[i - 1], centers[i]) for i in range(1, len(centers)))

    def _estimate_direction(self, start: list[float], end: list[float]) -> str:
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if abs(dx) < 5 and abs(dy) < 5:
            return "stationary"

        if abs(dx) > abs(dy):
            return "left_to_right" if dx > 0 else "right_to_left"

        return "top_to_bottom" if dy > 0 else "bottom_to_top"

    def _estimate_side(self, center: list[float]) -> str:
        x, y = center

        if self.frame_width is None or self.frame_height is None:
            return "unknown"

        x_margin = self.frame_width * self.edge_margin_ratio
        y_margin = self.frame_height * self.edge_margin_ratio

        if x <= x_margin:
            return "left"
        if x >= self.frame_width - x_margin:
            return "right"
        if y <= y_margin:
            return "top"
        if y >= self.frame_height - y_margin:
            return "bottom"
        return "interior"

    def _is_fragmented(self, frame_indices: list[int]) -> bool:
        if len(frame_indices) < 2:
            return False
        gaps = [frame_indices[i] - frame_indices[i - 1] for i in range(1, len(frame_indices))]
        return any(gap > self.fragmented_gap_threshold_frames for gap in gaps)


def load_tracks_json(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Loads your current pipeline tracks JSON.

    Expected top-level shape:
    {
      "fps": ...,
      "stats": ...,
      "tracks": [...]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tracks = payload.get("tracks", [])
    meta = {
        "fps": float(payload.get("fps", 0.0)),
        "processing_fps": float(payload.get("processing_fps", 0.0)),
        "stats": payload.get("stats", {}),
        "output_video_path": payload.get("output_video_path"),
    }
    return tracks, meta