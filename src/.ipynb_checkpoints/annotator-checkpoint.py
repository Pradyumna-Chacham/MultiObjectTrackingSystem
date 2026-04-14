from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from src.schemas import Track


class Annotator:
    def __init__(self, trail_length: int = 30, draw_trails: bool = True) -> None:
        self.trail_length = trail_length
        self.draw_trails = draw_trails
        self.history: dict[int, deque[tuple[int, int]]] = {}
        self.color_cache: dict[int, tuple[int, int, int]] = {}

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        frame_index: int | None = None,
        total_unique_ids: int | None = None,
    ) -> np.ndarray:
        output = frame.copy()
        active_ids = {track.track_id for track in tracks}

        missing_ids = [track_id for track_id in self.history if track_id not in active_ids]
        for track_id in missing_ids:
            del self.history[track_id]

        for track in tracks:
            color = self._color_for_track(track.track_id)
            x1, y1, x2, y2 = (int(value) for value in track.bbox)
            bbox = (x1, y1, x2, y2)
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

            if track.track_id not in self.history:
                self.history[track.track_id] = deque(maxlen=self.trail_length)
            self.history[track.track_id].append(centroid)

            if self.draw_trails:
                self._draw_trail(output, self.history[track.track_id], color)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.circle(output, centroid, 4, color, -1)
            self._draw_label(output, track, bbox, color)

        overlay_lines: list[str] = []
        if frame_index is not None:
            overlay_lines.append(f"Frame: {frame_index}")
        overlay_lines.append(f"Tracks: {len(tracks)}")
        if total_unique_ids is not None:
            overlay_lines.append(f"Total Unique IDs: {total_unique_ids}")

        for idx, text in enumerate(overlay_lines):
            cv2.putText(
                output,
                text,
                (20, 30 + idx * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output

    def reset(self) -> None:
        self.history.clear()

    def _draw_trail(
        self,
        frame: np.ndarray,
        points: deque[tuple[int, int]],
        color: tuple[int, int, int],
    ) -> None:
        if len(points) < 2:
            return

        pts = np.array(points, dtype=np.int32)
        for i in range(1, len(pts)):
            thickness = max(1, int(2 * i / len(pts)))
            cv2.line(
                frame,
                tuple(pts[i - 1]),
                tuple(pts[i]),
                color,
                thickness,
                cv2.LINE_AA,
            )

    def _draw_label(
        self,
        frame: np.ndarray,
        track: Track,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        x1, y1, _, _ = bbox
        label = f"ID {track.track_id} | {track.class_name} | {track.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        label_top = max(0, y1 - text_height - baseline - 8)
        label_bottom = label_top + text_height + baseline + 8
        label_right = x1 + text_width + 10

        cv2.rectangle(frame, (x1, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 5, label_bottom - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _color_for_track(self, track_id: int) -> tuple[int, int, int]:
        if track_id in self.color_cache:
            return self.color_cache[track_id]

        hue = (track_id * 37) % 180
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        self.color_cache[track_id] = color
        return color