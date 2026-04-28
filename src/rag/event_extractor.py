from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json


@dataclass
class ExtractedEvent:
    event_type: str
    track_ids: list[int]
    start_frame: int
    end_frame: int
    timestamp: float
    metadata: dict[str, Any]


@dataclass
class TrackFact:
    track_id: int
    class_id: int
    class_name: str
    first_frame: int
    last_frame: int
    duration_frames: int
    duration_seconds: float
    num_observations: int
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    direction: str
    entry_side: str
    exit_side: str
    displacement_px: float
    total_path_px: float
    avg_speed_px_per_frame: float
    is_short_lived: bool
    is_fragmented: bool
    appearance: dict[str, Any] | None = None
    appearance_skip_reason: str | None = None


class EventExtractor:
    def __init__(
        self,
        fps: float,
        long_presence_seconds: float = 3.0,
        crowded_window_frames: int = 30,
        crowded_threshold: int = 10,
    ) -> None:
        self.fps = fps
        self.long_presence_seconds = long_presence_seconds
        self.crowded_window_frames = crowded_window_frames
        self.crowded_threshold = crowded_threshold

    def build_track_facts(self, built_tracks: list[dict[str, Any]]) -> list[TrackFact]:
        facts: list[TrackFact] = []
        for t in built_tracks:
            facts.append(
                TrackFact(
                    track_id=int(t["track_id"]),
                    class_id=int(t["class_id"]),
                    class_name=str(t["class_name"]),
                    first_frame=int(t["first_frame"]),
                    last_frame=int(t["last_frame"]),
                    duration_frames=int(t["duration_frames"]),
                    duration_seconds=float(t["duration_seconds"]),
                    num_observations=int(t["num_observations"]),
                    avg_confidence=float(t["avg_confidence"]),
                    min_confidence=float(t["min_confidence"]),
                    max_confidence=float(t["max_confidence"]),
                    direction=str(t["direction"]),
                    entry_side=str(t["entry_side"]),
                    exit_side=str(t["exit_side"]),
                    displacement_px=float(t["displacement_px"]),
                    total_path_px=float(t["total_path_px"]),
                    avg_speed_px_per_frame=float(t["avg_speed_px_per_frame"]),
                    is_short_lived=bool(t["is_short_lived"]),
                    is_fragmented=bool(t["is_fragmented"]),
                    appearance=t.get("appearance"),
                    appearance_skip_reason=t.get("appearance_skip_reason"),
                )
            )
        return facts

    def build_events(self, built_tracks: list[dict[str, Any]]) -> list[ExtractedEvent]:
        events: list[ExtractedEvent] = []

        for t in built_tracks:
            track_id = int(t["track_id"])
            class_name = str(t["class_name"])
            first_frame = int(t["first_frame"])
            last_frame = int(t["last_frame"])
            duration_seconds = float(t["duration_seconds"])
            direction = str(t["direction"])
            entry_side = str(t["entry_side"])
            exit_side = str(t["exit_side"])

            appearance_metadata = None
            if t.get("appearance") is not None:
                appearance_metadata = {
                    "appearance": t.get("appearance"),
                    "appearance_skip_reason": t.get("appearance_skip_reason"),
                }

            enter_metadata = {
                "class_name": class_name,
                "entry_side": entry_side,
            }
            exit_metadata = {
                "class_name": class_name,
                "exit_side": exit_side,
            }
            if appearance_metadata is not None:
                enter_metadata.update(appearance_metadata)
                exit_metadata.update(appearance_metadata)

            events.append(
                ExtractedEvent(
                    event_type="enter",
                    track_ids=[track_id],
                    start_frame=first_frame,
                    end_frame=first_frame,
                    timestamp=first_frame / self.fps,
                    metadata=enter_metadata,
                )
            )

            events.append(
                ExtractedEvent(
                    event_type="exit",
                    track_ids=[track_id],
                    start_frame=last_frame,
                    end_frame=last_frame,
                    timestamp=last_frame / self.fps,
                    metadata=exit_metadata,
                )
            )

            direction_metadata = {
                "class_name": class_name,
                "direction": direction,
            }
            long_presence_metadata = {
                "class_name": class_name,
                "duration_seconds": duration_seconds,
            }
            fragmented_metadata = {
                "class_name": class_name,
            }
            if appearance_metadata is not None:
                direction_metadata.update(appearance_metadata)
                long_presence_metadata.update(appearance_metadata)
                fragmented_metadata.update(appearance_metadata)

            events.append(
                ExtractedEvent(
                    event_type="direction_motion",
                    track_ids=[track_id],
                    start_frame=first_frame,
                    end_frame=last_frame,
                    timestamp=first_frame / self.fps,
                    metadata=direction_metadata,
                )
            )

            if duration_seconds >= self.long_presence_seconds:
                events.append(
                    ExtractedEvent(
                        event_type="long_presence",
                        track_ids=[track_id],
                        start_frame=first_frame,
                        end_frame=last_frame,
                        timestamp=first_frame / self.fps,
                        metadata=long_presence_metadata,
                    )
                )

            if bool(t.get("is_fragmented", False)):
                events.append(
                    ExtractedEvent(
                        event_type="fragmented_track",
                        track_ids=[track_id],
                        start_frame=first_frame,
                        end_frame=last_frame,
                        timestamp=first_frame / self.fps,
                        metadata=fragmented_metadata,
                    )
                )

        events.extend(self._build_crowded_window_events(built_tracks))
        return events

    def _build_crowded_window_events(self, built_tracks: list[dict[str, Any]]) -> list[ExtractedEvent]:
        if not built_tracks:
            return []

        max_frame = max(int(t["last_frame"]) for t in built_tracks)
        window = self.crowded_window_frames
        events: list[ExtractedEvent] = []

        for start in range(0, max_frame + 1, window):
            end = min(start + window - 1, max_frame)

            active_track_ids: list[int] = []
            for t in built_tracks:
                first_frame = int(t["first_frame"])
                last_frame = int(t["last_frame"])
                overlaps = not (last_frame < start or first_frame > end)
                if overlaps:
                    active_track_ids.append(int(t["track_id"]))

            if len(active_track_ids) >= self.crowded_threshold:
                events.append(
                    ExtractedEvent(
                        event_type="crowded_window",
                        track_ids=active_track_ids,
                        start_frame=start,
                        end_frame=end,
                        timestamp=start / self.fps,
                        metadata={
                            "num_tracks": len(active_track_ids),
                            "window_frames": window,
                        },
                    )
                )

        return events

    def save_events(self, events: list[ExtractedEvent], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in events], f, indent=2)

    def save_track_facts(self, facts: list[TrackFact], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(x) for x in facts], f, indent=2)


def load_built_tracks(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)