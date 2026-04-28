from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json


@dataclass
class VideoFacts:
    fps: float
    duration_frames: int
    duration_seconds: float

    total_unique_tracks: int
    total_long_presence_tracks: int
    total_fragmented_tracks: int
    total_short_lived_tracks: int
    total_crowded_windows: int

    longest_track: dict[str, Any] | None
    shortest_track: dict[str, Any] | None
    most_crowded_window: dict[str, Any] | None

    entry_counts_by_side: dict[str, int]
    exit_counts_by_side: dict[str, int]
    direction_counts: dict[str, int]

    top_longest_tracks: list[dict[str, Any]]
    top_shortest_tracks: list[dict[str, Any]]
    top_high_confidence_tracks: list[dict[str, Any]]
    total_appearance_tracks: int
    total_confident_appearance_tracks: int
    top_appearance_tracks: list[dict[str, Any]]

    avg_track_duration_frames: float
    avg_track_duration_seconds: float
    avg_visible_tracks_per_window: float


class VideoFactBuilder:
    def __init__(self, fps: float) -> None:
        self.fps = fps

    def build(
        self,
        track_facts: list[dict[str, Any]],
        events: list[dict[str, Any]],
        chunks: list[dict[str, Any]] | None = None,
    ) -> VideoFacts:
        duration_frames = self._video_duration_frames(track_facts)
        duration_seconds = duration_frames / self.fps if self.fps > 0 else 0.0

        total_unique_tracks = len(track_facts)

        long_presence_track_ids = self._track_ids_for_event(events, "long_presence")
        fragmented_track_ids = self._track_ids_for_event(events, "fragmented_track")
        crowded_windows = [e for e in events if e.get("event_type") == "crowded_window"]

        total_long_presence_tracks = len(long_presence_track_ids)
        total_fragmented_tracks = len(fragmented_track_ids)
        total_short_lived_tracks = sum(1 for t in track_facts if bool(t.get("is_short_lived", False)))
        total_crowded_windows = len(crowded_windows)

        entry_counts_by_side = self._count_by_field(track_facts, "entry_side")
        exit_counts_by_side = self._count_by_field(track_facts, "exit_side")
        direction_counts = self._count_by_field(track_facts, "direction")

        longest_track = self._select_track(track_facts, key="duration_seconds", reverse=True)
        shortest_track = self._select_track(track_facts, key="duration_seconds", reverse=False)

        most_crowded_window = self._most_crowded_window(chunks, events)

        top_longest_tracks = self._top_tracks(track_facts, key="duration_seconds", reverse=True, n=5)
        top_shortest_tracks = self._top_tracks(track_facts, key="duration_seconds", reverse=False, n=5)
        top_high_confidence_tracks = self._top_tracks(track_facts, key="avg_confidence", reverse=True, n=5)

        appearance_tracks = [t for t in track_facts if t.get("appearance") is not None]
        confident_appearance_tracks = [
            t for t in appearance_tracks if not bool(t.get("appearance", {}).get("low_confidence", False))
        ]
        top_appearance_tracks = self._top_appearance_tracks(appearance_tracks, n=5)

        avg_track_duration_frames = (
            sum(float(t.get("duration_frames", 0)) for t in track_facts) / max(len(track_facts), 1)
        )
        avg_track_duration_seconds = (
            sum(float(t.get("duration_seconds", 0.0)) for t in track_facts) / max(len(track_facts), 1)
        )
        avg_visible_tracks_per_window = self._avg_visible_tracks_per_window(chunks)

        return VideoFacts(
            fps=self.fps,
            duration_frames=duration_frames,
            duration_seconds=duration_seconds,
            total_unique_tracks=total_unique_tracks,
            total_long_presence_tracks=total_long_presence_tracks,
            total_fragmented_tracks=total_fragmented_tracks,
            total_short_lived_tracks=total_short_lived_tracks,
            total_crowded_windows=total_crowded_windows,
            longest_track=longest_track,
            shortest_track=shortest_track,
            most_crowded_window=most_crowded_window,
            entry_counts_by_side=entry_counts_by_side,
            exit_counts_by_side=exit_counts_by_side,
            direction_counts=direction_counts,
            top_longest_tracks=top_longest_tracks,
            top_shortest_tracks=top_shortest_tracks,
            top_high_confidence_tracks=top_high_confidence_tracks,
            total_appearance_tracks=len(appearance_tracks),
            total_confident_appearance_tracks=len(confident_appearance_tracks),
            top_appearance_tracks=top_appearance_tracks,
            avg_track_duration_frames=avg_track_duration_frames,
            avg_track_duration_seconds=avg_track_duration_seconds,
            avg_visible_tracks_per_window=avg_visible_tracks_per_window,
        )

    def save(self, facts: VideoFacts, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(facts), f, indent=2)

    def _video_duration_frames(self, track_facts: list[dict[str, Any]]) -> int:
        if not track_facts:
            return 0
        return max(int(t.get("last_frame", 0)) for t in track_facts) + 1

    def _track_ids_for_event(self, events: list[dict[str, Any]], event_type: str) -> set[int]:
        out: set[int] = set()
        for e in events:
            if e.get("event_type") != event_type:
                continue
            for tid in e.get("track_ids", []):
                out.add(int(tid))
        return out

    def _count_by_field(self, track_facts: list[dict[str, Any]], field_name: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in track_facts:
            value = str(t.get(field_name, "unknown"))
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def _select_track(
        self,
        track_facts: list[dict[str, Any]],
        key: str,
        reverse: bool,
    ) -> dict[str, Any] | None:
        if not track_facts:
            return None
        sorted_tracks = sorted(track_facts, key=lambda t: float(t.get(key, 0.0)), reverse=reverse)
        t = sorted_tracks[0]
        return self._compact_track_summary(t)

    def _top_tracks(
        self,
        track_facts: list[dict[str, Any]],
        key: str,
        reverse: bool,
        n: int,
    ) -> list[dict[str, Any]]:
        sorted_tracks = sorted(track_facts, key=lambda t: float(t.get(key, 0.0)), reverse=reverse)
        return [self._compact_track_summary(t) for t in sorted_tracks[:n]]

    def _compact_track_summary(self, t: dict[str, Any]) -> dict[str, Any]:
        return {
            "track_id": int(t.get("track_id", -1)),
            "class_name": str(t.get("class_name", "unknown")),
            "first_frame": int(t.get("first_frame", 0)),
            "last_frame": int(t.get("last_frame", 0)),
            "duration_frames": int(t.get("duration_frames", 0)),
            "duration_seconds": float(t.get("duration_seconds", 0.0)),
            "direction": str(t.get("direction", "unknown")),
            "entry_side": str(t.get("entry_side", "unknown")),
            "exit_side": str(t.get("exit_side", "unknown")),
            "avg_confidence": float(t.get("avg_confidence", 0.0)),
            "is_short_lived": bool(t.get("is_short_lived", False)),
            "is_fragmented": bool(t.get("is_fragmented", False)),
        }

    def _most_crowded_window(
        self,
        chunks: list[dict[str, Any]] | None,
        events: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        best = None
        best_count = -1

        if chunks:
            for c in chunks:
                if c.get("chunk_type") != "time_window":
                    continue
                count = c.get("metadata", {}).get("visible_count")
                if count is None:
                    continue
                count = int(count)
                if count > best_count:
                    best_count = count
                    best = {
                        "start_frame": int(c.get("start_frame", 0)),
                        "end_frame": int(c.get("end_frame", 0)),
                        "start_time_seconds": float(c.get("timestamp", 0.0)),
                        "end_time_seconds": float(c.get("end_frame", 0)) / self.fps if self.fps > 0 else 0.0,
                        "visible_count": count,
                        "track_ids": [int(x) for x in c.get("track_ids", [])],
                    }

        if best is not None:
            return best

        for e in events:
            if e.get("event_type") != "crowded_window":
                continue
            count = int(e.get("metadata", {}).get("num_tracks", 0))
            if count > best_count:
                best_count = count
                best = {
                    "start_frame": int(e.get("start_frame", 0)),
                    "end_frame": int(e.get("end_frame", 0)),
                    "start_time_seconds": float(e.get("timestamp", 0.0)),
                    "end_time_seconds": float(e.get("end_frame", 0)) / self.fps if self.fps > 0 else 0.0,
                    "visible_count": count,
                    "track_ids": [int(x) for x in e.get("track_ids", [])],
                }

        return best

    def _top_appearance_tracks(self, appearance_tracks: list[dict[str, Any]], n: int = 5) -> list[dict[str, Any]]:
        sorted_tracks = sorted(
            appearance_tracks,
            key=lambda t: float(t.get("appearance", {}).get("confidence", 0.0)),
            reverse=True,
        )
        return [self._compact_appearance_summary(t) for t in sorted_tracks[:n]]

    def _compact_appearance_summary(self, t: dict[str, Any]) -> dict[str, Any]:
        appearance = t.get("appearance") or {}
        return {
            "track_id": int(t.get("track_id", -1)),
            "upper_color": appearance.get("upper_color"),
            "lower_color": appearance.get("lower_color"),
            "upper_color_base": appearance.get("upper_color_base"),
            "lower_color_base": appearance.get("lower_color_base"),
            "confidence": float(appearance.get("confidence", 0.0)),
            "low_confidence": bool(appearance.get("low_confidence", False)),
        }

    def _avg_visible_tracks_per_window(self, chunks: list[dict[str, Any]] | None) -> float:
        if not chunks:
            return 0.0
        window_chunks = [c for c in chunks if c.get("chunk_type") == "time_window"]
        if not window_chunks:
            return 0.0
        return sum(float(c.get("metadata", {}).get("visible_count", 0)) for c in window_chunks) / len(window_chunks)


def load_json_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)