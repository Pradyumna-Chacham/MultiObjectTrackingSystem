from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any
import json


@dataclass
class RetrievalChunk:
    chunk_id: str
    chunk_type: str
    text: str
    start_frame: int
    end_frame: int
    timestamp: float
    track_ids: list[int]
    metadata: dict[str, Any]


class ChunkBuilder:
    def __init__(
        self,
        fps: float,
        window_frames: int = 30,
    ) -> None:
        self.fps = fps
        self.window_frames = window_frames

    def build_all_chunks(
        self,
        track_facts: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> list[RetrievalChunk]:
        chunks: list[RetrievalChunk] = []
        chunks.extend(self.build_track_chunks(track_facts))
        chunks.extend(self.build_appearance_chunks(track_facts))
        chunks.extend(self.build_event_chunks(events))
        chunks.extend(self.build_time_window_chunks(track_facts, events))
        chunks.sort(key=lambda x: (x.start_frame, x.chunk_type, x.chunk_id))
        return chunks

    def build_track_chunks(self, track_facts: list[dict[str, Any]]) -> list[RetrievalChunk]:
        chunks: list[RetrievalChunk] = []

        for t in track_facts:
            track_id = int(t["track_id"])
            class_name = str(t["class_name"])
            first_frame = int(t["first_frame"])
            last_frame = int(t["last_frame"])
            duration_frames = int(t["duration_frames"])
            duration_seconds = float(t["duration_seconds"])
            direction = str(t["direction"])
            entry_side = str(t["entry_side"])
            exit_side = str(t["exit_side"])
            avg_confidence = float(t["avg_confidence"])
            is_short_lived = bool(t["is_short_lived"])
            is_fragmented = bool(t["is_fragmented"])
            displacement_px = float(t["displacement_px"])
            total_path_px = float(t["total_path_px"])

            text = (
                f"Track {track_id} is a {class_name} observed from frame {first_frame} to {last_frame} "
                f"for {duration_frames} frames ({duration_seconds:.2f} seconds). "
                f"It moves {direction}, enters from {entry_side}, exits at {exit_side}, "
                f"has average confidence {avg_confidence:.3f}, displacement {displacement_px:.1f} pixels, "
                f"and total path length {total_path_px:.1f} pixels. "
                f"Short lived: {is_short_lived}. Fragmented: {is_fragmented}."
            )

            chunks.append(
                RetrievalChunk(
                    chunk_id=f"track_{track_id}",
                    chunk_type="track",
                    text=text,
                    start_frame=first_frame,
                    end_frame=last_frame,
                    timestamp=first_frame / self.fps,
                    track_ids=[track_id],
                    metadata={
                        "track_id": track_id,
                        "class_name": class_name,
                        "duration_frames": duration_frames,
                        "duration_seconds": duration_seconds,
                        "direction": direction,
                        "entry_side": entry_side,
                        "exit_side": exit_side,
                        "avg_confidence": avg_confidence,
                        "is_short_lived": is_short_lived,
                        "is_fragmented": is_fragmented,
                        "displacement_px": displacement_px,
                        "total_path_px": total_path_px,
                    },
                )
            )

        return chunks

    def build_appearance_chunks(self, track_facts: list[dict[str, Any]]) -> list[RetrievalChunk]:
        chunks: list[RetrievalChunk] = []

        for t in track_facts:
            appearance = t.get("appearance")
            if appearance is None:
                continue

            if bool(appearance.get("low_confidence", False)):
                continue

            track_id = int(t["track_id"])
            class_name = str(t.get("class_name", "person"))
            first_frame = int(t.get("first_frame", 0))
            last_frame = int(t.get("last_frame", first_frame))
            confidence = float(appearance.get("confidence", 0.0))

            upper_color = appearance.get("upper_color")
            upper_color_base = appearance.get("upper_color_base", upper_color)
            lower_color = appearance.get("lower_color")
            lower_color_base = appearance.get("lower_color_base", lower_color)
            evidence_frames = [int(x) for x in appearance.get("evidence_frames", [])]

            if not upper_color and not lower_color:
                continue

            if upper_color and lower_color:
                text = (
                    f"Person with track ID {track_id} wore {upper_color} on top and {lower_color} "
                    f"on the bottom, observed consistently across multiple high-confidence frames."
                )
            elif upper_color:
                text = (
                    f"Person with track ID {track_id} wore {upper_color} on top, observed consistently "
                    f"across multiple high-confidence frames."
                )
            else:
                text = (
                    f"Person with track ID {track_id} wore {lower_color} on the bottom, observed consistently "
                    f"across multiple high-confidence frames."
                )

            chunks.append(
                RetrievalChunk(
                    chunk_id=f"appearance_{track_id}",
                    chunk_type="appearance",
                    text=text,
                    start_frame=first_frame,
                    end_frame=last_frame,
                    timestamp=first_frame / self.fps,
                    track_ids=[track_id],
                    metadata={
                        "track_id": track_id,
                        "class_name": class_name,
                        "upper_color": upper_color,
                        "upper_color_base": upper_color_base,
                        "lower_color": lower_color,
                        "lower_color_base": lower_color_base,
                        "confidence": confidence,
                        "low_confidence": False,
                        "evidence_frames": evidence_frames,
                    },
                )
            )

        return chunks

    def build_event_chunks(self, events: list[dict[str, Any]]) -> list[RetrievalChunk]:
        chunks: list[RetrievalChunk] = []

        for idx, e in enumerate(events):
            event_type = str(e["event_type"])
            track_ids = [int(x) for x in e.get("track_ids", [])]
            start_frame = int(e["start_frame"])
            end_frame = int(e["end_frame"])
            timestamp = float(e["timestamp"])
            metadata = dict(e.get("metadata", {}))

            text = (
                f"Event {event_type} occurs from frame {start_frame} to {end_frame} "
                f"at time {timestamp:.2f} seconds involving track IDs {track_ids}. "
                f"Metadata: {metadata}."
            )

            chunks.append(
                RetrievalChunk(
                    chunk_id=f"event_{event_type}_{idx}",
                    chunk_type="event",
                    text=text,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    timestamp=timestamp,
                    track_ids=track_ids,
                    metadata={
                        "event_type": event_type,
                        **metadata,
                    },
                )
            )

        return chunks

    def build_time_window_chunks(
        self,
        track_facts: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> list[RetrievalChunk]:
        chunks: list[RetrievalChunk] = []

        if not track_facts:
            return chunks

        track_map = {int(t["track_id"]): t for t in track_facts}
        max_frame = max(int(t["last_frame"]) for t in track_facts)
        window = self.window_frames

        for start in range(0, max_frame + 1, window):
            end = min(start + window - 1, max_frame)

            active_tracks = []
            for t in track_facts:
                first_frame = int(t["first_frame"])
                last_frame = int(t["last_frame"])
                overlaps = not (last_frame < start or first_frame > end)
                if overlaps:
                    active_tracks.append(t)

            active_track_ids = [int(t["track_id"]) for t in active_tracks]
            visible_count = len(active_track_ids)

            entering_tracks = []
            exiting_tracks = []
            long_presence_tracks = []
            crowded = False

            for e in events:
                e_type = str(e["event_type"])
                e_start = int(e["start_frame"])
                e_end = int(e["end_frame"])
                overlaps = not (e_end < start or e_start > end)
                if not overlaps:
                    continue

                e_track_ids = [int(x) for x in e.get("track_ids", [])]

                if e_type == "enter":
                    entering_tracks.extend(e_track_ids)
                elif e_type == "exit":
                    exiting_tracks.extend(e_track_ids)
                elif e_type == "long_presence":
                    long_presence_tracks.extend(e_track_ids)
                elif e_type == "crowded_window":
                    crowded = True

            entering_tracks = sorted(set(entering_tracks))
            exiting_tracks = sorted(set(exiting_tracks))
            long_presence_tracks = sorted(set(long_presence_tracks))

            entry_side_counts = Counter()
            exit_side_counts = Counter()

            for tid in entering_tracks:
                t = track_map.get(tid)
                if t is not None:
                    entry_side_counts[str(t.get("entry_side", "unknown"))] += 1

            for tid in exiting_tracks:
                t = track_map.get(tid)
                if t is not None:
                    exit_side_counts[str(t.get("exit_side", "unknown"))] += 1

            entry_side_counts_dict = dict(sorted(entry_side_counts.items()))
            exit_side_counts_dict = dict(sorted(exit_side_counts.items()))

            entry_side_summary = (
                ", ".join(f"{side}: {count}" for side, count in entry_side_counts_dict.items())
                if entry_side_counts_dict
                else "none"
            )
            exit_side_summary = (
                ", ".join(f"{side}: {count}" for side, count in exit_side_counts_dict.items())
                if exit_side_counts_dict
                else "none"
            )

            text = (
                f"Time window from frame {start} to {end} "
                f"({start / self.fps:.2f} to {end / self.fps:.2f} seconds) has {visible_count} active tracks. "
                f"Active track IDs: {active_track_ids}. "
                f"Entering tracks: {entering_tracks}. "
                f"Entry counts by side in this window: {entry_side_summary}. "
                f"Exiting tracks: {exiting_tracks}. "
                f"Exit counts by side in this window: {exit_side_summary}. "
                f"Long presence tracks overlapping this window: {long_presence_tracks}. "
                f"Crowded: {crowded}."
            )

            chunks.append(
                RetrievalChunk(
                    chunk_id=f"window_{start}_{end}",
                    chunk_type="time_window",
                    text=text,
                    start_frame=start,
                    end_frame=end,
                    timestamp=start / self.fps,
                    track_ids=active_track_ids,
                    metadata={
                        "visible_count": visible_count,
                        "entering_tracks": entering_tracks,
                        "exiting_tracks": exiting_tracks,
                        "long_presence_tracks": long_presence_tracks,
                        "crowded": crowded,
                        "window_frames": window,
                        "entry_side_counts": entry_side_counts_dict,
                        "exit_side_counts": exit_side_counts_dict,
                    },
                )
            )

        return chunks

    def save_chunks(self, chunks: list[RetrievalChunk], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, indent=2)


def load_json_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)