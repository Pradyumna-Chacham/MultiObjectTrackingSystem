from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from src.annotator import Annotator
from src.captioning.base import BaseCaptioner
from src.config import AppConfig
from src.detectors.base import BaseDetector
from src.io.mot_exporter import MOTExporter
from src.io.video_reader import VideoReader
from src.io.video_writer import VideoWriter
from src.schemas import PipelineResult, Track
from src.trackers.base import BaseTracker


class PipelineOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        detector: BaseDetector,
        tracker: BaseTracker,
        captioner: BaseCaptioner | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.captioner = captioner
        self.annotator = Annotator(
            trail_length=int(config.visualization["trail_length"]),
        )
        self.mot_exporter = MOTExporter()

    def run(self, input_video_path: str, output_video_path: str) -> PipelineResult:
        self.detector.load()
        self.tracker.reset()
        self.annotator.reset()

        reader = VideoReader(input_video_path)
        writer = VideoWriter(
            output_path=output_video_path,
            fps=reader.fps,
            width=reader.width,
            height=reader.height,
        )

        all_tracks: list[Track] = []
        seen_ids: set[int] = set()
        processed_frames = 0
        max_simultaneous_tracks = 0

        start_time = time.time()

        # ---- profiling accumulators ----
        det_total = 0.0
        trk_total = 0.0
        ann_total = 0.0
        wrt_total = 0.0
        profile_count = 0
        # --------------------------------

        try:
            for frame_index, frame in tqdm(
                reader.frames(),
                total=reader.frame_count,
                desc="Processing",
            ):
                frame_t0 = time.perf_counter()

                # ---- DETECTOR ----
                det_t0 = time.perf_counter()
                detections = self.detector.detect(frame, frame_index)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                det_t1 = time.perf_counter()

                # ---- TRACKER ----
                trk_t0 = time.perf_counter()
                tracks = self.tracker.update(detections, frame)
                trk_t1 = time.perf_counter()

                # bookkeeping
                for track in tracks:
                    all_tracks.append(track)
                    seen_ids.add(track.track_id)

                max_simultaneous_tracks = max(max_simultaneous_tracks, len(tracks))

                # ---- ANNOTATION ----
                ann_t0 = time.perf_counter()
                annotated = self.annotator.annotate(
                    frame,
                    tracks,
                    frame_index=frame_index,
                    total_unique_ids=len(seen_ids),
                )
                ann_t1 = time.perf_counter()

                # ---- WRITER ----
                wrt_t0 = time.perf_counter()
                writer.write(annotated)
                wrt_t1 = time.perf_counter()

                frame_t1 = time.perf_counter()

                # ---- accumulate ----
                det_total += det_t1 - det_t0
                trk_total += trk_t1 - trk_t0
                ann_total += ann_t1 - ann_t0
                wrt_total += wrt_t1 - wrt_t0
                profile_count += 1
                # --------------------

                # ---- print every 25 frames ----
                if profile_count % 25 == 0:
                    tqdm.write(
                    f"[PROFILE-AVG] frames={profile_count} "
                    f"det={det_total/profile_count:.3f}s "
                    f"trk={trk_total/profile_count:.3f}s "
                    f"ann={ann_total/profile_count:.3f}s "
                    f"wrt={wrt_total/profile_count:.3f}s "
                    f"total={(det_total + trk_total + ann_total + wrt_total)/profile_count:.3f}s"
                )
                # --------------------------------

                processed_frames += 1

        finally:
            reader.release()
            writer.release()

        elapsed = max(time.time() - start_time, 1e-6)
        processing_fps = processed_frames / elapsed
        class_counts = dict(Counter(track.class_name for track in all_tracks))
        track_lifetimes = {
            track.track_id: (track.last_frame - track.first_frame + 1)
            for track in all_tracks
        }
        avg_track_lifetime_frames = (
            sum(track_lifetimes.values()) / len(track_lifetimes)
            if track_lifetimes
            else 0.0
        )
        total_unique_ids = len(seen_ids)
        stats = {
            "unique_ids": total_unique_ids,
            "total_unique_ids": total_unique_ids,
            "frame_count": processed_frames or reader.frame_count,
            "input_fps": reader.fps,
            "processing_fps": processing_fps,
            "class_counts": class_counts,
            "max_simultaneous_tracks": max_simultaneous_tracks,
            "avg_track_lifetime_frames": avg_track_lifetime_frames,
        }

        result = PipelineResult(
            output_video_path=output_video_path,
            tracks=all_tracks,
            events=[],
            segments=[],
            full_summary="Pipeline run completed.",
            fps=reader.fps,
            processing_fps=processing_fps,
            stats=stats,
        )

        if bool(self.config.output.get("save_mot_format", True)):
            mot_path = self.mot_exporter.export(result.tracks, output_video_path)
            result.stats["mot_output_path"] = mot_path
        if bool(self.config.output.get("save_json", True)):
            self._save_tracks_json(result, output_video_path)
        return result

    def _save_tracks_json(self, result: PipelineResult, output_video_path: str) -> None:
        json_path = Path(output_video_path).with_suffix(".tracks.json")
        payload = {
            "output_video_path": result.output_video_path,
            "fps": result.fps,
            "processing_fps": result.processing_fps,
            "stats": result.stats,
            "tracks": [asdict(track) for track in result.tracks],
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)