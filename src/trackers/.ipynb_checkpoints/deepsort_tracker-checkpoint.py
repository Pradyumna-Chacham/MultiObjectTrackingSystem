from __future__ import annotations

import time
from collections import defaultdict, deque
from tqdm import tqdm
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.schemas import Detection, Track
from src.trackers.base import BaseTracker
from src.utils.bbox_utils import bbox_center, xyxy_to_tlwh


class DeepSortTracker(BaseTracker):
    def __init__(self, max_age: int = 30, n_init: int = 3, embedder: str = "mobilenet") -> None:
        self.max_age = max_age
        self.n_init = n_init
        self.embedder = embedder
        self.current_frame_index = -1

        # Keep history bounded
        self.history: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=30))
        self.track_first_frame: dict[int, int] = {}
        self.track_metadata: dict[int, dict[str, float | int | str]] = {}

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            embedder=embedder,
            embedder_gpu=True,
        )

        # ---- profiling accumulators ----
        self.prof_frames = 0
        self.prof_prepare_total = 0.0
        self.prof_update_tracks_total = 0.0
        self.prof_post_total = 0.0
        self.prof_traj_total = 0.0
        self.prof_cleanup_total = 0.0
        # -------------------------------

    def reset(self) -> None:
        self.current_frame_index = -1
        self.history.clear()
        self.track_first_frame.clear()
        self.track_metadata.clear()
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            embedder=self.embedder,
            embedder_gpu=True,
        )

        # reset profiling
        self.prof_frames = 0
        self.prof_prepare_total = 0.0
        self.prof_update_tracks_total = 0.0
        self.prof_post_total = 0.0
        self.prof_traj_total = 0.0
        self.prof_cleanup_total = 0.0

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        frame_t0 = time.perf_counter()

        if detections:
            self.current_frame_index = detections[0].frame_index
        else:
            self.current_frame_index += 1

        # ---- prepare detections for DeepSORT ----
        prep_t0 = time.perf_counter()
        ds_detections = []
        detection_metadata = []

        for det in detections:
            ds_detections.append((xyxy_to_tlwh(det.bbox), det.confidence, det.class_name))
            detection_metadata.append(
                {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "frame_index": det.frame_index,
                }
            )
        prep_t1 = time.perf_counter()

        # ---- DeepSORT internals (embedder likely inside here) ----
        upd_t0 = time.perf_counter()
        tracks = self.tracker.update_tracks(ds_detections, frame=frame, others=detection_metadata)
        upd_t1 = time.perf_counter()

        # ---- wrapper post-processing ----
        post_t0 = time.perf_counter()
        output: list[Track] = []
        active_ids: set[int] = set()
        current_frame = self.current_frame_index

        traj_total_this_frame = 0.0

        for trk in tracks:
            if not trk.is_confirmed():
                continue

            ltrb = trk.to_ltrb()
            bbox = [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]
            track_id = int(trk.track_id)
            active_ids.add(track_id)

            center = bbox_center(bbox)
            self.history[track_id].append(center)

            det_info = trk.get_det_supplementary()
            if det_info is not None:
                self.track_metadata[track_id] = {
                    "class_id": int(det_info["class_id"]),
                    "class_name": str(det_info["class_name"]),
                    "confidence": float(det_info["confidence"]),
                }

            metadata = self.track_metadata.get(track_id)
            if metadata is None:
                metadata = {
                    "class_id": -1,
                    "class_name": trk.get_det_class() or "object",
                    "confidence": float(trk.get_det_conf() or 0.0),
                }

            self.track_first_frame.setdefault(track_id, current_frame)

            # ---- specifically profile trajectory copy/build ----
            traj_t0 = time.perf_counter()
            trajectory_copy = list(self.history[track_id])
            traj_t1 = time.perf_counter()
            traj_total_this_frame += traj_t1 - traj_t0

            output.append(
                Track(
                    track_id=track_id,
                    frame_index=current_frame,
                    bbox=bbox,
                    class_id=int(metadata["class_id"]),
                    class_name=str(metadata["class_name"]),
                    confidence=float(metadata["confidence"]),
                    state="moving",
                    trajectory=trajectory_copy,
                    first_frame=self.track_first_frame[track_id],
                    last_frame=current_frame,
                    age=len(self.history[track_id]),
                )
            )

        post_t1 = time.perf_counter()

        # ---- cleanup stale ids ----
        cleanup_t0 = time.perf_counter()
        stale_ids = [tid for tid in self.history if tid not in active_ids]
        for tid in stale_ids:
            self.history.pop(tid, None)
            self.track_first_frame.pop(tid, None)
            self.track_metadata.pop(tid, None)
        cleanup_t1 = time.perf_counter()

        frame_t1 = time.perf_counter()

        # ---- accumulate profiling ----
        self.prof_frames += 1
        self.prof_prepare_total += prep_t1 - prep_t0
        self.prof_update_tracks_total += upd_t1 - upd_t0
        self.prof_post_total += post_t1 - post_t0
        self.prof_traj_total += traj_total_this_frame
        self.prof_cleanup_total += cleanup_t1 - cleanup_t0

        if self.prof_frames % 25 == 0:
            tqdm.write(
                f"[TRACKER-PROFILE] frames={self.prof_frames} "
                f"prep={self.prof_prepare_total/self.prof_frames:.3f}s "
                f"update_tracks={self.prof_update_tracks_total/self.prof_frames:.3f}s "
                f"post={self.prof_post_total/self.prof_frames:.3f}s "
                f"traj={self.prof_traj_total/self.prof_frames:.3f}s "
                f"cleanup={self.prof_cleanup_total/self.prof_frames:.3f}s "
                f"total={((frame_t1-frame_t0)):.3f}s"
            )

        return output