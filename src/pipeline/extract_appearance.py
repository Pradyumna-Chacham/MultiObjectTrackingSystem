from __future__ import annotations

import cv2
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from typing import Any, Dict, List, Tuple

from src.config import AppConfig


# ---------------------------------------------------------
# Utility: HSV distance
# ---------------------------------------------------------
def hsv_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------
# Utility: crop center region
# ---------------------------------------------------------
def center_crop(img: np.ndarray, ratio: float) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)

    y1 = (h - ch) // 2
    x1 = (w - cw) // 2

    return img[y1:y1 + ch, x1:x1 + cw]


# ---------------------------------------------------------
# Color mapping (HSV → label)
# ---------------------------------------------------------
def hsv_to_color_label(hsv: np.ndarray) -> str:
    h, s, v = hsv

    if v < 50:
        return "black"
    if v > 200 and s < 40:
        return "white"
    if s < 40:
        return "gray"

    if h < 10 or h > 170:
        return "red"
    if 10 <= h < 25:
        return "orange"
    if 25 <= h < 35:
        return "yellow"
    if 35 <= h < 85:
        return "green"
    if 85 <= h < 130:
        return "blue"
    if 130 <= h < 160:
        return "navy"

    return "unknown"


# ---------------------------------------------------------
# Skin filter
# ---------------------------------------------------------
def is_skin(hsv: np.ndarray, cfg: Dict[str, Any]) -> bool:
    h, s, v = hsv

    return (
        cfg["skin_hue_min"] <= h <= cfg["skin_hue_max"]
        and cfg["skin_sat_min"] <= s <= cfg["skin_sat_max"]
        and cfg["skin_val_min"] <= v <= cfg["skin_val_max"]
    )


# ---------------------------------------------------------
# Extract dominant clusters
# ---------------------------------------------------------
def get_clusters(hsv_img: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    pixels = hsv_img.reshape(-1, 3)

    if len(pixels) < k:
        return None, None

    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    return centers, labels


# ---------------------------------------------------------
# Assign upper/lower using vertical position
# ---------------------------------------------------------
def assign_upper_lower(
    centers: np.ndarray,
    labels: np.ndarray,
    hsv_img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    h = hsv_img.shape[0]
    pixels = hsv_img.reshape(-1, 3)

    y_coords = np.repeat(np.arange(h), hsv_img.shape[1])

    cluster_y = []
    for i in range(len(centers)):
        ys = y_coords[labels == i]
        cluster_y.append(np.mean(ys) if len(ys) > 0 else 0)

    # smaller y = upper
    order = np.argsort(cluster_y)

    upper = centers[order[0]]
    lower = centers[order[-1]]

    return upper, lower


# ---------------------------------------------------------
# Extract color from one crop
# ---------------------------------------------------------
def extract_from_crop(
    crop: np.ndarray,
    cfg: Dict[str, Any]
) -> Tuple[str, str, float]:

    crop = center_crop(crop, cfg["center_crop_ratio"])
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    centers, labels = get_clusters(hsv, cfg["kmeans_k"])
    if centers is None:
        return None, None, 0.0

    # Remove skin clusters
    valid = []
    for i, c in enumerate(centers):
        if not is_skin(c, cfg):
            valid.append((i, c))

    if len(valid) < 2:
        return None, None, 0.0

    idxs = [v[0] for v in valid]
    centers = np.array([v[1] for v in valid])

    # Monochrome check
    if len(centers) >= 2:
        dist = hsv_distance(centers[0], centers[1])
        if dist < cfg["cluster_distance_threshold"]:
            color = hsv_to_color_label(centers[0])
            return color, color, 0.5

    upper, lower = assign_upper_lower(centers, labels, hsv)

    upper_label = hsv_to_color_label(upper)
    lower_label = hsv_to_color_label(lower)

    return upper_label, lower_label, 1.0


# ---------------------------------------------------------
# Main extraction
# ---------------------------------------------------------
def extract_appearance(
    tracks: List[Dict[str, Any]],
    video_path: str,
    cfg: AppConfig
) -> List[Dict[str, Any]]:

    app_cfg = cfg.appearance
    if not app_cfg.get("enabled", False):
        return tracks

    cap = cv2.VideoCapture(video_path)

    for track in tracks:

        observations = track.get("observations", [])
        if len(observations) == 0:
            track["appearance"] = None
            track["appearance_skip_reason"] = "no_observations"
            continue

        # sort by score
        observations = sorted(
            observations,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[: app_cfg["sample_count"]]

        upper_votes = []
        lower_votes = []
        frames_used = []

        for obs in observations:
            frame_idx = int(obs["frame"])
            bbox = obs["bbox"]  # [x1,y1,x2,y2]

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]

            if crop.shape[0] < app_cfg["min_crop_height"] or crop.shape[1] < app_cfg["min_crop_width"]:
                continue

            up, low, conf = extract_from_crop(crop, app_cfg)
            if up is None:
                continue

            upper_votes.append(up)
            lower_votes.append(low)
            frames_used.append(frame_idx)

        if len(upper_votes) == 0:
            track["appearance"] = None
            track["appearance_skip_reason"] = "insufficient_valid_samples"
            continue

        # majority vote
        def majority(votes):
            return max(set(votes), key=votes.count)

        upper = majority(upper_votes)
        lower = majority(lower_votes)

        confidence = upper_votes.count(upper) / len(upper_votes)

        base_map = app_cfg["canonical_color_map"]

        track["appearance"] = {
            "upper_color": upper,
            "upper_color_base": base_map.get(upper, upper),
            "lower_color": lower,
            "lower_color_base": base_map.get(lower, lower),
            "confidence": confidence,
            "low_confidence": confidence < app_cfg["low_confidence_threshold"],
            "evidence_frames": frames_used,
        }

        track["appearance_skip_reason"] = None

    cap.release()
    return tracks


# ---------------------------------------------------------
# CLI usage
# ---------------------------------------------------------
def run(
    tracks_path: str,
    video_path: str,
    output_path: str,
    cfg: AppConfig,
):
    with open(tracks_path, "r") as f:
        tracks = json.load(f)

    tracks = extract_appearance(tracks, video_path, cfg)

    with open(output_path, "w") as f:
        json.dump(tracks, f, indent=2)