from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.pipeline.extract_appearance import run


def parse_args():
    parser = argparse.ArgumentParser(description="Extract appearance (color) for tracked persons")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config"
    )

    parser.add_argument(
        "--tracks",
        type=str,
        required=True,
        help="Path to input tracks JSON (from track_builder)"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save updated tracks with appearance"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)

    tracks_path = Path(args.tracks)
    video_path = Path(args.video)
    output_path = Path(args.output)

    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks file not found: {tracks_path}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    run(
        tracks_path=str(tracks_path),
        video_path=str(video_path),
        output_path=str(output_path),
        cfg=cfg,
    )

    print("\n✅ Appearance extraction complete.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()