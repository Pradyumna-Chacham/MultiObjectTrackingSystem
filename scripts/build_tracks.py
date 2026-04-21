from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.track_builder import TrackBuilder, load_tracks_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consolidated tracks from frame-level tracking JSON.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to pipeline tracks JSON, e.g. demo/sample_outputs/mot17-04-ocsort.tracks.json",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Path to save built tracks JSON. Defaults next to input with .built_tracks.json suffix.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS. If omitted, uses fps from input JSON.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=None,
        help="Frame width for entry/exit side estimation.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=None,
        help="Frame height for entry/exit side estimation.",
    )
    parser.add_argument(
        "--short-lived-threshold",
        type=int,
        default=10,
        help="Tracks with fewer observations than this are marked short-lived.",
    )
    parser.add_argument(
        "--fragmented-gap-threshold",
        type=int,
        default=5,
        help="If a track has frame gaps larger than this, it is marked fragmented.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    if input_path.name.endswith(".tracks.json"):
        return input_path.with_name(input_path.name.replace(".tracks.json", ".built_tracks.json"))
    if input_path.suffix == ".json":
        return input_path.with_name(f"{input_path.stem}.built_tracks.json")
    return input_path.with_name(f"{input_path.name}.built_tracks.json")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_tracks, meta = load_tracks_json(str(input_path))
    fps = args.fps if args.fps is not None else float(meta.get("fps", 0.0))

    if fps <= 0:
        raise ValueError("FPS must be > 0. Pass --fps explicitly or ensure the input JSON contains fps.")

    builder = TrackBuilder(
        fps=fps,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        short_lived_threshold_frames=args.short_lived_threshold,
        fragmented_gap_threshold_frames=args.fragmented_gap_threshold,
    )

    built_tracks = builder.build(raw_tracks)
    builder.save_json(built_tracks, str(output_path))

    print("Built track histories complete.")
    print(f"Input snapshots      : {len(raw_tracks)}")
    print(f"Consolidated tracks  : {len(built_tracks)}")
    print(f"FPS                  : {fps}")
    print(f"Output               : {output_path}")

    if built_tracks:
        avg_obs = sum(t.num_observations for t in built_tracks) / len(built_tracks)
        avg_duration = sum(t.duration_frames for t in built_tracks) / len(built_tracks)
        num_fragmented = sum(1 for t in built_tracks if t.is_fragmented)
        num_short = sum(1 for t in built_tracks if t.is_short_lived)

        print(f"Avg observations/track : {avg_obs:.2f}")
        print(f"Avg duration (frames)  : {avg_duration:.2f}")
        print(f"Short-lived tracks     : {num_short}")
        print(f"Fragmented tracks      : {num_fragmented}")


if __name__ == "__main__":
    main()