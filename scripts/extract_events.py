from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.event_extractor import EventExtractor, load_built_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract events and track facts from built tracks.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to built tracks JSON, e.g. demo/sample_outputs/mot17-04-ocsort.built_tracks.json",
    )
    parser.add_argument(
        "--events-output",
        required=False,
        help="Path to save extracted events JSON.",
    )
    parser.add_argument(
        "--facts-output",
        required=False,
        help="Path to save track facts JSON.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Video FPS.",
    )
    parser.add_argument(
        "--long-presence-seconds",
        type=float,
        default=3.0,
        help="Minimum duration for a long_presence event.",
    )
    parser.add_argument(
        "--crowded-window-frames",
        type=int,
        default=30,
        help="Window size in frames for crowded_window events.",
    )
    parser.add_argument(
        "--crowded-threshold",
        type=int,
        default=10,
        help="Minimum active tracks in a window to mark it crowded.",
    )
    return parser.parse_args()


def default_events_path(input_path: Path) -> Path:
    if input_path.name.endswith(".built_tracks.json"):
        return input_path.with_name(input_path.name.replace(".built_tracks.json", ".events.json"))
    return input_path.with_name(f"{input_path.stem}.events.json")


def default_facts_path(input_path: Path) -> Path:
    if input_path.name.endswith(".built_tracks.json"):
        return input_path.with_name(input_path.name.replace(".built_tracks.json", ".track_facts.json"))
    return input_path.with_name(f"{input_path.stem}.track_facts.json")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    events_output = Path(args.events_output) if args.events_output else default_events_path(input_path)
    facts_output = Path(args.facts_output) if args.facts_output else default_facts_path(input_path)

    events_output.parent.mkdir(parents=True, exist_ok=True)
    facts_output.parent.mkdir(parents=True, exist_ok=True)

    built_tracks = load_built_tracks(str(input_path))

    extractor = EventExtractor(
        fps=args.fps,
        long_presence_seconds=args.long_presence_seconds,
        crowded_window_frames=args.crowded_window_frames,
        crowded_threshold=args.crowded_threshold,
    )

    track_facts = extractor.build_track_facts(built_tracks)
    events = extractor.build_events(built_tracks)

    extractor.save_track_facts(track_facts, str(facts_output))
    extractor.save_events(events, str(events_output))

    print("Event extraction complete.")
    print(f"Built tracks : {len(built_tracks)}")
    print(f"Track facts  : {len(track_facts)}")
    print(f"Events       : {len(events)}")
    print(f"Facts output : {facts_output}")
    print(f"Events output: {events_output}")

    counts: dict[str, int] = {}
    for e in events:
        counts[e.event_type] = counts.get(e.event_type, 0) + 1

    if counts:
        print("Event counts:")
        for k in sorted(counts):
            print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()