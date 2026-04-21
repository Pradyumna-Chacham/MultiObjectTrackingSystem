from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.video_fact_builder import VideoFactBuilder, load_json_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build precomputed video facts from track facts, events, and chunks.")
    parser.add_argument(
        "--track-facts",
        required=True,
        help="Path to track_facts.json",
    )
    parser.add_argument(
        "--events",
        required=True,
        help="Path to events.json",
    )
    parser.add_argument(
        "--chunks",
        required=False,
        help="Optional path to chunks.json for richer window/global statistics.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Video FPS.",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Path to save video facts JSON.",
    )
    return parser.parse_args()


def default_output_path(track_facts_path: Path) -> Path:
    name = track_facts_path.name
    if name.endswith(".track_facts.json"):
        return track_facts_path.with_name(name.replace(".track_facts.json", ".video_facts.json"))
    return track_facts_path.with_name(f"{track_facts_path.stem}.video_facts.json")


def main() -> None:
    args = parse_args()

    track_facts_path = Path(args.track_facts)
    events_path = Path(args.events)
    chunks_path = Path(args.chunks) if args.chunks else None

    if not track_facts_path.exists():
        raise FileNotFoundError(f"Track facts file not found: {track_facts_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")
    if chunks_path is not None and not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    output_path = Path(args.output) if args.output else default_output_path(track_facts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    track_facts = load_json_records(str(track_facts_path))
    events = load_json_records(str(events_path))
    chunks = load_json_records(str(chunks_path)) if chunks_path is not None else None

    builder = VideoFactBuilder(fps=args.fps)
    facts = builder.build(track_facts=track_facts, events=events, chunks=chunks)
    builder.save(facts, str(output_path))

    print("Video fact building complete.")
    print(f"Track facts : {len(track_facts)}")
    print(f"Events      : {len(events)}")
    print(f"Chunks      : {len(chunks) if chunks is not None else 0}")
    print(f"Output      : {output_path}")
    print()
    print("Key facts:")
    print(f"  total_unique_tracks       : {facts.total_unique_tracks}")
    print(f"  total_long_presence       : {facts.total_long_presence_tracks}")
    print(f"  total_fragmented_tracks   : {facts.total_fragmented_tracks}")
    print(f"  total_short_lived_tracks  : {facts.total_short_lived_tracks}")
    print(f"  total_crowded_windows     : {facts.total_crowded_windows}")
    print(f"  avg_track_duration_frames : {facts.avg_track_duration_frames:.2f}")
    print(f"  avg_track_duration_secs   : {facts.avg_track_duration_seconds:.2f}")
    print(f"  avg_visible/window        : {facts.avg_visible_tracks_per_window:.2f}")
    if facts.longest_track:
        print(f"  longest_track             : {facts.longest_track['track_id']} ({facts.longest_track['duration_seconds']:.2f}s)")
    if facts.most_crowded_window:
        print(
            f"  most_crowded_window       : frames {facts.most_crowded_window['start_frame']}.."
            f"{facts.most_crowded_window['end_frame']} "
            f"with {facts.most_crowded_window['visible_count']} tracks"
        )


if __name__ == "__main__":
    main()