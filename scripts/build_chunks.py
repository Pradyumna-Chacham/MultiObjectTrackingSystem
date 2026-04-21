from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.chunker import ChunkBuilder, load_json_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval chunks from track facts and events.")
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
        "--output",
        required=False,
        help="Path to save chunks JSON.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Video FPS.",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=30,
        help="Time-window chunk size in frames.",
    )
    return parser.parse_args()


def default_output_path(track_facts_path: Path) -> Path:
    name = track_facts_path.name
    if name.endswith(".track_facts.json"):
        return track_facts_path.with_name(name.replace(".track_facts.json", ".chunks.json"))
    return track_facts_path.with_name(f"{track_facts_path.stem}.chunks.json")


def main() -> None:
    args = parse_args()

    track_facts_path = Path(args.track_facts)
    events_path = Path(args.events)

    if not track_facts_path.exists():
        raise FileNotFoundError(f"Track facts file not found: {track_facts_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    output_path = Path(args.output) if args.output else default_output_path(track_facts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    track_facts = load_json_records(str(track_facts_path))
    events = load_json_records(str(events_path))

    builder = ChunkBuilder(
        fps=args.fps,
        window_frames=args.window_frames,
    )

    chunks = builder.build_all_chunks(track_facts, events)
    builder.save_chunks(chunks, str(output_path))

    print("Chunk building complete.")
    print(f"Track facts : {len(track_facts)}")
    print(f"Events      : {len(events)}")
    print(f"Chunks      : {len(chunks)}")
    print(f"Output      : {output_path}")

    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.chunk_type] = counts.get(c.chunk_type, 0) + 1

    if counts:
        print("Chunk counts:")
        for k in sorted(counts):
            print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()