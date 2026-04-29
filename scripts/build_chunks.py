from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag.chunker import ChunkBuilder
from src.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Build retrieval chunks")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--track_facts",
        "--track-facts",
        type=str,
        required=True,
        help="Tracks facts JSON file path",
    )
    parser.add_argument("--events", type=str, required=True)

    # 🔥 NEW: appearance-enhanced tracks
    parser.add_argument(
        "--tracks_with_appearance",
        type=str,
        required=False,
        help="Tracks JSON that contains appearance field"
    )

    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def merge_appearance(track_facts, tracks_with_appearance):
    """
    Inject appearance from tracks into track_facts
    """
    track_map = {
        int(t["track_id"]): t for t in tracks_with_appearance
    }

    for tf in track_facts:
        tid = int(tf["track_id"])

        if tid in track_map:
            tf["appearance"] = track_map[tid].get("appearance")
            tf["appearance_skip_reason"] = track_map[tid].get("appearance_skip_reason")
        else:
            tf["appearance"] = None
            tf["appearance_skip_reason"] = "missing_track"

    return track_facts


def main():
    args = parse_args()

    cfg = load_config(args.config)

    track_facts_path = Path(args.track_facts)
    events_path = Path(args.events)
    output_path = Path(args.output)

    track_facts = load_json(track_facts_path)
    events = load_json(events_path)

    # 🔥 Merge appearance if provided
    if args.tracks_with_appearance:
        tracks_with_app = load_json(Path(args.tracks_with_appearance))
        track_facts = merge_appearance(track_facts, tracks_with_app)
        print("✅ Appearance merged into track_facts")
    else:
        # graceful fallback
        for tf in track_facts:
            tf["appearance"] = None
            tf["appearance_skip_reason"] = "extractor_not_run"

        print("⚠️ No appearance file provided — continuing without appearance")

    fps = cfg.visualization.get("fps", 30.0)

    builder = ChunkBuilder(fps=fps)
    chunks = builder.build_all_chunks(track_facts, events)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    builder.save_chunks(chunks, str(output_path))

    print(f"\n✅ Chunks saved to: {output_path}")


if __name__ == "__main__":
    main()