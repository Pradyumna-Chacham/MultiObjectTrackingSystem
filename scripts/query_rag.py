from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag.retriever import ChunkRetriever
from src.rag.answer_engine import AnswerEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query retrieval index for MOT RAG.")

    parser.add_argument("--index", required=True, help="Path to FAISS index file.")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON file.")
    parser.add_argument("--query", required=True, help="Natural language query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks to return.")
    parser.add_argument(
        "--chunk-type",
        default=None,
        choices=["track", "event", "time_window", "appearance", None],
        help="Optional chunk type filter.",
    )
    parser.add_argument("--video-facts", required=False, help="Optional path to video_facts.json")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )

    return parser.parse_args()


def load_video_facts(path: str | None) -> dict | None:
    if path is None:
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video facts file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    index_path = Path(args.index)
    metadata_path = Path(args.metadata)

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    retriever = ChunkRetriever(model_name=args.model)
    retriever.load(str(index_path), str(metadata_path))

    video_facts = load_video_facts(args.video_facts)

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        chunk_type=args.chunk_type,
    )

    engine = AnswerEngine()
    package = engine.answer(
        query=args.query,
        retrieved_chunks=results,
        video_facts=video_facts,
    )

    answer_source = "video_facts" if package.supporting_fact_key is not None else "retrieval"

    print("=" * 90)
    print(f"Query: {args.query}")
    print(f"Answer source: {answer_source}")
    print("=" * 90)

    print("\n🧠 FINAL ANSWER:\n")
    print(package.final_answer)

    if package.supporting_fact_key is not None:
        print("\n📌 SUPPORTING FACT (video_facts.json):\n")
        print(f"Fact key: {package.supporting_fact_key}")
        print(json.dumps(package.supporting_fact_value, indent=2))

        print("\nℹ️ NOTE:\n")
        print(
            "The final answer was computed from precomputed video facts. "
            "The retrieved evidence below is supporting semantic context and may not contain the exact extreme/global fact."
        )

    print("\n📊 RETRIEVED EVIDENCE:\n")

    for i, item in enumerate(results, start=1):
        print(f"[{i}] score={item['score']:.4f} type={item.get('chunk_type')} id={item.get('chunk_id')}")
        print(f"frames={item.get('start_frame')}..{item.get('end_frame')} tracks={item.get('track_ids')}")
        print(item.get("text", ""))
        print("-" * 90)

    print("\n📦 RAW JSON:\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()