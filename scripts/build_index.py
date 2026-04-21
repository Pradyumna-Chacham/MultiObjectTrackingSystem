from __future__ import annotations

import argparse
from pathlib import Path
import json

from src.rag.retriever import ChunkRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from retrieval chunks.")
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to chunks JSON, e.g. demo/sample_outputs/mot17-04-ocsort.chunks.json",
    )
    parser.add_argument(
        "--index-output",
        required=False,
        help="Path to save FAISS index file.",
    )
    parser.add_argument(
        "--metadata-output",
        required=False,
        help="Path to save chunk metadata JSON.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    return parser.parse_args()


def default_index_path(chunks_path: Path) -> Path:
    if chunks_path.name.endswith(".chunks.json"):
        return chunks_path.with_name(chunks_path.name.replace(".chunks.json", ".faiss.index"))
    return chunks_path.with_name(f"{chunks_path.stem}.faiss.index")


def default_metadata_path(chunks_path: Path) -> Path:
    if chunks_path.name.endswith(".chunks.json"):
        return chunks_path.with_name(chunks_path.name.replace(".chunks.json", ".index_meta.json"))
    return chunks_path.with_name(f"{chunks_path.stem}.index_meta.json")


def main() -> None:
    args = parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    index_output = Path(args.index_output) if args.index_output else default_index_path(chunks_path)
    metadata_output = Path(args.metadata_output) if args.metadata_output else default_metadata_path(chunks_path)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    retriever = ChunkRetriever(model_name=args.model)
    retriever.build_index(chunks)
    retriever.save(str(index_output), str(metadata_output))

    print("Index build complete.")
    print(f"Chunks         : {len(chunks)}")
    print(f"Index output   : {index_output}")
    print(f"Metadata output: {metadata_output}")


if __name__ == "__main__":
    main()