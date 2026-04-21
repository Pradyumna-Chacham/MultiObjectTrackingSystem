from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class ChunkRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []

    def encode_texts(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        if normalize:
            faiss.normalize_L2(embeddings)

        return embeddings

    def build_index(self, chunks: list[dict[str, Any]], normalize: bool = True) -> None:
        self.metadata = chunks
        texts = [str(c["text"]) for c in chunks]
        embeddings = self.encode_texts(texts, normalize=normalize)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim) if normalize else faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index

    def save(self, index_path: str, metadata_path: str) -> None:
        if self.index is None:
            raise ValueError("Index has not been built.")

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, index_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, index_path: str, metadata_path: str) -> None:
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(
        self,
        query: str,
        top_k: int = 5,
        normalize: bool = True,
        chunk_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index has not been loaded or built.")

        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        if normalize:
            faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k * 5 if chunk_type else top_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            item = dict(self.metadata[idx])
            item["score"] = float(score)

            if chunk_type is not None and item.get("chunk_type") != chunk_type:
                continue

            results.append(item)

            if len(results) >= top_k:
                break

        return results