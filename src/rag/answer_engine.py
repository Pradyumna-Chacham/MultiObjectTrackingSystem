from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AnswerPackage:
    final_answer: str
    supporting_fact_key: str | None = None
    supporting_fact_value: Any | None = None


class AnswerEngine:
    def __init__(self) -> None:
        pass

    def answer(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        video_facts: dict[str, Any] | None = None,
    ) -> AnswerPackage:
        q = query.lower().strip()

        fact_answer = self._try_answer_from_video_facts(q, video_facts)
        if fact_answer is not None:
            return fact_answer

        if self._is_appearance_query(q):
            appearance_answer = self._handle_appearance_query(q, retrieved_chunks)
            if appearance_answer is not None:
                return AnswerPackage(final_answer=appearance_answer)

        if self._is_describe_everyone_query(q):
            everyone_answer = self._handle_describe_everyone(retrieved_chunks)
            return AnswerPackage(final_answer=everyone_answer)

        return AnswerPackage(final_answer=self._fallback_answer(retrieved_chunks))

    # ---------------------------------------------------------
    # Fact-based answering
    # ---------------------------------------------------------
    def _try_answer_from_video_facts(
        self,
        q: str,
        video_facts: dict[str, Any] | None,
    ) -> AnswerPackage | None:
        if video_facts is None:
            return None

        if "longest" in q:
            t = video_facts.get("longest_track")
            if t:
                return AnswerPackage(
                    final_answer=(
                        f"Track {t['track_id']} stayed the longest for "
                        f"{t['duration_seconds']:.2f} seconds "
                        f"(frames {t['first_frame']}–{t['last_frame']})."
                    ),
                    supporting_fact_key="longest_track",
                    supporting_fact_value=t,
                )

        if "crowded" in q or "peak" in q:
            w = video_facts.get("most_crowded_window")
            if w:
                return AnswerPackage(
                    final_answer=(
                        f"The scene was most crowded from frame {w['start_frame']} to {w['end_frame']} "
                        f"({w['start_time_seconds']:.2f}–{w['end_time_seconds']:.2f}s) "
                        f"with {w['visible_count']} tracks."
                    ),
                    supporting_fact_key="most_crowded_window",
                    supporting_fact_value=w,
                )

        if "how many" in q and "unique" in q:
            value = video_facts.get("total_unique_tracks")
            return AnswerPackage(
                final_answer=f"There are {value} unique tracks.",
                supporting_fact_key="total_unique_tracks",
                supporting_fact_value=value,
            )

        if "long" in q and "how many" in q:
            value = video_facts.get("total_long_presence_tracks")
            return AnswerPackage(
                final_answer=f"There are {value} long-presence tracks.",
                supporting_fact_key="total_long_presence_tracks",
                supporting_fact_value=value,
            )

        if "fragmented" in q and "how many" in q:
            value = video_facts.get("total_fragmented_tracks")
            return AnswerPackage(
                final_answer=f"There are {value} fragmented tracks.",
                supporting_fact_key="total_fragmented_tracks",
                supporting_fact_value=value,
            )

        side = self._extract_side(q)

        if ("enter" in q or "entered" in q or "entry" in q) and side is not None and ("how many" in q or "count" in q):
            counts = video_facts.get("entry_counts_by_side", {})
            value = int(counts.get(side, 0))
            payload = {"side": side, "count": value, "all_counts": counts}
            return AnswerPackage(
                final_answer=f"{value} people entered from the {side}.",
                supporting_fact_key="entry_counts_by_side",
                supporting_fact_value=payload,
            )

        if ("exit" in q or "exited" in q) and side is not None and ("how many" in q or "count" in q):
            counts = video_facts.get("exit_counts_by_side", {})
            value = int(counts.get(side, 0))
            payload = {"side": side, "count": value, "all_counts": counts}
            return AnswerPackage(
                final_answer=f"{value} people exited from the {side}.",
                supporting_fact_key="exit_counts_by_side",
                supporting_fact_value=payload,
            )

        return None

    # ---------------------------------------------------------
    # Appearance answering
    # ---------------------------------------------------------
    def _is_appearance_query(self, q: str) -> bool:
        keywords = ["wear", "wearing", "color", "clothes", "shirt", "pants", "top", "bottom"]
        return any(k in q for k in keywords)

    def _handle_appearance_query(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> str | None:
        track_id = self._extract_track_id(query)

        if track_id is not None:
            for ch in chunks:
                if ch.get("chunk_type") != "appearance":
                    continue
                if track_id not in ch.get("track_ids", []):
                    continue

                meta = ch.get("metadata", {})
                upper = meta.get("upper_color")
                lower = meta.get("lower_color")
                low_conf = bool(meta.get("low_confidence", False))

                if low_conf:
                    return (
                        f"Track {track_id} appears to be wearing {upper} on top "
                        f"and {lower} on the bottom, but detection confidence is low."
                    )

                if upper and lower:
                    return f"Track {track_id} is wearing {upper} on top and {lower} on the bottom."
                if upper:
                    return f"Track {track_id} is wearing {upper} on top."
                if lower:
                    return f"Track {track_id} is wearing {lower} on the bottom."

            return f"Appearance data is not available for track {track_id}."

        colors = ["red", "blue", "black", "white", "gray", "green", "yellow", "navy", "brown", "orange", "pink"]
        mentioned_colors = [c for c in colors if c in query]
        if mentioned_colors:
            matches: list[int] = []
            for ch in chunks:
                if ch.get("chunk_type") != "appearance":
                    continue
                meta = ch.get("metadata", {})
                tid = meta.get("track_id")
                upper = meta.get("upper_color")
                lower = meta.get("lower_color")
                for c in mentioned_colors:
                    if c == upper or c == lower:
                        matches.append(int(tid))
                        break

            if matches:
                matches = sorted(set(matches))
                return f"Tracks matching the description: {matches}"

            return "No matching person found with that color."

        return None

    # ---------------------------------------------------------
    # Scene summary
    # ---------------------------------------------------------
    def _is_describe_everyone_query(self, q: str) -> bool:
        return "describe everyone" in q or "all people" in q

    def _handle_describe_everyone(self, chunks: list[dict[str, Any]]) -> str:
        results: list[str] = []

        for ch in chunks:
            if ch.get("chunk_type") != "appearance":
                continue

            meta = ch.get("metadata", {})
            if meta.get("low_confidence", False):
                continue

            tid = meta.get("track_id")
            upper = meta.get("upper_color")
            lower = meta.get("lower_color")

            if upper and lower:
                results.append(f"Track {tid}: {upper} top, {lower} bottom")
            elif upper:
                results.append(f"Track {tid}: {upper} top")
            elif lower:
                results.append(f"Track {tid}: {lower} bottom")

        if not results:
            return "No reliable appearance data available."

        return "\n".join(results)

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------
    def _fallback_answer(self, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant information found."
        return chunks[0].get("text", "No answer available.")

    # ---------------------------------------------------------
    # Utils
    # ---------------------------------------------------------
    def _extract_track_id(self, query: str) -> int | None:
        words = query.split()
        for i, w in enumerate(words):
            if w == "track" and i + 1 < len(words):
                try:
                    return int(words[i + 1])
                except ValueError:
                    return None
        return None

    def _extract_side(self, q: str) -> str | None:
        for side in ["left", "right", "top", "bottom", "interior"]:
            if side in q:
                return side
        return None