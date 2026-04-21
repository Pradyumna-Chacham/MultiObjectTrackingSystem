from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json


@dataclass
class AnswerPackage:
    query: str
    answer_source: str
    final_answer: str
    supporting_fact_key: str | None
    supporting_fact_value: dict[str, Any] | int | float | str | None
    retrieved_chunks: list[dict[str, Any]]
    llm_context: str
    prompt: str
    intent: str


class AnswerEngine:
    """
    Hybrid answer engine for MOT/video RAG.

    Responsibilities:
    1. Prefer exact answers from precomputed video facts for common analytics queries
    2. Fall back to retrieved chunks for retrieval-heavy questions
    3. Build LLM-ready grounded context + prompt
    4. Return a structured package that can be:
       - printed directly
       - sent to an LLM
       - shown in a UI
    """

    def answer(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        video_facts: dict[str, Any] | None = None,
    ) -> AnswerPackage:
        fact_answer, fact_key, fact_value, intent = self._try_answer_from_facts(query, video_facts)

        if fact_answer is not None:
            final_answer = fact_answer
            answer_source = "video_facts"
        else:
            retrieval_answer, intent = self._answer_from_retrieval(query, retrieved_chunks)
            final_answer = retrieval_answer
            answer_source = "retrieval"
            fact_key = None
            fact_value = None

        llm_context = self.build_llm_context(
            query=query,
            video_facts=video_facts,
            supporting_fact_key=fact_key,
            supporting_fact_value=fact_value,
            retrieved_chunks=retrieved_chunks,
        )

        prompt = self.build_prompt(
            query=query,
            llm_context=llm_context,
        )

        return AnswerPackage(
            query=query,
            answer_source=answer_source,
            final_answer=final_answer,
            supporting_fact_key=fact_key,
            supporting_fact_value=fact_value,
            retrieved_chunks=retrieved_chunks,
            llm_context=llm_context,
            prompt=prompt,
            intent=intent,
        )

    # ------------------------------------------------------------------
    # Fact-first answering
    # ------------------------------------------------------------------

    def _try_answer_from_facts(
        self,
        query: str,
        facts: dict[str, Any] | None,
    ) -> tuple[str | None, str | None, dict[str, Any] | int | float | str | None, str]:
        if facts is None:
            return None, None, None, "no_facts"

        q = query.lower().strip()

        if "longest" in q and any(x in q for x in ["stay", "stayed", "visible", "present"]):
            t = facts.get("longest_track")
            if t:
                return (
                    f"Track {t['track_id']} stayed the longest for {t['duration_seconds']:.2f} seconds "
                    f"(frames {t['first_frame']}–{t['last_frame']}).",
                    "longest_track",
                    t,
                    "longest_presence",
                )

        if "shortest" in q and any(x in q for x in ["stay", "stayed", "visible", "present"]):
            t = facts.get("shortest_track")
            if t:
                return (
                    f"Track {t['track_id']} stayed the shortest for {t['duration_seconds']:.2f} seconds "
                    f"(frames {t['first_frame']}–{t['last_frame']}).",
                    "shortest_track",
                    t,
                    "shortest_presence",
                )

        if "crowded" in q or "peak" in q:
            w = facts.get("most_crowded_window")
            if w:
                return (
                    f"The scene was most crowded from frame {w['start_frame']} to {w['end_frame']} "
                    f"({w['start_time_seconds']:.2f}–{w['end_time_seconds']:.2f}s) "
                    f"with {w['visible_count']} active tracks.",
                    "most_crowded_window",
                    w,
                    "most_crowded",
                )

        if "how many" in q and "unique" in q:
            value = facts.get("total_unique_tracks")
            if value is not None:
                return (
                    f"There are {value} unique tracks in the video.",
                    "total_unique_tracks",
                    value,
                    "unique_count",
                )

        if "how many" in q and ("long" in q or "long-presence" in q or "long presence" in q):
            value = facts.get("total_long_presence_tracks")
            if value is not None:
                return (
                    f"There are {value} long-presence tracks in the video.",
                    "total_long_presence_tracks",
                    value,
                    "long_presence_count",
                )

        if "how many" in q and "fragmented" in q:
            value = facts.get("total_fragmented_tracks")
            if value is not None:
                return (
                    f"There are {value} fragmented tracks in the video.",
                    "total_fragmented_tracks",
                    value,
                    "fragmented_count",
                )

        if "entry" in q and "count" in q:
            value = facts.get("entry_counts_by_side")
            if value is not None:
                return (
                    f"Entry counts by side are: {value}.",
                    "entry_counts_by_side",
                    value,
                    "entry_counts",
                )

        if "exit" in q and "count" in q:
            value = facts.get("exit_counts_by_side")
            if value is not None:
                return (
                    f"Exit counts by side are: {value}.",
                    "exit_counts_by_side",
                    value,
                    "exit_counts",
                )

        if "direction" in q and "count" in q:
            value = facts.get("direction_counts")
            if value is not None:
                return (
                    f"Direction counts are: {value}.",
                    "direction_counts",
                    value,
                    "direction_counts",
                )

        return None, None, None, "fact_fallback"

    # ------------------------------------------------------------------
    # Retrieval answering fallback
    # ------------------------------------------------------------------

    def _answer_from_retrieval(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> tuple[str, str]:
        if not retrieved_chunks:
            return "No relevant evidence was retrieved.", "no_results"

        top = retrieved_chunks[0]
        chunk_type = top.get("chunk_type", "unknown")

        if chunk_type == "track":
            return f"Best retrieved track evidence: {top.get('text', '')}", "retrieval_track"
        if chunk_type == "event":
            return f"Best retrieved event evidence: {top.get('text', '')}", "retrieval_event"
        if chunk_type == "time_window":
            return f"Best retrieved time-window evidence: {top.get('text', '')}", "retrieval_time_window"

        return f"Best retrieved evidence: {top.get('text', '')}", "retrieval_fallback"

    # ------------------------------------------------------------------
    # LLM context + prompt building
    # ------------------------------------------------------------------

    def build_llm_context(
        self,
        query: str,
        video_facts: dict[str, Any] | None,
        supporting_fact_key: str | None,
        supporting_fact_value: dict[str, Any] | int | float | str | None,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []

        lines.append("VIDEO QA CONTEXT")
        lines.append("=" * 60)
        lines.append(f"User query: {query}")
        lines.append("")

        if video_facts is not None:
            lines.append("GLOBAL VIDEO FACTS")
            lines.append("-" * 60)

            important_fact_keys = [
                "total_unique_tracks",
                "total_long_presence_tracks",
                "total_fragmented_tracks",
                "total_short_lived_tracks",
                "total_crowded_windows",
                "avg_track_duration_seconds",
                "avg_visible_tracks_per_window",
                "longest_track",
                "shortest_track",
                "most_crowded_window",
            ]

            for key in important_fact_keys:
                if key in video_facts:
                    lines.append(f"{key}: {json.dumps(video_facts[key])}")

            lines.append("")

        if supporting_fact_key is not None:
            lines.append("SUPPORTING FACT USED FOR ANSWER")
            lines.append("-" * 60)
            lines.append(f"fact_key: {supporting_fact_key}")
            lines.append(f"fact_value: {json.dumps(supporting_fact_value)}")
            lines.append("")

        lines.append("RETRIEVED CHUNKS")
        lines.append("-" * 60)

        if not retrieved_chunks:
            lines.append("No chunks retrieved.")
        else:
            for i, chunk in enumerate(retrieved_chunks, start=1):
                lines.append(
                    f"[{i}] chunk_id={chunk.get('chunk_id')} "
                    f"type={chunk.get('chunk_type')} "
                    f"score={chunk.get('score')}"
                )
                lines.append(
                    f"frames={chunk.get('start_frame')}..{chunk.get('end_frame')} "
                    f"tracks={chunk.get('track_ids')}"
                )
                lines.append(f"text={chunk.get('text', '')}")
                lines.append(f"metadata={json.dumps(chunk.get('metadata', {}))}")
                lines.append("")

        return "\n".join(lines)

    def build_prompt(
        self,
        query: str,
        llm_context: str,
    ) -> str:
        return f"""You are answering questions about a tracked video.

Use only the provided context.
Prefer exact structured facts when available.
If retrieved evidence is semantically related but does not contain the exact maximum/minimum/count, do not override exact facts.
Be explicit about frame ranges, times, and track IDs when possible.
Do not invent details.

{llm_context}

QUESTION:
{query}

RESPONSE FORMAT:
1. Direct answer
2. Supporting fact or evidence used
3. Mention any uncertainty briefly if needed
"""

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def to_dict(self, package: AnswerPackage) -> dict[str, Any]:
        return asdict(package)