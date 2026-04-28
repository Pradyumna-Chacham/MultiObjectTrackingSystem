from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import gradio as gr

from src.rag.retriever import ChunkRetriever
from src.rag.answer_engine import AnswerEngine


ROOT = Path(__file__).resolve().parent
TMP_DIR = ROOT / "demo" / "api_runs"
TMP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = "configs/rtdetr_ocsort.yaml"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

APP_CSS = """
#video_facts_box .cm-editor,
#fact_box .cm-editor,
#evidence_box .cm-editor {
    height: 400px !important;
    overflow-y: auto !important;
}

#video_facts_box .cm-scroller,
#fact_box .cm-scroller,
#evidence_box .cm-scroller {
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

#video_facts_box,
#fact_box,
#evidence_box {
    overflow: hidden !important;
}
"""


def run_command(cmd: list[str], cwd: Path | None = None) -> str:
    import os

    workdir = cwd or ROOT
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    result = subprocess.run(
        cmd,
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        raise gr.Error(
            f"Command failed with exit code {result.returncode}:\n\n"
            f"{' '.join(cmd)}\n\n"
            f"Output:\n{result.stdout}"
        )

    return result.stdout


def safe_stem(path_str: str) -> str:
    return Path(path_str).stem.replace(" ", "_")


def safe_slug(text: str, max_len: int = 60) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return (cleaned[:max_len] or "query").strip("_") or "query"


def make_run_paths(video_path: str) -> dict[str, Path]:
    stem = safe_stem(video_path)
    run_dir = TMP_DIR / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "input_video": run_dir / Path(video_path).name,
        "output_video": run_dir / f"{stem}.tracked.mp4",
        "mot_txt": run_dir / f"{stem}.mot.txt",
        "tracks_json": run_dir / f"{stem}.tracks.json",
        "built_tracks": run_dir / f"{stem}.built_tracks.json",
        "events_json": run_dir / f"{stem}.events.json",
        "track_facts": run_dir / f"{stem}.track_facts.json",
        "chunks_json": run_dir / f"{stem}.chunks.json",
        "video_facts": run_dir / f"{stem}.video_facts.json",
        "faiss_index": run_dir / f"{stem}.faiss.index",
        "index_meta": run_dir / f"{stem}.index_meta.json",
        "bundle_zip": run_dir / f"{stem}_bundle.zip",
    }


def copy_uploaded_video_to_run_dir(uploaded_video: str, run_dir: Path) -> Path:
    src = Path(uploaded_video)
    dst = run_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def build_bundle_zip(run_dir: Path, bundle_path: Path) -> str:
    if bundle_path.exists():
        bundle_path.unlink()

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.iterdir()):
            if not path.is_file():
                continue
            if path.resolve() == bundle_path.resolve():
                continue
            zf.write(path, arcname=path.name)

    return str(bundle_path)


def copy_latest_pipeline_outputs(paths: dict[str, Path]) -> None:
    sample_outputs_dir = ROOT / "demo" / "sample_outputs"

    track_candidates = sorted(
        sample_outputs_dir.glob("*.tracks.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    mot_candidates = sorted(
        sample_outputs_dir.glob("*.mot.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not track_candidates:
        existing = "\n".join(str(p) for p in sample_outputs_dir.glob("*"))
        raise gr.Error(
            "Tracking finished, but tracks JSON was not found.\n\n"
            f"Searched in: {sample_outputs_dir}\n\n"
            f"Existing files there:\n{existing}"
        )

    shutil.copy2(track_candidates[0], paths["tracks_json"])

    if mot_candidates:
        shutil.copy2(mot_candidates[0], paths["mot_txt"])


def run_pipeline_backend(
    input_video: Path,
    config_path: str,
    fps: float,
    frame_width: int,
    frame_height: int,
    model_name: str,
) -> dict[str, Any]:
    cfg_path = ROOT / config_path
    if not cfg_path.exists():
        raise gr.Error(f"Config not found: {cfg_path}")

    paths = make_run_paths(input_video.name)

    if input_video.resolve() != paths["input_video"].resolve():
        shutil.copy2(input_video, paths["input_video"])

    logs: list[str] = []

    cmd_demo = [
        sys.executable,
        "scripts/run_demo.py",
        "--config",
        config_path,
        "--input",
        str(paths["input_video"]),
        "--output",
        str(paths["output_video"]),
    ]
    logs.append(">>> Running tracking pipeline")
    logs.append(run_command(cmd_demo))

    copy_latest_pipeline_outputs(paths)

    cmd_tracks = [
        sys.executable,
        "scripts/build_tracks.py",
        "--input",
        str(paths["tracks_json"]),
        "--output",
        str(paths["built_tracks"]),
        "--fps",
        str(fps),
        "--frame-width",
        str(frame_width),
        "--frame-height",
        str(frame_height),
    ]
    logs.append(">>> Building consolidated tracks")
    logs.append(run_command(cmd_tracks))

    cmd_events = [
        sys.executable,
        "scripts/extract_events.py",
        "--input",
        str(paths["built_tracks"]),
        "--events-output",
        str(paths["events_json"]),
        "--facts-output",
        str(paths["track_facts"]),
        "--fps",
        str(fps),
    ]
    logs.append(">>> Extracting events")
    logs.append(run_command(cmd_events))

    cmd_chunks = [
        sys.executable,
        "scripts/build_chunks.py",
        "--config",
        config_path,
        "--track_facts",
        str(paths["track_facts"]),
        "--events",
        str(paths["events_json"]),
        "--output",
        str(paths["chunks_json"]),
    ]
    logs.append(">>> Building retrieval chunks")
    logs.append(run_command(cmd_chunks))

    cmd_video_facts = [
        sys.executable,
        "scripts/build_video_facts.py",
        "--track-facts",
        str(paths["track_facts"]),
        "--events",
        str(paths["events_json"]),
        "--chunks",
        str(paths["chunks_json"]),
        "--fps",
        str(fps),
        "--output",
        str(paths["video_facts"]),
    ]
    logs.append(">>> Building video facts")
    logs.append(run_command(cmd_video_facts))

    cmd_index = [
        sys.executable,
        "scripts/build_index.py",
        "--chunks",
        str(paths["chunks_json"]),
        "--index-output",
        str(paths["faiss_index"]),
        "--metadata-output",
        str(paths["index_meta"]),
        "--model",
        model_name,
    ]
    logs.append(">>> Building FAISS index")
    logs.append(run_command(cmd_index))

    with open(paths["video_facts"], "r", encoding="utf-8") as f:
        video_facts = json.load(f)

    bundle_zip = build_bundle_zip(paths["run_dir"], paths["bundle_zip"])

    return {
        "run_id": paths["run_dir"].name,
        "run_dir": str(paths["run_dir"]),
        "output_video": str(paths["output_video"]),
        "mot_txt": str(paths["mot_txt"]) if paths["mot_txt"].exists() else None,
        "tracks_json": str(paths["tracks_json"]),
        "built_tracks": str(paths["built_tracks"]),
        "events_json": str(paths["events_json"]),
        "track_facts": str(paths["track_facts"]),
        "chunks_json": str(paths["chunks_json"]),
        "video_facts": video_facts,
        "video_facts_path": str(paths["video_facts"]),
        "faiss_index": str(paths["faiss_index"]),
        "index_meta": str(paths["index_meta"]),
        "bundle_zip": bundle_zip,
        "logs": "\n\n".join(logs),
    }


def answer_query_backend(
    run_id: str,
    query: str,
    top_k: int,
    chunk_type: str | None,
    model_name: str,
) -> dict[str, Any]:
    run_dir = TMP_DIR / run_id
    if not run_dir.exists():
        raise gr.Error(f"Run not found: {run_id}")

    video_facts_path = next(run_dir.glob("*.video_facts.json"), None)
    faiss_index_path = next(run_dir.glob("*.faiss.index"), None)
    index_meta_path = next(run_dir.glob("*.index_meta.json"), None)

    if video_facts_path is None or faiss_index_path is None or index_meta_path is None:
        raise gr.Error("Required artifacts not found. Run the pipeline first.")

    with open(video_facts_path, "r", encoding="utf-8") as f:
        video_facts = json.load(f)

    retriever = ChunkRetriever(model_name=model_name)
    retriever.load(str(faiss_index_path), str(index_meta_path))

    results = retriever.search(
        query=query,
        top_k=int(top_k),
        chunk_type=None if chunk_type in (None, "", "auto") else chunk_type,
    )

    engine = AnswerEngine()
    package = engine.answer(
        query=query,
        retrieved_chunks=results,
        video_facts=video_facts,
    )

    return {
        "run_id": run_id,
        "query": query,
        "answer": package.final_answer,
        "answer_source": "video_facts" if package.supporting_fact_key is not None else "retrieval",
        "supporting_fact_key": package.supporting_fact_key,
        "supporting_fact_value": package.supporting_fact_value,
        "retrieved_evidence": results,
    }


def run_tracking_pipeline(
    uploaded_video: str,
    config_path: str,
    fps: float,
    frame_width: int,
    frame_height: int,
    model_name: str,
) -> tuple[str, str, str, str, str, str, str, str, str, str, str]:
    if not uploaded_video:
        raise gr.Error("Please upload a video.")

    cfg = config_path.strip() or DEFAULT_CONFIG
    cfg_path = ROOT / cfg
    if not cfg_path.exists():
        raise gr.Error(f"Config not found: {cfg_path}")

    paths = make_run_paths(uploaded_video)
    input_video = copy_uploaded_video_to_run_dir(uploaded_video, paths["run_dir"])

    logs: list[str] = []

    cmd_demo = [
        sys.executable,
        "scripts/run_demo.py",
        "--config",
        cfg,
        "--input",
        str(input_video),
        "--output",
        str(paths["output_video"]),
    ]
    logs.append(">>> Running tracking pipeline")
    logs.append(run_command(cmd_demo))

    sample_outputs_dir = ROOT / "demo" / "sample_outputs"

    track_candidates = sorted(
        sample_outputs_dir.glob("*.tracks.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    mot_candidates = sorted(
        sample_outputs_dir.glob("*.mot.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if track_candidates:
        shutil.copy2(track_candidates[0], paths["tracks_json"])
    else:
        existing = "\n".join(str(p) for p in sample_outputs_dir.glob("*"))
        raise gr.Error(
            "Tracking finished, but tracks JSON was not found.\n\n"
            f"Searched in: {sample_outputs_dir}\n\n"
            f"Existing files there:\n{existing}"
        )

    if mot_candidates:
        shutil.copy2(mot_candidates[0], paths["mot_txt"])

    cmd_tracks = [
        sys.executable,
        "scripts/build_tracks.py",
        "--input",
        str(paths["tracks_json"]),
        "--output",
        str(paths["built_tracks"]),
        "--fps",
        str(fps),
        "--frame-width",
        str(frame_width),
        "--frame-height",
        str(frame_height),
    ]
    logs.append(">>> Building consolidated tracks")
    logs.append(run_command(cmd_tracks))

    cmd_events = [
        sys.executable,
        "scripts/extract_events.py",
        "--input",
        str(paths["built_tracks"]),
        "--events-output",
        str(paths["events_json"]),
        "--facts-output",
        str(paths["track_facts"]),
        "--fps",
        str(fps),
    ]
    logs.append(">>> Extracting events")
    logs.append(run_command(cmd_events))

    cmd_chunks = [
        sys.executable,
        "scripts/build_chunks.py",
        "--track-facts",
        str(paths["track_facts"]),
        "--events",
        str(paths["events_json"]),
        "--output",
        str(paths["chunks_json"]),
        "--fps",
        str(fps),
    ]
    logs.append(">>> Building retrieval chunks")
    logs.append(run_command(cmd_chunks))

    cmd_video_facts = [
        sys.executable,
        "scripts/build_video_facts.py",
        "--track-facts",
        str(paths["track_facts"]),
        "--events",
        str(paths["events_json"]),
        "--chunks",
        str(paths["chunks_json"]),
        "--fps",
        str(fps),
        "--output",
        str(paths["video_facts"]),
    ]
    logs.append(">>> Building video facts")
    logs.append(run_command(cmd_video_facts))

    cmd_index = [
        sys.executable,
        "scripts/build_index.py",
        "--chunks",
        str(paths["chunks_json"]),
        "--index-output",
        str(paths["faiss_index"]),
        "--metadata-output",
        str(paths["index_meta"]),
        "--model",
        model_name,
    ]
    logs.append(">>> Building FAISS index")
    logs.append(run_command(cmd_index))

    with open(paths["video_facts"], "r", encoding="utf-8") as f:
        video_facts = json.load(f)

    bundle_zip_path = build_bundle_zip(paths["run_dir"], paths["bundle_zip"])

    return (
        str(paths["output_video"]),
        str(paths["mot_txt"]) if paths["mot_txt"].exists() else "",
        str(paths["tracks_json"]),
        str(paths["built_tracks"]),
        str(paths["events_json"]),
        str(paths["chunks_json"]),
        json.dumps(video_facts, indent=2),
        "\n\n".join(logs),
        str(paths["run_dir"]),
        str(paths["video_facts"]),
        bundle_zip_path,
    )


def answer_question(
    run_dir_str: str,
    query: str,
    top_k: int,
    chunk_type: str,
    model_name: str,
) -> tuple[str, str, str, str | None, str]:
    if not run_dir_str:
        raise gr.Error("Please run the pipeline first.")
    if not query.strip():
        raise gr.Error("Please enter a question.")

    result = answer_query_backend(
        run_id=run_dir_str,
        query=query,
        top_k=top_k,
        chunk_type=None if chunk_type in (None, "", "auto") else chunk_type,
        model_name=model_name,
    )

    fact_text = ""
    fact_download_path: Path | None = None

    if result["supporting_fact_key"] is not None:
        fact_payload: dict[str, Any] = {
            "fact_key": result["supporting_fact_key"],
            "fact_value": result["supporting_fact_value"],
        }
        fact_text = json.dumps(fact_payload, indent=2)
        fact_download_path = Path(run_dir_str) / "supporting_fact.json"
        with open(fact_download_path, "w", encoding="utf-8") as f:
            json.dump(fact_payload, f, indent=2)

    evidence_text = json.dumps(result["retrieved_evidence"], indent=2)
    evidence_download_path = Path(run_dir_str) / f"retrieved_evidence.json"
    with open(evidence_download_path, "w", encoding="utf-8") as f:
        json.dump(result["retrieved_evidence"], f, indent=2)

    return (
        result["answer"],
        fact_text,
        evidence_text,
        str(fact_download_path) if fact_download_path else None,
        str(evidence_download_path),
    )


with gr.Blocks(title="RT-DETR + OC-SORT Video RAG", css=APP_CSS) as demo:
    gr.Markdown(
        """
        # Multi Object Tracking System with AI-Driven Retrieval and Question Answering

        Upload a video, run multi-object tracking, build structured video knowledge,
        and ask grounded questions about the result.
        """
    )

    run_dir_state = gr.State("")

    with gr.Tab("1) Run Pipeline"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                config_input = gr.Textbox(label="Config Path", value=DEFAULT_CONFIG)
                fps_input = gr.Number(label="FPS", value=30.0)
                frame_width_input = gr.Number(label="Frame Width", value=1920)
                frame_height_input = gr.Number(label="Frame Height", value=1080)
                model_input = gr.Textbox(label="Embedding Model", value=DEFAULT_MODEL)
                run_button = gr.Button("Run Tracking + Build RAG Artifacts", variant="primary")

            with gr.Column():
                output_video = gr.Video(label="Tracked Output Video")
                video_facts_box = gr.Textbox(
                    label="Video Facts",
                    lines=18,
                    interactive=False,
                    show_copy_button=True,
                    elem_id="video_facts_box",
                )
                with gr.Row():
                    video_facts_file = gr.File(label="Download Video Facts JSON")
                    bundle_file = gr.File(label="Download Full Bundle ZIP")
                logs_box = gr.Textbox(label="Logs", lines=20, interactive=False)

        with gr.Row():
            mot_file = gr.File(label="MOT Output")
            tracks_file = gr.File(label="Raw Tracks JSON")
            built_tracks_file = gr.File(label="Built Tracks JSON")
            events_file = gr.File(label="Events JSON")
            chunks_file = gr.File(label="Chunks JSON")

        run_button.click(
            fn=run_tracking_pipeline,
            inputs=[
                video_input,
                config_input,
                fps_input,
                frame_width_input,
                frame_height_input,
                model_input,
            ],
            outputs=[
                output_video,
                mot_file,
                tracks_file,
                built_tracks_file,
                events_file,
                chunks_file,
                video_facts_box,
                logs_box,
                run_dir_state,
                video_facts_file,
                bundle_file,
            ],
        )

    with gr.Tab("2) Ask Questions"):
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Which person stayed the longest?",
                )
                top_k_input = gr.Slider(label="Top K", minimum=1, maximum=20, value=10, step=1)
                chunk_type_input = gr.Dropdown(
                    label="Chunk Type Filter",
                    choices=["auto", "track", "event", "time_window"],
                    value="auto",
                )
                model_query_input = gr.Textbox(label="Embedding Model", value=DEFAULT_MODEL)
                ask_button = gr.Button("Ask", variant="primary")

            with gr.Column():
                answer_box = gr.Textbox(label="Final Answer", lines=4, interactive=False)
                fact_box = gr.Textbox(
                    label="Supporting Fact",
                    lines=10,
                    interactive=False,
                    show_copy_button=True,
                    elem_id="fact_box",
                )
                evidence_box = gr.Textbox(
                    label="Retrieved Evidence",
                    lines=16,
                    interactive=False,
                    show_copy_button=True,
                    elem_id="evidence_box",
                )
                with gr.Row():
                    fact_file = gr.File(label="Download Supporting Fact JSON")
                    evidence_file = gr.File(label="Download Retrieved Evidence JSON")

        ask_button.click(
            fn=answer_question,
            inputs=[run_dir_state, question_input, top_k_input, chunk_type_input, model_query_input],
            outputs=[answer_box, fact_box, evidence_box, fact_file, evidence_file],
        )

    with gr.Tab("3) Notes"):
        gr.Markdown(
            """
            ## Recommended deployment path
            Start with uploaded-video demos first. Add webcam/live mode after the core app is stable.

            ## Required repo contents
            This app assumes your scripts and source tree already exist:
            - `scripts/run_demo.py`
            - `scripts/build_tracks.py`
            - `scripts/extract_events.py`
            - `scripts/build_chunks.py`
            - `scripts/build_video_facts.py`
            - `scripts/build_index.py`
            - `src/rag/retriever.py`
            - `src/rag/answer_engine.py`
            """
        )


if __name__ == "__main__":
    demo.launch()