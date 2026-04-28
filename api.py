from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.rag.retriever import ChunkRetriever
from src.rag.answer_engine import AnswerEngine


ROOT = Path(__file__).resolve().parent
TMP_DIR = ROOT / "demo" / "api_runs"
TMP_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = "configs/rtdetr_ocsort.yaml"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


app = FastAPI(title="Video RAG API", version="1.0.0")

# ------------------
# Request Models
# ------------------


class QueryRequest(BaseModel):
    run_id: str
    query: str
    top_k: int = 5
    chunk_type: str | None = None
    model_name: str = DEFAULT_MODEL


# ----------
# Helpers
# -----------
def run_command(cmd: list[str], cwd: Path | None = None) -> str:
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
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Command failed",
                "command": " ".join(cmd),
                "output": result.stdout,
            },
        )

    return result.stdout


def safe_stem(path_str: str) -> str:
    return Path(path_str).stem.replace(" ", "_")


def safe_slug(text: str, max_len: int = 60) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return (cleaned[:max_len] or "query").strip("_") or "query"


def make_run_paths(video_filename: str) -> dict[str, Path]:
    stem = safe_stem(video_filename)
    run_dir = TMP_DIR / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "input_video": run_dir / video_filename,
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
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Tracking finished, but tracks JSON was not found.",
                "searched_in": str(sample_outputs_dir),
                "existing_files": existing,
            },
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
        raise HTTPException(status_code=400, detail=f"Config not found: {cfg_path}")

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
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    video_facts_path = next(run_dir.glob("*.video_facts.json"), None)
    faiss_index_path = next(run_dir.glob("*.faiss.index"), None)
    index_meta_path = next(run_dir.glob("*.index_meta.json"), None)

    if video_facts_path is None or faiss_index_path is None or index_meta_path is None:
        raise HTTPException(
            status_code=400,
            detail="Required artifacts not found. Run the pipeline first.",
        )

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run-pipeline")
async def run_pipeline(
    video: UploadFile = File(...),
    config_path: str = Form(DEFAULT_CONFIG),
    fps: float = Form(default=30.0),
    frame_width: int = Form(default=1920),
    frame_height: int = Form(default=1080),
    model_name: str = Form(DEFAULT_MODEL),
) -> dict[str, Any]:
    """Run the multi-object tracking pipeline on an uploaded video."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="Uploaded video must have a filename")
    
    # Save uploaded video to temporary directory
    temp_input = TMP_DIR / video.filename
    temp_input.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_input, "wb") as f:
        contents = await video.read()
        f.write(contents)
    
    try:
        result = run_pipeline_backend(
            input_video=temp_input,
            config_path=config_path,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            model_name=model_name,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """Query the video knowledge base using RAG."""
    try:
        result = answer_query_backend(
            run_id=request.run_id,
            query=request.query,
            top_k=request.top_k,
            chunk_type=request.chunk_type,
            model_name=request.model_name,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
