---
title: MultiObjectTracking RAG App
emoji: 🎥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Multi-Object Tracking RAG App

Upload a video, run RT-DETR + OC-SORT tracking, generate tracking artifacts, and ask natural-language questions about the video.

# Multi-Object Tracking with RT-DETR, OC-SORT, and Video RAG

This repository is a modular video analytics project that combines:

- object detection
- multi-object tracking
- MOTChallenge-format export and evaluation
- structured track/event summarization
- retrieval-augmented querying over tracking results
- a Gradio app for upload, processing, and question answering

The codebase started with a YOLO/DeepSORT-style pipeline and now also contains a newer RT-DETR + OC-SORT + RAG workflow. In the current workspace, the RT-DETR + OC-SORT path is the most complete and safest path to document as the primary quick start.

## What The Project Does

At a high level, the project processes a video in stages:

1. Read frames from an input video.
2. Detect objects in each frame with an Ultralytics model.
3. Associate detections over time with a tracker.
4. Draw tracked bounding boxes, IDs, and trails on an output video.
5. Export frame-level tracking results to:
   - an annotated `.mp4`
   - a frame-level `.tracks.json`
   - a MOT-format `.mot.txt`
6. Consolidate frame snapshots into per-track histories.
7. Extract structured events and per-track facts.
8. Convert those facts into retrieval chunks.
9. Build a FAISS index over the chunks.
10. Answer grounded natural-language questions using precomputed video facts plus retrieval evidence.

## Main Features

- Modular detector and tracker factories under `src/detectors/` and `src/trackers/`
- End-to-end pipeline orchestration in `src/pipeline/orchestrator.py`
- Annotated video generation with per-track labels and motion trails
- MOT-format export for benchmark-style evaluation
- MOT evaluation script using `motmetrics`
- Track consolidation and event extraction for downstream analytics
- FAISS-based retrieval over tracking-derived text chunks
- Gradio interface for uploaded-video processing and QA

## Current Architecture

### Tracking pipeline

The tracking path is driven by:

- `scripts/run_demo.py`
- `src/config.py`
- `src/detectors/factory.py`
- `src/trackers/factory.py`
- `src/pipeline/orchestrator.py`
- `src/io/video_reader.py`
- `src/io/video_writer.py`
- `src/io/mot_exporter.py`
- `src/annotator.py`

The orchestrator loads the detector, resets the tracker, reads frames, runs detection and tracking, annotates frames, writes the output video, and optionally saves:

- `*.tracks.json`
- `*.mot.txt`

### RAG/analytics pipeline

The structured analytics path is driven by:

- `scripts/build_tracks.py`
- `scripts/extract_appearance.py`
- `scripts/extract_events.py`
- `scripts/build_chunks.py`
- `scripts/build_video_facts.py`
- `scripts/build_index.py`
- `scripts/query_rag.py`

Those scripts use the modules in `src/rag/` to transform tracking output into:

- built track histories
- extracted events
- track facts
- retrieval chunks
- video-level summary facts
- a FAISS index and metadata file

## Execution Order

The core processing pipeline runs in this order:

1. `scripts/run_demo.py`
   - runs the tracking pipeline, writes `*.tracks.json` and `*.mot.txt`
2. `scripts/build_tracks.py`
   - consolidates frame-level tracks into built track histories
3. `scripts/extract_appearance.py`
   - extracts clothing/appearance metadata for tracked subjects
4. `scripts/extract_events.py`
   - extracts structured events and per-track facts
5. `scripts/build_video_facts.py` *(optional: omit to skip to chunks-first path)*
   - builds global video facts from tracks and events (chunks are optional for richer statistics)
6. `scripts/build_chunks.py`
   - converts facts and events into retrieval chunks
7. `scripts/build_index.py`
   - creates a FAISS index and metadata for retrieval
8. `scripts/query_rag.py`
   - runs question-answering queries against the built artifacts

**Note:** Steps 5 and 6 are interchangeable. You can:
- Build video facts first (minimal), then chunks, or
- Build chunks first, then pass chunks to video facts for enriched statistics (current example order)

If you want appearance-aware color queries, run `scripts/extract_appearance.py` before `scripts/build_chunks.py`, then include the appearance-enhanced tracks file when chunking.

## App and API

- `app.py` is the Gradio front end for upload-based processing, artifact downloads, and QA.
- `api.py` provides the same underlying flow as a FastAPI server with:
  - `POST /run-pipeline`
  - `POST /query`
- Both use the same backend logic to run the pipeline and answer queries.

## Repository Structure

```text
.
├── api.py
├── app.py
├── configs/
│   ├── default.yaml
│   ├── rtdetr_ocsort.yaml
│   ├── rtdetr_ocsort_fast.yaml
│   ├── rtdetr_ocsort_x.yaml
│   └── ultralytics_deepsort.yaml
├── evaluation/
│   └── evaluate_mot.py
├── scripts/
│   ├── run_demo.py
│   ├── download_models.py
│   ├── build_tracks.py
│   ├── extract_appearance.py
│   ├── extract_events.py
│   ├── build_chunks.py
│   ├── build_video_facts.py
│   ├── build_index.py
│   └── query_rag.py
├── src/
│   ├── detectors/
│   ├── trackers/
│   ├── rag/
│   ├── io/
│   ├── captioning/
│   ├── pipeline/
│   ├── utils/
│   ├── annotator.py
│   ├── config.py
│   └── schemas.py
├── tests/
├── demo/
├── MOT17-04-FRCNN/
├── MOT17-09-FRCNN/
├── MOT17-11-FRCNN/
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Important Files And Folders

### Configs

- `configs/rtdetr_ocsort.yaml`
  Primary config for RT-DETR + OC-SORT.
- `configs/rtdetr_ocsort_fast.yaml`
  Faster, shorter run variant.
- `configs/rtdetr_ocsort_x.yaml`
  Higher-capacity RT-DETR-X variant.
- `configs/default.yaml`
  Older config intended for a DeepSORT-based path.
- `configs/ultralytics_deepsort.yaml`
  Another DeepSORT-oriented config.

### Demo folders

- `demo/sample_videos/`
  Local input videos for command-line runs.
- `demo/sample_outputs/`
  Generated videos and JSON/TXT artifacts.
- `demo/hf_runs/`
  Per-run Gradio app output bundles from the earlier UI flow.
- `demo/api_runs/`
  Per-run backend artifacts for the FastAPI/Gradio pipeline.

### MOT sequence folders

The workspace includes local MOT-style sequence folders such as:

- `MOT17-09-FRCNN/`
- `MOT17-11-FRCNN/`
- `MOT17-04-FRCNN/`

These are treated as local data, not portable source assets. The `.gitignore` excludes model weights, datasets, videos, generated outputs, and evaluation results, so anyone cloning this repo should expect to add those assets locally.

## Implemented Outputs

Depending on the workflow, the project can generate:

- `output.mp4`
  Annotated tracking video
- `output.tracks.json`
  Frame-level track snapshots
- `output.mot.txt`
  MOTChallenge-style prediction file
- `output.built_tracks.json`
  Consolidated track histories
- `output.events.json`
  Extracted event records
- `output.track_facts.json`
  Per-track analytics facts
- `output.chunks.json`
  Retrieval chunks
- `output.video_facts.json`
  Global summary facts
- `output.faiss.index`
  Vector index for retrieval
- `output.index_meta.json`
  Chunk metadata used during retrieval

## Detection And Tracking Backends

### Detector support

The detector factory currently supports:

- `ultralytics`
- `yolov9` placeholder/stub

In practice, the working configs use the Ultralytics integration with RT-DETR weights such as `rtdetr-l.pt` and `rtdetr-x.pt`.

### Tracker support

The tracker factory currently supports:

- `ocsort`
- `deepsort`

In this workspace, the `ocsort` path is the reliable documented path. The current DeepSORT configs do not include `tracker.max_iou_distance`, which the factory expects, so the RT-DETR + OC-SORT configs are the recommended choice for running the project as-is.

## RAG Workflow Details

After tracking finishes, the RAG pipeline adds structure in several stages:

### 1. Track building

`scripts/build_tracks.py` groups frame-level snapshots by `track_id` and computes:

- first/last frame
- duration
- average confidence
- displacement
- path length
- direction
- estimated entry/exit side
- short-lived and fragmented flags

### 2. Event extraction

`scripts/extract_events.py` emits events such as:

- `enter`
- `exit`
- `direction_motion`
- `long_presence`
- `fragmented_track`
- `crowded_window`

### 3. Chunk generation

`scripts/build_chunks.py` creates retrieval chunks of three types:

- `track`
- `event`
- `time_window`

### 4. Video facts

`scripts/build_video_facts.py` precomputes summary facts such as:

- total unique tracks
- longest track
- shortest track
- most crowded window
- entry/exit counts by side
- direction counts
- average track duration

**Note:** This script requires `track_facts` and `events`, but `chunks` is optional. If chunks are provided, they enrich the statistics; otherwise, facts are computed from tracks and events alone.

### 5. Retrieval and answering

`scripts/build_index.py` builds a FAISS index over chunk text using a sentence-transformer embedding model.

`scripts/query_rag.py` and `app.py` then:

- retrieve the most relevant chunks
- prefer exact answers from global video facts when possible
- fall back to retrieval evidence when needed

## Evaluation

`evaluation/evaluate_mot.py` compares a generated MOT prediction file against a MOT-style sequence directory containing:

- `gt/gt.txt`
- `seqinfo.ini`

The script computes metrics such as:

- MOTA
- MOTP
- IDF1
- ID precision / recall
- ID switches
- false positives
- misses
- mostly tracked / partially tracked / mostly lost

Results are written into `evaluation/results/`.

## Requirements

### Python

- Python 3.10+ is declared in `pyproject.toml`

### Main libraries

The repository uses packages from `requirements.txt`, including:

- `torch`
- `torchvision`
- `ultralytics`
- `boxmot`
- `deep-sort-realtime`
- `opencv-python`
- `motmetrics`
- `faiss-cpu`
- `sentence-transformers`
- `gradio`
- `fastapi`
- `pytest`
- `ruff`

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd MultiObjectTracking-RTDETR-OCSORT-RAG
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create local runtime folders

```bash
mkdir -p demo/sample_videos demo/sample_outputs demo/hf_runs evaluation/results models
```

If you prefer using the `Makefile`:

```bash
make install
make dirs
```

## Download Model Weights

The repository ignores model weights, so they need to exist locally.

To download the default RT-DETR-L weight used by the current download script:

```bash
PYTHONPATH=. venv/bin/python scripts/download_models.py
```

That script saves the model into:

```text
models/rtdetr-l.pt
```

The RT-DETR configs in this workspace currently reference:

- `rtdetr-l.pt`
- `rtdetr-x.pt`

If your chosen config points to a different location, either place the file there or update the config.

## Commands To Run The Project

The execution order for the full workflow is:

1. `scripts/run_demo.py`
2. `scripts/build_tracks.py`
3. `scripts/extract_appearance.py`
4. `scripts/extract_events.py`
5. `scripts/build_video_facts.py` *(optional first step)*
6. `scripts/build_chunks.py`
7. `scripts/build_index.py`
8. `scripts/query_rag.py`

This section is ordered from the simplest working path to the full analytics and QA workflow.

### 1. Run the main tracking pipeline

Recommended command:

```bash
PYTHONPATH=. venv/bin/python scripts/run_demo.py \
  --config configs/rtdetr_ocsort.yaml \
  --input demo/sample_videos/mot17_09_frcnn.mp4 \
  --output demo/sample_outputs/mot17-09-ocsort.mp4
```

This produces:

- `demo/sample_outputs/mot17-09-ocsort.mp4`
- `demo/sample_outputs/mot17-09-ocsort.tracks.json`
- `demo/sample_outputs/mot17-09-ocsort.mot.txt`

You can also run the faster variant:

```bash
PYTHONPATH=. venv/bin/python scripts/run_demo.py \
  --config configs/rtdetr_ocsort_fast.yaml \
  --input demo/sample_videos/mot17_09_frcnn.mp4 \
  --output demo/sample_outputs/mot17-09-ocsort-fast.mp4
```

And the RT-DETR-X variant:

```bash
PYTHONPATH=. venv/bin/python scripts/run_demo.py \
  --config configs/rtdetr_ocsort_x.yaml \
  --input demo/sample_videos/mot17_09_frcnn.mp4 \
  --output demo/sample_outputs/mot17-09-ocsort-x.mp4
```

### 2. Build consolidated tracks

```bash
PYTHONPATH=. venv/bin/python scripts/build_tracks.py \
  --input demo/sample_outputs/mot17-09-ocsort.tracks.json \
  --output demo/sample_outputs/mot17-09-ocsort.built_tracks.json \
  --fps 30 \
  --frame-width 1920 \
  --frame-height 1080
```

### 3. Extract appearance for tracked subjects

```bash
PYTHONPATH=. venv/bin/python scripts/extract_appearance.py \
  --config configs/rtdetr_ocsort.yaml \
  --tracks demo/sample_outputs/mot17-09-ocsort.built_tracks.json \
  --video demo/sample_videos/mot17_09_frcnn.mp4 \
  --output demo/sample_outputs/mot17-09-ocsort.built_tracks.appearance.json
```

### 4. Extract events and per-track facts

```bash
PYTHONPATH=. venv/bin/python scripts/extract_events.py \
  --input demo/sample_outputs/mot17-09-ocsort.built_tracks.appearance.json \
  --events-output demo/sample_outputs/mot17-09-ocsort.events.json \
  --facts-output demo/sample_outputs/mot17-09-ocsort.track_facts.json \
  --fps 30
```

This step preserves any `appearance` metadata on tracks so downstream video facts and chunk generation can use it.

If you prefer to keep the original built tracks file, you can instead pass the appearance-enhanced tracks file separately:

```bash
PYTHONPATH=. venv/bin/python scripts/extract_events.py \
  --input demo/sample_outputs/mot17-09-ocsort.built_tracks.json \
  --appearance-tracks demo/sample_outputs/mot17-09-ocsort.built_tracks.appearance.json \
  --events-output demo/sample_outputs/mot17-09-ocsort.events.json \
  --facts-output demo/sample_outputs/mot17-09-ocsort.track_facts.json \
  --fps 30
```

### 5. Build global video facts (optional: can skip to chunks if preferred)

Minimal approach (facts from events only):

```bash
PYTHONPATH=. venv/bin/python scripts/build_video_facts.py \
  --track-facts demo/sample_outputs/mot17-09-ocsort.track_facts.json \
  --events demo/sample_outputs/mot17-09-ocsort.events.json \
  --fps 30 \
  --output demo/sample_outputs/mot17-09-ocsort.video_facts.json
```

### 6. Build retrieval chunks

```bash
PYTHONPATH=. venv/bin/python scripts/build_chunks.py \
  --config configs/rtdetr_ocsort.yaml \
  --track_facts demo/sample_outputs/mot17-09-ocsort.track_facts.json \
  --events demo/sample_outputs/mot17-09-ocsort.events.json \
  --output demo/sample_outputs/mot17-09-ocsort.chunks.json
```

### 6b. Optional: include appearance metadata in chunk creation

If you have run `scripts/extract_appearance.py` and then `scripts/extract_events.py` on the appearance-enhanced tracks, your generated `track_facts.json` already carries the `appearance` field and can be chunked directly.

If you instead want to merge appearance from a separate appearance-enhanced tracks file, use:

```bash
PYTHONPATH=. venv/bin/python scripts/build_chunks.py \
  --config configs/rtdetr_ocsort.yaml \
  --track_facts demo/sample_outputs/mot17-09-ocsort.track_facts.json \
  --events demo/sample_outputs/mot17-09-ocsort.events.json \
  --tracks_with_appearance demo/sample_outputs/mot17-09-ocsort.built_tracks.appearance.json \
  --output demo/sample_outputs/mot17-09-ocsort.chunks.json
```

### 7. Optional: Rebuild video facts with chunk enrichment

If you skipped step 5, or want richer statistics after building chunks:

```bash
PYTHONPATH=. venv/bin/python scripts/build_video_facts.py \
  --track-facts demo/sample_outputs/mot17-09-ocsort.track_facts.json \
  --events demo/sample_outputs/mot17-09-ocsort.events.json \
  --chunks demo/sample_outputs/mot17-09-ocsort.chunks.json \
  --fps 30 \
  --output demo/sample_outputs/mot17-09-ocsort.video_facts.json
```

### 8. Build the FAISS retrieval index

```bash
PYTHONPATH=. venv/bin/python scripts/build_index.py \
  --chunks demo/sample_outputs/mot17-09-ocsort.chunks.json \
  --index-output demo/sample_outputs/mot17-09-ocsort.faiss.index \
  --metadata-output demo/sample_outputs/mot17-09-ocsort.index_meta.json \
  --model sentence-transformers/all-MiniLM-L6-v2
```

### 7. Ask a question from the command line

```bash
PYTHONPATH=. venv/bin/python scripts/query_rag.py \
  --index demo/sample_outputs/mot17-09-ocsort.faiss.index \
  --metadata demo/sample_outputs/mot17-09-ocsort.index_meta.json \
  --video-facts demo/sample_outputs/mot17-09-ocsort.video_facts.json \
  --query "Which track stayed the longest?" \
  --top-k 5 \
  --model sentence-transformers/all-MiniLM-L6-v2
```

### 8. Launch the Gradio app

```bash
PYTHONPATH=. venv/bin/python app.py
```

Then open the local Gradio URL shown in the terminal, upload a video, run the full pipeline, and ask questions in the second tab.

## Quick Start

If you want the shortest end-to-end sequence:

```bash
git clone <your-repo-url>
cd MultiObjectTracking-RTDETR-OCSORT-RAG
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p demo/sample_videos demo/sample_outputs demo/hf_runs evaluation/results models
PYTHONPATH=. venv/bin/python scripts/download_models.py
PYTHONPATH=. venv/bin/python scripts/run_demo.py \
  --config configs/rtdetr_ocsort.yaml \
  --input demo/sample_videos/mot17_09_frcnn.mp4 \
  --output demo/sample_outputs/mot17-09-ocsort.mp4
```

## MOT Evaluation Commands

Evaluate a generated MOT file against a local MOT sequence directory:

```bash
PYTHONPATH=. venv/bin/python evaluation/evaluate_mot.py \
  --pred demo/sample_outputs/mot17-09-ocsort.mot.txt \
  --sequence-dir MOT17-09-FRCNN
```

Optional example with a duration cap:

```bash
PYTHONPATH=. venv/bin/python evaluation/evaluate_mot.py \
  --pred demo/sample_outputs/mot17-09-ocsort.mot.txt \
  --sequence-dir MOT17-09-FRCNN \
  --sample-seconds 10
```

Results are saved under `evaluation/results/`.

## Testing

Run the test suite:

```bash
PYTHONPATH=. venv/bin/python -m pytest tests
```

Run tests with coverage through the `Makefile`:

```bash
make test
```

Run linting:

```bash
make lint
```

## Makefile Commands

The repository includes these convenience targets:

- `make install`
- `make dirs`
- `make run`
- `make test`
- `make lint`
- `make metrics`
- `make download-models`

Important note: `make run` currently defaults to `configs/default.yaml`, but the present code path is most reliable with `configs/rtdetr_ocsort.yaml`. If you want to use `make run`, override the config explicitly:

```bash
make run \
  CONFIG=configs/rtdetr_ocsort.yaml \
  INPUT=demo/sample_videos/mot17_09_frcnn.mp4 \
  OUTPUT=demo/sample_outputs/mot17-09-ocsort.mp4
```

## Notes About Local Assets

- Model weights are ignored by Git.
- `demo/sample_videos/` is ignored by Git.
- `demo/sample_outputs/` is ignored by Git.
- `demo/hf_runs/` is ignored by Git.
- `MOT17*` folders are ignored by Git.
- `evaluation/results/` is ignored by Git.

That means a fresh clone gives you the code, but not the large runtime assets.

## Known Caveats

- The DeepSORT config path is currently incomplete because `src/trackers/factory.py` expects `tracker.max_iou_distance`, but that key is missing from `configs/default.yaml` and `configs/ultralytics_deepsort.yaml`.
- `scripts/download_models.py` currently downloads `rtdetr-l.pt`; if you use the `rtdetr_ocsort_x.yaml` config, you need `rtdetr-x.pt` locally as well.
- The sample commands assume your local videos already exist in `demo/sample_videos/`.
- MOT evaluation requires a valid local MOT sequence directory with `gt/gt.txt` and `seqinfo.ini`.

## Verified In This Workspace

- The unit test suite passes with:

```bash
venv/bin/python -m pytest tests
```

- The RT-DETR + OC-SORT config path can be instantiated successfully.
- The older DeepSORT configs are not the best default for the current codebase without config updates.
