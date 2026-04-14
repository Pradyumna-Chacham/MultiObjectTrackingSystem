# Multi-Object Tracking with YOLOv9 and DeepSORT

This project is a modular multi-object tracking pipeline for videos and MOT-style benchmark sequences. It combines an Ultralytics YOLO detector with a DeepSORT tracker, draws annotated output videos, exports track results in MOTChallenge text format, and provides a small evaluation utility for computing standard MOT metrics such as MOTA, MOTP, and IDF1.

At a high level, the workflow is:

1. Load a YAML configuration from `configs/`.
2. Run object detection on each video frame.
3. Associate detections across frames with DeepSORT.
4. Draw IDs, boxes, and short track trails on the output video.
5. Save tracking results as:
   - an annotated video
   - a `.tracks.json` file
   - a `.mot.txt` file in MOTChallenge-compatible prediction format
6. Optionally evaluate the `.mot.txt` predictions against MOT ground truth.

The codebase is organized so that detection, tracking, captioning, export, and evaluation are separated into small modules under `src/` and `evaluation/`.

## Project Overview

### What the project currently does

- Runs video-based multi-object tracking using YOLO + DeepSORT
- Supports configurable model/device settings through YAML files
- Produces annotated `.mp4` output videos
- Produces `.tracks.json` files containing exported tracking data
- Produces `.mot.txt` files for benchmark evaluation
- Evaluates predictions against MOT ground truth using `motmetrics`

### Main implementation flow

- `scripts/run_demo.py`
  - command-line entry point for running tracking on a video
- `src/config.py`
  - loads YAML config and resolves device selection (`cpu`, `cuda`, `mps`, or `auto`)
- `src/detectors/`
  - detector adapters and detector factory
- `src/trackers/`
  - DeepSORT tracker wrapper and tracker factory
- `src/pipeline/orchestrator.py`
  - coordinates reading frames, detection, tracking, annotation, video writing, and export
- `src/io/mot_exporter.py`
  - writes MOTChallenge-format prediction files
- `evaluation/evaluate_mot.py`
  - computes MOTA, MOTP, IDF1, ID switches, misses, false positives, and related metrics

## Repository Structure

```text
.
├── configs/                 # YAML configuration files
├── demo/
│   ├── sample_videos/       # Input videos for demo runs
│   └── sample_outputs/      # Generated videos, MOT text files, and JSON exports
├── evaluation/
│   ├── evaluate_mot.py      # Metrics script
│   └── results/             # Saved evaluation reports
├── models/                  # YOLO weights
├── scripts/
│   ├── download_models.py   # Downloads/saves YOLO weights
│   └── run_demo.py          # Main tracking runner
├── src/
│   ├── detectors/           # Detection adapters
│   ├── trackers/            # Tracking adapters
│   ├── io/                  # Video and MOT export utilities
│   ├── pipeline/            # End-to-end orchestration
│   ├── captioning/          # Template captioning scaffolding
│   └── utils/               # Bounding-box helpers
├── tests/                   # Unit tests
├── MOT17-04-FRCNN/          # Local/manual dataset folder placeholder
├── MOT17-09-FRCNN/          # MOT sequence folder
├── MOT17-11-FRCNN/          # MOT sequence folder
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Important Folders You Should Have

Before running the project, make sure these folders exist:

```bash
mkdir -p demo/sample_videos
mkdir -p demo/sample_outputs
mkdir -p evaluation/results
mkdir -p models
```

If you use the provided `Makefile`, you can create them with:

```bash
make dirs
```

### What each runtime folder is used for

- `demo/sample_videos/`
  - put input videos here
- `demo/sample_outputs/`
  - output annotated videos, `.mot.txt`, and `.tracks.json` files are written here
- `evaluation/results/`
  - metric reports are saved here
- `models/`
  - YOLO model weights are stored here

## About the `MOT17-04-FRCNN`, `MOT17-09-FRCNN`, and `MOT17-11-FRCNN` Folders

These folders are MOTChallenge-style sequence directories placed directly in the workspace.

### Current state in this workspace

- `MOT17-09-FRCNN/` contains:
  - `img1/`
  - `gt/gt.txt`
  - `det/det.txt`
  - `seqinfo.ini`
- `MOT17-11-FRCNN/` contains:
  - `img1/`
  - `gt/gt.txt`
  - `det/det.txt`
  - `seqinfo.ini`
- `MOT17-04-FRCNN/` currently appears to be a manually added placeholder folder and does not yet contain the standard MOT sequence contents.

### Guidance for other users

If someone clones this repository, they should not expect these MOT folders to be downloaded automatically. The `.gitignore` is configured to ignore large datasets, images, model weights, videos, and generated outputs, so those assets are expected to be added locally.

To use a MOT sequence folder with the evaluation script, the folder should follow this structure:

```text
MOT17-XX-FRCNN/
├── det/
│   └── det.txt
├── gt/
│   └── gt.txt
├── img1/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── seqinfo.ini
```

If you want `MOT17-04-FRCNN/` to be usable, populate it with the same structure as the other MOT sequence folders, especially:

- `MOT17-04-FRCNN/img1/`
- `MOT17-04-FRCNN/gt/gt.txt`
- `MOT17-04-FRCNN/det/det.txt`
- `MOT17-04-FRCNN/seqinfo.ini`

Without `gt/gt.txt` and `seqinfo.ini`, you cannot run benchmark evaluation on that sequence.

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd MultiObjectTracking-Yolov9
```

### 2. Install the project

The repo uses a virtual environment in `venv/`.

```bash
make install
```

What `make install` does:

- creates `venv/`
- upgrades `pip`
- installs dependencies from `requirements.txt`

If you prefer to run the same steps manually:

```bash
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
```

### 3. Create required folders

```bash
make dirs
```

Or manually:

```bash
mkdir -p demo/sample_videos demo/sample_outputs evaluation/results models
```

### 4. Download model weights

```bash
make download-models
```

This downloads or saves the YOLOv9 weights into:

```text
models/yolov9c.pt
```

## Running the Tracker

### Quick run with the default Make command

```bash
make run
```

This uses:

- config: `configs/default.yaml`
- input video: `demo/sample_videos/sample.mp4`
- output video: `demo/sample_outputs/output.mp4`

### Run with your own paths

You can override the paths through `make` variables:

```bash
make run \
  CONFIG=configs/default.yaml \
  INPUT=demo/sample_videos/mot17_09_frcnn.mp4 \
  OUTPUT=demo/sample_outputs/mot17_09_frcnn.mp4
```

### Run directly with Python

```bash
PYTHONPATH=. venv/bin/python scripts/run_demo.py \
  --config configs/default.yaml \
  --input demo/sample_videos/mot17_09_frcnn.mp4 \
  --output demo/sample_outputs/mot17_09_frcnn.mp4
```

### What gets generated after a run

If your output is:

```text
demo/sample_outputs/mot17_09_frcnn.mp4
```

the pipeline will also generate:

```text
demo/sample_outputs/mot17_09_frcnn.tracks.json
demo/sample_outputs/mot17_09_frcnn.mot.txt
```

These extra files are created automatically by `src/pipeline/orchestrator.py`.

## Example Tracking Commands

### Run on MOT17-09 sample video

```bash
make run \
  CONFIG=configs/default.yaml \
  INPUT=demo/sample_videos/mot17_09_frcnn.mp4 \
  OUTPUT=demo/sample_outputs/mot17_09_frcnn.mp4
```

### Run on MOT17-11 sample video

```bash
make run \
  CONFIG=configs/default.yaml \
  INPUT=demo/sample_videos/mot17_11_frcnn.mp4 \
  OUTPUT=demo/sample_outputs/mot17_11_frcnn.mp4
```

### Run on MOT17-04 after you populate the folder and create a video

```bash
make run \
  CONFIG=configs/default.yaml \
  INPUT=demo/sample_videos/mot17_04.mp4 \
  OUTPUT=demo/sample_outputs/mot17_04.mp4
```

## Checking Tracking Metrics

The evaluation script compares a prediction file in MOT format against a MOT sequence directory containing `gt/gt.txt` and `seqinfo.ini`.

### Generic metrics command

```bash
make metrics \
  PRED=demo/sample_outputs/mot17_09_frcnn.mot.txt \
  SEQ=MOT17-09-FRCNN
```

### Direct Python command

```bash
PYTHONPATH=. venv/bin/python evaluation/evaluate_mot.py \
  --pred demo/sample_outputs/mot17_09_frcnn.mot.txt \
  --sequence-dir MOT17-09-FRCNN
```

### Evaluate only a limited duration

```bash
PYTHONPATH=. venv/bin/python evaluation/evaluate_mot.py \
  --pred demo/sample_outputs/mot17_09_frcnn.mot.txt \
  --sequence-dir MOT17-09-FRCNN \
  --sample-seconds 10
```

### Start evaluation from a later frame

```bash
PYTHONPATH=. venv/bin/python evaluation/evaluate_mot.py \
  --pred demo/sample_outputs/mot17_09_frcnn.mot.txt \
  --sequence-dir MOT17-09-FRCNN \
  --start-frame 100
```

### Example metrics commands for the included local sequences

```bash
make metrics \
  PRED=demo/sample_outputs/mot17_09_frcnn.mot.txt \
  SEQ=MOT17-09-FRCNN
```

```bash
make metrics \
  PRED=demo/sample_outputs/mot17_11_frcnn.mot.txt \
  SEQ=MOT17-11-FRCNN
```

For `MOT17-04-FRCNN`, the same command will work only after that folder has been populated correctly:

```bash
make metrics \
  PRED=demo/sample_outputs/mot17_04.mot.txt \
  SEQ=MOT17-04-FRCNN
```

### Where evaluation output is saved

The evaluation script prints metrics to the terminal and also writes reports into:

```text
evaluation/results/
```

Typical outputs include:

- `evaluation/results/<prediction_name>_metrics.txt`
- `evaluation/results/metrics_summary.csv`

## Makefile Commands

### Main commands

```bash
make install
make dirs
make download-models
make run
make test
make lint
make metrics PRED=demo/sample_outputs/output.mot.txt SEQ=MOT17-09-FRCNN
```

### Notes on the Makefile

- `make install`
  - creates the virtual environment and installs dependencies
- `make dirs`
  - creates runtime directories used by the pipeline
- `make run`
  - runs tracking with configurable `CONFIG`, `INPUT`, and `OUTPUT`
- `make test`
  - runs unit tests
- `make lint`
  - runs `ruff` on source, tests, and scripts
- `make metrics`
  - evaluates a generated MOT prediction file against a MOT sequence folder

## Configuration Files

### `configs/default.yaml`

This is the default runtime configuration and includes:

- automatic device selection
- Ultralytics detector
- DeepSORT tracker
- output settings for JSON and MOT export
- visualization settings such as trail length and labels

### `configs/ultralytics_deepsort.yaml`

This is another detector/tracker configuration with different thresholds and class settings.

## Typical End-to-End Workflow

```bash
git clone <your-repo-url>
cd MultiObjectTracking-Yolov9
make install
make dirs
make download-models
make run \
  INPUT=demo/sample_videos/mot17_09_frcnn.mp4 \
  OUTPUT=demo/sample_outputs/mot17_09_frcnn.mp4
make metrics \
  PRED=demo/sample_outputs/mot17_09_frcnn.mot.txt \
  SEQ=MOT17-09-FRCNN
```

## Tests

Run:

```bash
make test
```

Current tests cover:

- schema dataclasses
- template captioning fallback behavior
- bounding-box conversion utilities
- MOT export formatting

## Important Notes for Contributors and Users

- Large assets are intentionally not tracked in Git.
- Dataset folders, model weights, sample videos, and generated outputs are expected to be added locally.
- If you share this project with another user, make sure they know they must:
  - create the runtime folders
  - place videos into `demo/sample_videos/`
  - download or copy weights into `models/`
  - locally add any MOT17 sequence folders they want to evaluate
- `MOT17-04-FRCNN/` in this workspace should currently be treated as a local placeholder unless it is populated with a valid MOT structure.

## Troubleshooting

### `make install` fails

Make sure you are using Python 3.10 or newer:

```bash
python3 --version
```

### `make run` fails because input video is missing

Make sure the input file exists, for example:

```bash
ls demo/sample_videos
```

### `make metrics` fails

Check all of the following:

- the prediction `.mot.txt` file exists
- the sequence directory exists
- the sequence directory contains `gt/gt.txt`
- the sequence directory contains `seqinfo.ini`

### `MOT17-04-FRCNN` does not evaluate

That is expected until the folder is populated with:

- `img1/`
- `gt/gt.txt`
- `det/det.txt`
- `seqinfo.ini`

## Summary

This repository provides a practical local workflow for:

- running multi-object tracking on videos
- exporting MOT-format tracking predictions
- evaluating those predictions against MOTChallenge-style sequence folders

For most users, the core commands to remember are:

```bash
make install
make dirs
make download-models
make run
make metrics PRED=demo/sample_outputs/output.mot.txt SEQ=MOT17-09-FRCNN
```
