from __future__ import annotations

from pathlib import Path
import shutil
from ultralytics import YOLO


MODEL_NAME = "rtdetr-l.pt"   # change to yolov9c.pt if needed


def main() -> None:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    output_path = models_dir / MODEL_NAME

    if output_path.exists():
        print(f"Model already present at: {output_path.resolve()}")
        return

    print(f"Downloading {MODEL_NAME} via Ultralytics...")

    # This downloads + loads the model
    model = YOLO(MODEL_NAME)

    # Get actual downloaded file path
    source_path = Path(model.ckpt_path)

    print(f"Downloaded to cache: {source_path}")

    # Copy to your models directory
    shutil.copy(source_path, output_path)

    print(f"Saved locally to: {output_path.resolve()}")


if __name__ == "__main__":
    main()