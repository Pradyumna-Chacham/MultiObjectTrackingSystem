from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "yolov9c.pt"

    if model_path.exists():
        print(f"YOLOv9c weights already present at: {model_path.resolve()}")
        return

    print("Downloading YOLOv9c weights via Ultralytics...")

    # This triggers auto-download from Ultralytics
    model = YOLO("yolov9c.pt")

    # Save weights locally in your models directory
    model.export(format="torchscript")  # optional (forces load)
    model.model.save(str(model_path))   # ensure saved locally

    print(f"Download complete. Saved to: {model_path.resolve()}")


if __name__ == "__main__":
    main()