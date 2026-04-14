from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


YOLOV9_C_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt"


def main() -> None:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / "yolov9-c.pt"

    if output_path.exists():
        print(f"YOLOv9 weights already present at: {output_path.resolve()}")
        return

    print(f"Downloading YOLOv9 weights to: {output_path.resolve()}")
    urlretrieve(YOLOV9_C_URL, output_path)
    print("Download complete.")


if __name__ == "__main__":
    main()
