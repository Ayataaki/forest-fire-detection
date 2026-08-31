import os
import cv2
import numpy as np
from pathlib import Path

TARGET_SIZE = (224, 224)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32) / 255.0
    return img

def process_all():
    for raw_dir in ["data/raw/wildfire", "data/raw/wildfire2", "data/raw/fire_rgb"]:
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            continue
        for class_dir in raw_path.iterdir():
            if not class_dir.is_dir():
                continue
            out_dir = Path("data/processed") / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_file in list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")):
                img = preprocess_image(str(img_file))
                if img is not None:
                    cv2.imwrite(str(out_dir / img_file.name), (img * 255).astype(np.uint8))
    print("Preprocessing done!")

if __name__ == "__main__":
    process_all()
