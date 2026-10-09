"""
Helmet Detection Model Training & Fine-Tuning Pipeline
Author: Antigravity AI
Description: Train or fine-tune YOLOv8 models for helmet vs no-helmet detection.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed. Please install it with: pip install ultralytics")
    sys.exit(1)


DEFAULT_DATA_YAML = """# Dataset configuration for Helmet Detection
path: ./dataset # dataset root dir
train: images/train
val: images/val
test: images/test # optional

# Classes
names:
  0: With Helmet
  1: Without Helmet
"""


def create_demo_dataset():
    """Create a minimal runnable dataset structure to verify training pipeline."""
    import numpy as np
    import cv2

    base_dir = Path("dataset")
    for split in ["train", "val"]:
        (base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("[INFO] Creating demo dataset structure in ./dataset ...")
    yaml_path = base_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(DEFAULT_DATA_YAML)

    # Generate sample placeholder images with annotations if directory is empty
    train_imgs = list((base_dir / "images" / "train").glob("*.jpg"))
    if not train_imgs:
        print("[INFO] Generating sample training images for pipeline verification...")
        for i in range(10):
            # Synthetic image
            img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            # Draw fake person/helmet shapes
            cv2.circle(img, (320, 240), 60, (0, 255, 0) if i % 2 == 0 else (0, 0, 255), -1)
            img_path = base_dir / "images" / "train" / f"sample_{i}.jpg"
            lbl_path = base_dir / "labels" / "train" / f"sample_{i}.txt"
            cv2.imwrite(str(img_path), img)

            # YOLO annotation: class_id x_center y_center width height (normalized)
            cls_id = 0 if i % 2 == 0 else 1
            with open(lbl_path, "w") as f:
                f.write(f"{cls_id} 0.5 0.5 0.2 0.25\n")

        # Copy 2 images for val
        for i in range(2):
            img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            img_path = base_dir / "images" / "val" / f"val_{i}.jpg"
            lbl_path = base_dir / "labels" / "val" / f"val_{i}.txt"
            cv2.imwrite(str(img_path), img)
            with open(lbl_path, "w") as f:
                f.write(f"{i % 2} 0.5 0.5 0.2 0.25\n")

        print("[SUCCESS] Demo dataset created in ./dataset.")
    return str(yaml_path)


def train_model(
    data_yaml="dataset/data.yaml",
    base_model="yolov8n.pt",
    epochs=25,
    imgsz=224,
    batch_size=16,
    project_name="helmet_training",
    export_onnx=True,
):
    """Fine-tune YOLOv8 on helmet dataset."""
    print("=" * 60)
    print("        HELMET DETECTION MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Base Model:       {base_model}")
    print(f"Data YAML:        {data_yaml}")
    print(f"Epochs:           {epochs}")
    print(f"Image Size:       {imgsz}x{imgsz}")
    print(f"Batch Size:       {batch_size}")
    print("=" * 60)

    if not os.path.exists(data_yaml):
        print(f"[WARNING] {data_yaml} not found.")
        print("[INFO] Creating demo dataset so training can be tested...")
        data_yaml = create_demo_dataset()

    # Load base YOLO model
    print(f"\n[1/3] Loading base architecture: {base_model}...")
    model = YOLO(base_model)

    # Train model
    print(f"\n[2/3] Starting training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project_name,
        name="experiment_1",
        exist_ok=True,
        verbose=True,
    )

    best_pt = os.path.join(project_name, "experiment_1", "weights", "best.pt")
    if os.path.exists(best_pt):
        print(f"\n[SUCCESS] Training finished! Best model weights saved at: {best_pt}")
        
        # Copy to root as helmet_model.pt
        import shutil
        shutil.copy(best_pt, "helmet_model.pt")
        print("[INFO] Updated root 'helmet_model.pt' with newly trained weights.")

        # Export to ONNX
        if export_onnx:
            print("\n[3/3] Exporting trained model to ONNX for Web & Vercel deployment...")
            trained_model = YOLO("helmet_model.pt")
            trained_model.export(format="onnx", imgsz=imgsz)
            print("[SUCCESS] Exported 'helmet_model.onnx' successfully.")
    else:
        print("[WARNING] Could not locate best.pt. Check training output above.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Helmet Detection Model")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Base model (yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Image resolution (224 or 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data and test training")
    args = parser.parse_args()

    if args.demo:
        data_file = create_demo_dataset()
        train_model(data_yaml=data_file, base_model=args.base_model, epochs=2, imgsz=args.imgsz, batch_size=4)
    else:
        train_model(
            data_yaml=args.data,
            base_model=args.base_model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch,
        )


if __name__ == "__main__":
    main()
