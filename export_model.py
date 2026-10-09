"""
Helmet Detection Model Exporter
Author: Antigravity AI
Description: Export PyTorch YOLO (.pt) models to optimized ONNX format for Web & Edge deployment.
"""

import os
import sys
import argparse

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed. Please run: pip install ultralytics")
    sys.exit(1)


def export_onnx(model_path="helmet_model.pt", imgsz=224, opset=17, simplify=True):
    """Export model to ONNX."""
    if not os.path.exists(model_path):
        print(f"[ERROR] {model_path} not found.")
        return False

    print("=" * 60)
    print(f"Exporting '{model_path}' -> ONNX (imgsz={imgsz}, opset={opset}, simplify={simplify})")
    print("=" * 60)

    model = YOLO(model_path)
    exported_file = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
    )
    print(f"\n[SUCCESS] Model successfully exported: {exported_file}")
    return exported_file


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--model", type=str, default="helmet_model.pt", help="Path to input .pt weights")
    parser.add_argument("--imgsz", type=int, default=224, help="Image resolution for inference (default: 224)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    export_onnx(model_path=args.model, imgsz=args.imgsz, opset=args.opset)


if __name__ == "__main__":
    main()
