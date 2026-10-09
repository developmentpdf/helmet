"""
Helmet Detection System - Live OpenCV Webcam Feed
Author: Antigravity AI
Description: Real-time helmet compliance detection using YOLOv8 & OpenCV.
"""

import sys
import os
import time
import threading
import argparse
from datetime import datetime
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed. Please run: pip install ultralytics")
    sys.exit(1)

# Platform-specific audio alert
def play_alert_sound():
    """Play alert sound in a background thread to prevent frame drops."""
    def _beep():
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1200, 200)  # 1200 Hz for 200ms
            else:
                # Terminal bell on Unix/Mac
                print('\a', end='', flush=True)
        except Exception:
            pass

    threading.Thread(target=_beep, daemon=True).start()


def draw_hud(frame, helmet_count, no_helmet_count, fps, alert_active, audio_enabled):
    """Render futuristic safety HUD and compliance banner."""
    h, w, _ = frame.shape

    # 1. Top Status Banner
    banner_height = 50
    overlay = frame.copy()

    if no_helmet_count > 0:
        # Red Alert Banner
        banner_color = (25, 25, 200)  # BGR Red
        status_text = f"SAFETY VIOLATION: {no_helmet_count} PERSON(S) WITHOUT HELMET!"
        status_icon = "[!]"
    elif helmet_count > 0:
        # Green Safe Banner
        banner_color = (35, 160, 45)  # BGR Green
        status_text = f"ALL CLEAR: {helmet_count} PERSON(S) WEARING HELMET"
        status_icon = "[OK]"
    else:
        # Blue/Gray Standby Banner
        banner_color = (70, 70, 70)  # BGR Dark Gray
        status_text = "MONITORING ACTIVE: SCANNING FOR OCCUPANTS..."
        status_icon = "[*]"

    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)
    # Blend banner for subtle transparency
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Status Banner Text
    cv2.putText(
        frame,
        f"{status_icon} {status_text}",
        (20, 33),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 2. Bottom Stats Panel
    bottom_height = 42
    bottom_y = h - bottom_height
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, bottom_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay_bot, 0.85, frame, 0.15, 0, frame)

    # Stats: FPS, Helmets, Violations, Audio Status
    audio_str = "ON" if audio_enabled else "OFF"
    stats_str = (
        f"FPS: {fps:.1f}  |  "
        f"With Helmet: {helmet_count}  |  "
        f"Without Helmet: {no_helmet_count}  |  "
        f"Sound [A]: {audio_str}  |  "
        f"Snapshot [S]  |  Quit [Q]"
    )
    cv2.putText(
        frame,
        stats_str,
        (15, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # 3. Flashing Border if violation
    if alert_active and (int(time.time() * 4) % 2 == 0):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)


def draw_detection(frame, box, class_name, confidence):
    """Draw stylish bounding box with label badge."""
    x1, y1, x2, y2 = map(int, box)

    # Color configuration (BGR)
    is_safe = ("with" in class_name.lower()) and ("without" not in class_name.lower())

    if is_safe:
        color = (0, 220, 70)      # Neon Emerald Green
        label_text = f"HELMET OK ({confidence:.1%})"
        box_thickness = 2
    else:
        color = (20, 20, 240)     # Neon Alert Red
        label_text = f"NO HELMET! ({confidence:.1%})"
        box_thickness = 3

    # Draw corner brackets for tech aesthetic
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, box_thickness + 1)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, box_thickness + 1)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, box_thickness + 1)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, box_thickness + 1)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, box_thickness + 1)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, box_thickness + 1)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, box_thickness + 1)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, box_thickness + 1)

    # Main rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Label Badge Background
    (font_w, font_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    badge_y1 = max(0, y1 - font_h - 10)
    badge_y2 = y1
    badge_x2 = min(frame.shape[1], x1 + font_w + 14)

    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
    cv2.putText(
        frame,
        label_text,
        (x1 + 6, y1 - 6),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (0, 0, 0) if is_safe else (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run_detector(
    model_path="helmet_model.pt",
    source=0,
    conf_threshold=0.45,
    iou_threshold=0.45,
    enable_sound=True,
    save_violations=True,
):
    """Run real-time helmet detection on webcam stream."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        print("Please check the model path or run python train.py first.")
        return

    print("=" * 60)
    print("  HELMET DETECTION SYSTEM (OPENCV LIVE CAMERA)")
    print("=" * 60)
    print(f"Loading YOLO Model: {model_path} ...")
    model = YOLO(model_path)
    print(f"Model Classes: {model.names}")
    print(f"Connecting to video source: {source} ...")

    # Create snapshots directory
    snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    # Open Video Source
    # On Windows, cv2.CAP_DSHOW can start webcam faster
    cap = None
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source {source}.")
        print("Tip: If using external webcam, try running with --source 1 or --source 2.")
        return

    # Set camera resolution (720p if supported)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[SUCCESS] Webcam active! Press 'q' to quit, 's' to snapshot, 'a' to toggle audio.\n")

    # Metrics trackers
    prev_time = time.time()
    fps_smooth = 30.0
    last_beep_time = 0
    last_violation_snapshot_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received from camera stream. Retrying...")
                time.sleep(0.05)
                continue

            current_time = time.time()
            fps_instant = 1.0 / max(1e-5, (current_time - prev_time))
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant
            prev_time = current_time

            # Run YOLO inference
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )

            helmet_count = 0
            no_helmet_count = 0

            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy()

                    class_name = model.names.get(cls_id, f"Class {cls_id}")

                    if "without" in class_name.lower():
                        no_helmet_count += 1
                    else:
                        helmet_count += 1

                    draw_detection(frame, xyxy, class_name, conf)

            # Trigger alert sound on violation
            alert_active = (no_helmet_count > 0)
            if alert_active and enable_sound:
                if current_time - last_beep_time > 1.2:  # Beep cooldown
                    play_alert_sound()
                    last_beep_time = current_time

            # Auto-save violation snapshot (cooldown 5s)
            if alert_active and save_violations:
                if current_time - last_violation_snapshot_time > 5.0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    violation_path = os.path.join(snapshot_dir, f"violation_{timestamp}.jpg")
                    cv2.imwrite(violation_path, frame)
                    print(f"[ALERT] Violation detected! Snapshot saved: {violation_path}")
                    last_violation_snapshot_time = current_time

            # Draw HUD
            draw_hud(frame, helmet_count, no_helmet_count, fps_smooth, alert_active, enable_sound)

            # Display window
            cv2.imshow("Helmet Compliance Detector - OpenCV Live Feed", frame)

            # Key Controls
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:  # 27 = ESC
                print("[INFO] Exiting detection session...")
                break
            elif key in [ord('s'), ord('S')]:
                # Manual snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(snapshot_dir, f"snapshot_{timestamp}.jpg")
                cv2.imwrite(snap_path, frame)
                print(f"[INFO] Manual snapshot saved to: {snap_path}")
            elif key in [ord('a'), ord('A')]:
                enable_sound = not enable_sound
                print(f"[INFO] Audio alerts: {'ENABLED' if enable_sound else 'MUTED'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows closed. Done.")


def main():
    parser = argparse.ArgumentParser(description="Live OpenCV Helmet Detection")
    parser.add_argument("--model", type=str, default="helmet_model.pt", help="Path to YOLO model .pt")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1) or path to video file")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (0.0 - 1.0)")
    parser.add_argument("--no-sound", action="store_true", help="Disable violation beep sound")
    parser.add_argument("--no-save", action="store_true", help="Disable auto-saving violation snapshots")
    args = parser.parse_args()

    # Convert source to int if numeric
    source = int(args.source) if args.source.isdigit() else args.source

    run_detector(
        model_path=args.model,
        source=source,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        enable_sound=not args.no_sound,
        save_violations=not args.no_save,
    )


if __name__ == "__main__":
    main()
