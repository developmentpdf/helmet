"""
Vercel Serverless Function: Helmet Detection API
Endpoint: POST /api/detect
"""

import os
import io
import json
import base64
import time
from http.server import BaseHTTPRequestHandler
import numpy as np
from PIL import Image

# Global cache for ONNX inference session across warm invocations
_SESSION = None

def get_session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort

        # Look for model in standard locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "public", "models", "helmet_model.onnx"),
            os.path.join(os.path.dirname(__file__), "..", "helmet_model.onnx"),
            os.path.join(os.path.dirname(__file__), "helmet_model.onnx"),
            "helmet_model.onnx",
        ]
        model_path = None
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break

        if not model_path:
            raise FileNotFoundError(f"helmet_model.onnx not found. Checked: {possible_paths}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSION = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    return _SESSION


def nms(boxes, scores, iou_thresh=0.45):
    """Simple Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def process_image(image_bytes, conf_thresh=0.45, iou_thresh=0.45):
    """Run ONNX model inference on input image bytes."""
    t0 = time.time()
    session = get_session()

    # Open image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    # Resize to 224x224
    img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0  # (224, 224, 3)
    img_np = np.transpose(img_np, (2, 0, 1))  # (3, 224, 224)
    img_tensor = np.expand_dims(img_np, axis=0)  # (1, 3, 224, 224)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_tensor})
    raw_out = outputs[0][0]  # shape (6, 1029)

    # raw_out format:
    # row 0: cx
    # row 1: cy
    # row 2: w
    # row 3: h
    # row 4: score0 (With Helmet)
    # row 5: score1 (Without Helmet)

    cx = raw_out[0]
    cy = raw_out[1]
    w = raw_out[2]
    h = raw_out[3]
    score_with = raw_out[4]
    score_without = raw_out[5]

    max_scores = np.maximum(score_with, score_without)
    cls_ids = np.where(score_without > score_with, 1, 0)

    # Filter by confidence
    mask = max_scores >= conf_thresh
    if not np.any(mask):
        return {
            "detections": [],
            "helmet_count": 0,
            "violation_count": 0,
            "image_size": [orig_w, orig_h],
            "inference_time_ms": round((time.time() - t0) * 1000, 2),
        }

    valid_cx = cx[mask]
    valid_cy = cy[mask]
    valid_w = w[mask]
    valid_h = h[mask]
    valid_scores = max_scores[mask]
    valid_cls = cls_ids[mask]

    # Convert to x1, y1, x2, y2 in original pixel coordinates
    scale_x = orig_w / 224.0
    scale_y = orig_h / 224.0

    x1 = (valid_cx - valid_w / 2.0) * scale_x
    y1 = (valid_cy - valid_h / 2.0) * scale_y
    x2 = (valid_cx + valid_w / 2.0) * scale_x
    y2 = (valid_cy + valid_h / 2.0) * scale_y

    # Clip to image bounds
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Non-maximum suppression
    keep_indices = nms(boxes, valid_scores, iou_thresh)

    classes_map = {0: "With Helmet", 1: "Without Helmet"}
    detections = []
    helmet_count = 0
    violation_count = 0

    for idx in keep_indices:
        c_id = int(valid_cls[idx])
        c_name = classes_map[c_id]
        score_val = float(valid_scores[idx])
        box_coords = [round(float(v), 2) for v in boxes[idx]]

        if c_id == 1:
            violation_count += 1
        else:
            helmet_count += 1

        detections.append({
            "box": box_coords,  # [x1, y1, x2, y2]
            "class_id": c_id,
            "class_name": c_name,
            "confidence": round(score_val, 4),
        })

    return {
        "detections": detections,
        "helmet_count": helmet_count,
        "violation_count": violation_count,
        "image_size": [orig_w, orig_h],
        "inference_time_ms": round((time.time() - t0) * 1000, 2),
    }


class handler(BaseHTTPRequestHandler):
    """Vercel Python serverless HTTP request handler."""

    def _set_cors_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_cors_headers(204)

    def do_GET(self):
        # If request is for root or index.html, serve index.html
        clean_path = self.path.split("?")[0]
        if clean_path in ["", "/", "/index.html"]:
            possible_html_paths = [
                os.path.join(os.path.dirname(__file__), "..", "index.html"),
                os.path.join(os.path.dirname(__file__), "index.html"),
                "index.html",
            ]
            for p in possible_html_paths:
                if os.path.exists(p):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    with open(p, "rb") as f:
                        self.wfile.write(f.read())
                    return

        # Default API health check JSON
        self._set_cors_headers(200)
        res = {
            "status": "online",
            "service": "Helmet Compliance Detection API",
            "classes": ["With Helmet", "Without Helmet"],
            "model_format": "ONNX",
            "framework": "YOLOv8",
        }
        self.wfile.write(json.dumps(res).encode("utf-8"))

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))

            image_data = payload.get("image", "")
            conf = float(payload.get("conf", 0.45))
            iou = float(payload.get("iou", 0.45))

            if not image_data:
                self._set_cors_headers(400)
                self.wfile.write(json.dumps({"error": "No image data provided"}).encode("utf-8"))
                return

            # Extract base64 payload
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]

            image_bytes = base64.b64decode(image_data)
            results = process_image(image_bytes, conf_thresh=conf, iou_thresh=iou)

            self._set_cors_headers(200)
            self.wfile.write(json.dumps(results).encode("utf-8"))

        except Exception as e:
            self._set_cors_headers(500)
            err_res = {"error": str(e)}
            self.wfile.write(json.dumps(err_res).encode("utf-8"))
