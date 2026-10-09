<<<<<<< HEAD
# helmetdetection
=======
# SafeGuard Vision: AI Real-Time Helmet Compliance Detection

A complete, production-grade computer vision system to detect whether a person on screen is wearing a safety helmet or not.

Supports:
1. **Local Desktop / Device Camera**: Real-time live webcam processing via **OpenCV** with futuristic HUD, dynamic bounding boxes, and audio alarms.
2. **Live Web Deployment on Vercel**: High-speed, in-browser AI inference using **ONNX Runtime Web** (60 FPS on CPU/GPU without cloud server costs) plus an optional **Vercel Serverless Python API** (`/api/detect`).
3. **Training & Export Pipeline**: Ready-to-run YOLOv8 training script (`train.py`) and ONNX exporter (`export_model.py`).

---

## 🚀 Key Features

- **Class Detection**:
  - 🟩 `With Helmet` (Safety Compliant - Neon Green HUD)
  - 🟥 `Without Helmet` (Safety Violation - Radiant Red HUD + Audio Warning)
- **Audio Alarm**: Plays immediate audible alert when a safety violation is detected.
- **Incident Logger**: Automatically captures timestamped incident snapshots of violators into an incident gallery.
- **Dual Execution Engines**:
  - **Browser AI Engine**: Runs directly in client browser via WebAssembly/WebGL with ONNX Runtime Web.
  - **Vercel Python API**: Lightweight serverless function (`/api/detect`) powered by `onnxruntime`.

---

## 📁 Project Structure

```
d:/helmet/
├── api/
│   ├── detect.py             # Vercel Python Serverless Function (/api/detect)
│   └── requirements.txt      # Lightweight dependencies for Vercel runtime
├── public/
│   ├── models/
│   │   └── helmet_model.onnx # Pre-trained ONNX model (~11.6 MB)
│   └── sample_test.jpg       # Test sample image
├── detect_webcam.py          # Local OpenCV live webcam feed detector
├── train.py                  # Model training and fine-tuning pipeline
├── export_model.py           # PyTorch to ONNX model export utility
├── index.html                # Modern web application dashboard
├── styles.css                # Cyber-industrial dark UI stylesheet
├── app.js                    # Web camera, ONNX engine & HUD rendering
├── helmet_model.pt           # Trained PyTorch YOLOv8n weights (~6.2 MB)
├── helmet_model.onnx         # Root ONNX model weights
├── requirements-local.txt    # Full dependencies for local OpenCV desktop environment
├── vercel.json               # Vercel deployment & routing configuration
└── README.md
```

---

## 🛠️ 1. Local OpenCV Live Webcam Usage

Run the real-time detector on your device's connected camera:

```bash
# Run on default webcam (Index 0)
python detect_webcam.py

# Run on external webcam
python detect_webcam.py --source 1

# Adjust confidence threshold (e.g., 0.50)
python detect_webcam.py --conf 0.50
```

### Keyboard Shortcuts in OpenCV Window:
- **`Q`** or **`ESC`**: Quit and release camera.
- **`S`**: Save a high-resolution snapshot to `snapshots/`.
- **`A`**: Toggle audio alert on/off.

---

## 🌐 2. Running the Web Application Locally

To test the web interface in your local browser:

```bash
# Start local static server
python -m http.server 3000
```

Open your browser and navigate to:
```
http://localhost:3000
```

Click **"Start Camera"** to grant browser webcam access and begin live helmet compliance scanning!

---

## ☁️ 3. Deploying Live to Vercel

### Option A: Deploy via Vercel CLI

1. Install the Vercel CLI (if not already installed):
   ```bash
   npm install -g vercel
   ```
2. Log in and deploy:
   ```bash
   vercel
   ```
3. Deploy to production:
   ```bash
   vercel --prod
   ```

### Option B: Deploy via GitHub + Vercel Dashboard

1. Push this folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Helmet Detection System"
   git remote add origin https://github.com/<your-username>/helmet-detection.git
   git push -u origin main
   ```
2. Go to [vercel.com](https://vercel.com) and click **"Add New Project"**.
3. Import your GitHub repository.
4. Framework Preset: **Other** (Root directory: `./`).
5. Click **"Deploy"**. Vercel will build both the static frontend and the Python serverless API automatically!

---

## 🧠 4. Model Training & Fine-Tuning Pipeline

If you want to fine-tune the model on custom images:

```bash
# Verify pipeline with quick demo dataset
python train.py --demo

# Train on custom Roboflow / YAML dataset for 25 epochs
python train.py --data path/to/data.yaml --epochs 25 --imgsz 224
```

To re-export to ONNX:
```bash
python export_model.py --model helmet_model.pt --imgsz 224
```
>>>>>>> d28ddf3 (Initial commit)
