/**
 * SafeGuard Vision - Helmet Compliance Web Application
 * Engine: ONNX Runtime Web + Vercel Python Serverless API Fallback
 */

// Application State
const state = {
  ortSession: null,
  isModelLoaded: false,
  isStreaming: false,
  executionEngine: 'client', // 'client' or 'server'
  confThreshold: 0.28,
  iouThreshold: 0.45,
  audioAlertEnabled: true,
  lastAudioAlertTime: 0,
  lastSnapshotTime: 0,
  currentStream: null,
  incidentLogs: [],
  fpsTracker: {
    lastTime: performance.now(),
    frames: 0,
    fps: 0,
  },
  audioCtx: null,
};

// DOM References
const DOM = {
  video: document.getElementById('webcamVideo'),
  canvas: document.getElementById('outputCanvas'),
  viewportContainer: document.getElementById('viewportContainer'),
  engineBadge: document.getElementById('engineBadge'),
  engineBadgeText: document.getElementById('engineBadgeText'),
  statusPill: document.getElementById('statusPill'),
  statusPillText: document.getElementById('statusPillText'),
  fpsDisplay: document.getElementById('fpsDisplay'),
  loadingOverlay: document.getElementById('loadingOverlay'),
  loadProgressBar: document.getElementById('loadProgressBar'),
  loaderTitle: document.getElementById('loaderTitle'),
  loaderSubtitle: document.getElementById('loaderSubtitle'),
  startCamBtn: document.getElementById('startCamBtn'),
  stopCamBtn: document.getElementById('stopCamBtn'),
  cameraSelect: document.getElementById('cameraSelect'),
  audioToggleBtn: document.getElementById('audioToggleBtn'),
  audioIconOn: document.getElementById('audioIconOn'),
  audioIconOff: document.getElementById('audioIconOff'),
  snapshotBtn: document.getElementById('snapshotBtn'),
  helmetCount: document.getElementById('helmetCount'),
  violationCount: document.getElementById('violationCount'),
  latencyDisplay: document.getElementById('latencyDisplay'),
  engineSelect: document.getElementById('engineSelect'),
  confRange: document.getElementById('confRange'),
  confValue: document.getElementById('confValue'),
  iouRange: document.getElementById('iouRange'),
  iouValue: document.getElementById('iouValue'),
  dropZone: document.getElementById('dropZone'),
  imageFileInput: document.getElementById('imageFileInput'),
  loadSampleBtn: document.getElementById('loadSampleBtn'),
  incidentList: document.getElementById('incidentList'),
  incidentTotal: document.getElementById('incidentTotal'),
  clearIncidentsBtn: document.getElementById('clearIncidentsBtn'),
  violationFlash: document.getElementById('violationFlash'),
  incidentModal: document.getElementById('incidentModal'),
  modalCloseBtn: document.getElementById('modalCloseBtn'),
  modalImg: document.getElementById('modalImg'),
  modalTitle: document.getElementById('modalTitle'),
  modalDetails: document.getElementById('modalDetails'),
  modalDownloadBtn: document.getElementById('modalDownloadBtn'),
};

// Offscreen 224x224 Canvas for Model Preprocessing
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = 224;
offscreenCanvas.height = 224;
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

/**
 * 1. Sound Synthesizer via Web Audio API
 */
function initAudio() {
  if (!state.audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      state.audioCtx = new AudioContextClass();
    }
  }
}

function playViolationAlarm() {
  if (!state.audioAlertEnabled) return;
  initAudio();
  if (!state.audioCtx) return;

  if (state.audioCtx.state === 'suspended') {
    state.audioCtx.resume();
  }

  const now = performance.now();
  if (now - state.lastAudioAlertTime < 1400) return; // Alert cooldown
  state.lastAudioAlertTime = now;

  try {
    const osc = state.audioCtx.createOscillator();
    const gain = state.audioCtx.createGain();

    osc.type = 'sawtooth';
    // Two-tone warning beep
    osc.frequency.setValueAtTime(880, state.audioCtx.currentTime); // A5
    osc.frequency.setValueAtTime(1174, state.audioCtx.currentTime + 0.12); // D6

    gain.gain.setValueAtTime(0.18, state.audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, state.audioCtx.currentTime + 0.32);

    osc.connect(gain);
    gain.connect(state.audioCtx.destination);

    osc.start();
    osc.stop(state.audioCtx.currentTime + 0.32);
  } catch (err) {
    console.warn('Audio play failed:', err);
  }
}

/**
 * 2. Model Initialization (ONNX Runtime Web)
 */
async function loadOnnxModel() {
  const modelPaths = [
    'public/models/helmet_model.onnx',
    'models/helmet_model.onnx',
    'helmet_model.onnx',
  ];

  try {
    DOM.loadProgressBar.style.width = '30%';

    // Configure ORT
    ort.env.wasm.numThreads = 2;
    ort.env.wasm.simd = true;

    let loadedSession = null;
    let successfulPath = '';

    for (const path of modelPaths) {
      try {
        DOM.loaderSubtitle.textContent = `Testing path: ${path}...`;
        const session = await ort.InferenceSession.create(path, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        if (session) {
          loadedSession = session;
          successfulPath = path;
          break;
        }
      } catch (e) {
        console.log(`Failed loading from ${path}:`, e.message);
      }
    }

    if (!loadedSession) {
      throw new Error('Could not find helmet_model.onnx in any static directory.');
    }

    state.ortSession = loadedSession;
    state.isModelLoaded = true;
    DOM.loadProgressBar.style.width = '100%';

    DOM.engineBadgeText.textContent = 'Engine: Browser AI (Active)';
    DOM.engineBadge.classList.add('ready');

    setTimeout(() => {
      DOM.loadingOverlay.classList.add('hidden');
    }, 400);

    console.log('ONNX Model Loaded from:', successfulPath);
  } catch (err) {
    console.error('Model load error:', err);
    DOM.loaderTitle.textContent = 'Model Loading Notice';
    DOM.loaderSubtitle.textContent =
      'Client ONNX model could not be loaded directly. Switching to Vercel Python API mode.';
    DOM.loadProgressBar.style.width = '100%';
    DOM.loadProgressBar.style.background = 'var(--accent-red)';
    state.executionEngine = 'server';
    DOM.engineSelect.value = 'server';
    DOM.engineBadgeText.textContent = 'Engine: Vercel Python API';

    setTimeout(() => {
      DOM.loadingOverlay.classList.add('hidden');
    }, 1500);
  }
}

/**
 * 3. Video & Camera Controls
 */
async function listCameras() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === 'videoinput');

    DOM.cameraSelect.innerHTML = '<option value="">Default Camera</option>';
    videoDevices.forEach((dev, index) => {
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      opt.textContent = dev.label || `Camera ${index + 1}`;
      DOM.cameraSelect.appendChild(opt);
    });
  } catch (e) {
    console.warn('Error listing camera devices:', e);
  }
}

function handleCameraError(err) {
  console.error('Camera access error details:', err.name, err.message);

  let message = 'Unable to access webcam.';
  if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
    message = 'Camera is currently locked or in use by another application (e.g., Python OpenCV, Zoom, Teams, or another browser tab).\n\nThe OpenCV Python script was previously holding the camera. We have released it! Please click "Start Camera" again.';
  } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
    message = 'Camera access is blocked by the browser.\n\nTo enable it:\n1. Click the site settings icon (lock/sliders) next to the URL in your address bar (http://localhost:3000).\n2. Set "Camera" to "Allow".\n3. Refresh the page and click "Start Camera".';
  } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
    message = 'No camera hardware detected on this device. Please connect a webcam or use the "Offline Image Test" feature below.';
  } else {
    message = `Camera error (${err.name || 'Unknown'}): ${err.message || 'Could not start stream'}.`;
  }

  alert(message);
  setStatus('STANDBY', 'CAMERA ACCESS ERROR');
}

async function startCamera(deviceId = null) {
  initAudio();
  if (state.isStreaming) stopCamera();

  const constraints = {
    audio: false,
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      deviceId: deviceId ? { exact: deviceId } : undefined,
      facingMode: deviceId ? undefined : 'user',
    },
  };

  let stream = null;

  try {
    // Attempt with ideal resolution
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (initialErr) {
    console.warn('Initial camera constraints failed, attempting basic fallback...', initialErr);
    try {
      // Fallback to basic unconstrained stream
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: deviceId ? { deviceId: { exact: deviceId } } : true,
      });
    } catch (fallbackErr) {
      handleCameraError(fallbackErr);
      return;
    }
  }

  try {
    state.currentStream = stream;
    DOM.video.srcObject = stream;
    await DOM.video.play();

    state.isStreaming = true;
    DOM.startCamBtn.disabled = true;
    DOM.stopCamBtn.disabled = false;
    DOM.viewportContainer.classList.add('scanning-active');

    // Sync Canvas Size
    DOM.canvas.width = DOM.video.videoWidth || 640;
    DOM.canvas.height = DOM.video.videoHeight || 480;

    setStatus('STANDBY', 'CAMERA ACTIVE: SCANNING...');
    requestAnimationFrame(renderLoop);
    listCameras();
  } catch (playErr) {
    handleCameraError(playErr);
  }
}

function stopCamera() {
  if (state.currentStream) {
    state.currentStream.getTracks().forEach((track) => track.stop());
    state.currentStream = null;
  }
  state.isStreaming = false;
  DOM.startCamBtn.disabled = false;
  DOM.stopCamBtn.disabled = true;
  DOM.viewportContainer.classList.remove('scanning-active');
  setStatus('STANDBY', 'CAMERA STANDBY');
}

/**
 * 4. Image Preprocessing for YOLOv8 (224x224 RGB Float32 Tensor)
 */
function createInputTensor(imageSource) {
  offscreenCtx.drawImage(imageSource, 0, 0, 224, 224);
  const imgData = offscreenCtx.getImageData(0, 0, 224, 224).data;

  const float32Data = new Float32Array(3 * 224 * 224);
  const channelSize = 224 * 224;

  for (let i = 0; i < channelSize; i++) {
    const r = imgData[i * 4] / 255.0;
    const g = imgData[i * 4 + 1] / 255.0;
    const b = imgData[i * 4 + 2] / 255.0;

    float32Data[i] = r;
    float32Data[channelSize + i] = g;
    float32Data[channelSize * 2 + i] = b;
  }

  return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
}

/**
 * 5. Non-Maximum Suppression (NMS) in JavaScript
 */
function nms(boxes, scores, iouThreshold) {
  const indices = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
  const picked = [];

  while (indices.length > 0) {
    const current = indices.shift();
    picked.push(current);

    const [cx1, cy1, cx2, cy2] = boxes[current];
    const area1 = Math.max(0, cx2 - cx1) * Math.max(0, cy2 - cy1);

    for (let i = indices.length - 1; i >= 0; i--) {
      const idx = indices[i];
      const [ox1, oy1, ox2, oy2] = boxes[idx];
      const area2 = Math.max(0, ox2 - ox1) * Math.max(0, oy2 - oy1);

      const xx1 = Math.max(cx1, ox1);
      const yy1 = Math.max(cy1, oy1);
      const xx2 = Math.min(cx2, ox2);
      const yy2 = Math.min(cy2, oy2);

      const interW = Math.max(0, xx2 - xx1);
      const interH = Math.max(0, yy2 - yy1);
      const interArea = interW * interH;

      const union = area1 + area2 - interArea;
      const iou = union > 0 ? interArea / union : 0;

      if (iou >= iouThreshold) {
        indices.splice(i, 1);
      }
    }
  }

  return picked;
}

/**
 * 6. Inference Execution (Dual Engine)
 */
async function runInference(sourceElement) {
  const t0 = performance.now();

  if (state.executionEngine === 'server') {
    // Mode B: Vercel Python Serverless API
    return await runServerlessInference(sourceElement, t0);
  }

  // Mode A: Client ONNX Runtime Web
  if (!state.ortSession) return { detections: [], latency: 0 };

  const inputTensor = createInputTensor(sourceElement);
  const inputName = state.ortSession.inputNames[0];
  const feeds = {};
  feeds[inputName] = inputTensor;

  const results = await state.ortSession.run(feeds);
  const outputName = state.ortSession.outputNames[0];
  const outputTensor = results[outputName];

  const data = outputTensor.data; // Float32Array shape [1, 6, 1029] -> 6174 values
  const numChannels = 6;
  const numAnchors = 1029;

  const candidateBoxes = [];
  const candidateScores = [];
  const candidateClasses = [];

  const canvasW = DOM.canvas.width;
  const canvasH = DOM.canvas.height;
  const lb = state.letterbox || {
    scale: 224 / canvasW,
    padX: 0,
    padY: 0,
  };

  for (let i = 0; i < numAnchors; i++) {
    const cx = data[0 * numAnchors + i];
    const cy = data[1 * numAnchors + i];
    const w = data[2 * numAnchors + i];
    const h = data[3 * numAnchors + i];
    const scoreWith = data[4 * numAnchors + i];
    const scoreWithout = data[5 * numAnchors + i];

    const maxScore = Math.max(scoreWith, scoreWithout);
    if (maxScore >= state.confThreshold) {
      const clsId = scoreWithout > scoreWith ? 1 : 0; // 0 = With Helmet, 1 = Without Helmet

      // Unpad coordinates from 224 letterbox space back to original canvas scale
      const origCx = (cx - lb.padX) / lb.scale;
      const origCy = (cy - lb.padY) / lb.scale;
      const origW = w / lb.scale;
      const origH = h / lb.scale;

      const x1 = Math.max(0, origCx - origW / 2);
      const y1 = Math.max(0, origCy - origH / 2);
      const x2 = Math.min(canvasW, origCx + origW / 2);
      const y2 = Math.min(canvasH, origCy + origH / 2);

      candidateBoxes.push([x1, y1, x2, y2]);
      candidateScores.push(maxScore);
      candidateClasses.push(clsId);
    }
  }

  const keepIndices = nms(candidateBoxes, candidateScores, state.iouThreshold);
  const detections = keepIndices.map((idx) => ({
    box: candidateBoxes[idx],
    score: candidateScores[idx],
    clsId: candidateClasses[idx],
    label: candidateClasses[idx] === 0 ? 'With Helmet' : 'Without Helmet',
  }));

  const latency = performance.now() - t0;
  return { detections, latency };
}

/**
 * 7. Serverless Python API Call
 */
async function runServerlessInference(sourceElement, t0) {
  try {
    // Render current frame to offscreen canvas
    offscreenCtx.drawImage(sourceElement, 0, 0, 224, 224);
    const base64Data = offscreenCanvas.toDataURL('image/jpeg', 0.85);

    const res = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64Data,
        conf: state.confThreshold,
        iou: state.iouThreshold,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();

    const canvasW = DOM.canvas.width;
    const canvasH = DOM.canvas.height;
    const scaleX = canvasW / (json.image_size ? json.image_size[0] : 224);
    const scaleY = canvasH / (json.image_size ? json.image_size[1] : 224);

    const detections = (json.detections || []).map((d) => ({
      box: [
        d.box[0] * scaleX,
        d.box[1] * scaleY,
        d.box[2] * scaleX,
        d.box[3] * scaleY,
      ],
      score: d.confidence,
      clsId: d.class_id,
      label: d.class_name,
    }));

    return { detections, latency: performance.now() - t0 };
  } catch (err) {
    console.warn('Serverless API inference error:', err.message);
    return { detections: [], latency: performance.now() - t0 };
  }
}

/**
 * 8. Drawing Futuristic HUD and Detections
 */
function drawHUD(ctx, detections, frameWidth, frameHeight) {
  ctx.clearRect(0, 0, frameWidth, frameHeight);

  // Draw camera video frame onto canvas
  if (state.isStreaming && DOM.video.readyState >= 2) {
    ctx.drawImage(DOM.video, 0, 0, frameWidth, frameHeight);
  }

  let helmetCount = 0;
  let violationCount = 0;

  detections.forEach((det) => {
    const [x1, y1, x2, y2] = det.box;
    const w = x2 - x1;
    const h = y2 - y1;
    const isSafe = det.clsId === 0;

    if (isSafe) {
      helmetCount++;
    } else {
      violationCount++;
    }

    const strokeColor = isSafe ? '#00e676' : '#ff1744';
    const fillColor = isSafe ? 'rgba(0, 230, 118, 0.15)' : 'rgba(255, 23, 68, 0.22)';
    const tagText = isSafe
      ? `HELMET OK (${Math.round(det.score * 100)}%)`
      : `! NO HELMET ! (${Math.round(det.score * 100)}%)`;

    // 1. Fill Box
    ctx.fillStyle = fillColor;
    ctx.fillRect(x1, y1, w, h);

    // 2. High-Tech Corner Brackets
    const cornerSize = Math.min(24, Math.max(10, Math.min(w, h) * 0.25));
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = isSafe ? 2.5 : 3.5;
    ctx.lineCap = 'round';

    // Top-Left
    ctx.beginPath();
    ctx.moveTo(x1, y1 + cornerSize);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1 + cornerSize, y1);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(x2 - cornerSize, y1);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x2, y1 + cornerSize);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(x1, y2 - cornerSize);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x1 + cornerSize, y2);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(x2 - cornerSize, y2);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2, y2 - cornerSize);
    ctx.stroke();

    // Subtle outline
    ctx.lineWidth = 1;
    ctx.strokeRect(x1, y1, w, h);

    // 3. Label Badge
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    const textMetrics = ctx.measureText(tagText);
    const badgeW = textMetrics.width + 16;
    const badgeH = 22;
    const badgeY = Math.max(0, y1 - badgeH - 2);

    ctx.fillStyle = strokeColor;
    ctx.fillRect(x1, badgeY, badgeW, badgeH);

    ctx.fillStyle = isSafe ? '#000' : '#fff';
    ctx.fillText(tagText, x1 + 8, badgeY + 15);
  });

  // Update Counters
  DOM.helmetCount.textContent = helmetCount;
  DOM.violationCount.textContent = violationCount;

  // Handle Violations & Status
  if (violationCount > 0) {
    setStatus('ALERT', `VIOLATION DETECTED: ${violationCount} WITHOUT HELMET`);
    DOM.violationFlash.classList.add('active');
    playViolationAlarm();
    triggerAutoSnapshot();
  } else if (helmetCount > 0) {
    setStatus('SAFE', `COMPLIANT: ${helmetCount} HELMET(S) VERIFIED`);
    DOM.violationFlash.classList.remove('active');
  } else {
    setStatus('STANDBY', 'MONITORING ACTIVE: SCANNING...');
    DOM.violationFlash.classList.remove('active');
  }
}

function setStatus(mode, text) {
  DOM.statusPill.className = 'status-pill';
  if (mode === 'SAFE') {
    DOM.statusPill.classList.add('status-safe');
  } else if (mode === 'ALERT') {
    DOM.statusPill.classList.add('status-alert');
  } else {
    DOM.statusPill.classList.add('status-standby');
  }
  DOM.statusPillText.textContent = text;
}

/**
 * 9. Real-Time Render Loop
 */
let isProcessingFrame = false;

async function renderLoop() {
  if (!state.isStreaming) return;

  if (DOM.video.readyState >= 2 && !isProcessingFrame) {
    isProcessingFrame = true;

    try {
      const { detections, latency } = await runInference(DOM.video);
      const ctx = DOM.canvas.getContext('2d');
      drawHUD(ctx, detections, DOM.canvas.width, DOM.canvas.height);

      // Latency & FPS
      DOM.latencyDisplay.textContent = `${Math.round(latency)} ms`;
      updateFPS();
    } catch (e) {
      console.warn('Frame inference error:', e);
    } finally {
      isProcessingFrame = false;
    }
  }

  requestAnimationFrame(renderLoop);
}

function updateFPS() {
  const now = performance.now();
  state.fpsTracker.frames++;
  if (now - state.fpsTracker.lastTime >= 1000) {
    state.fpsTracker.fps = (state.fpsTracker.frames * 1000) / (now - state.fpsTracker.lastTime);
    DOM.fpsDisplay.textContent = state.fpsTracker.fps.toFixed(1);
    state.fpsTracker.frames = 0;
    state.fpsTracker.lastTime = now;
  }
}

/**
 * 10. Incident Logging & Snapshots
 */
function triggerAutoSnapshot() {
  const now = performance.now();
  if (now - state.lastSnapshotTime < 6000) return; // Cooldown 6s
  state.lastSnapshotTime = now;
  captureSnapshot('Auto-Logged Safety Violation');
}

function captureSnapshot(reason = 'Manual Snapshot') {
  if (!state.isStreaming && DOM.canvas.width === 0) return;

  const dataUrl = DOM.canvas.toDataURL('image/jpeg', 0.9);
  const timeStr = new Date().toLocaleTimeString();
  const dateStr = new Date().toLocaleDateString();

  const incident = {
    id: Date.now(),
    image: dataUrl,
    reason,
    time: `${dateStr} ${timeStr}`,
  };

  state.incidentLogs.unshift(incident);
  renderIncidentList();
}

function renderIncidentList() {
  DOM.incidentTotal.textContent = state.incidentLogs.length;

  if (state.incidentLogs.length === 0) {
    DOM.incidentList.innerHTML = `
      <div class="empty-state" id="incidentEmpty">
        <span class="empty-icon">🛡️</span>
        <p>No safety violations recorded yet. Real-time violations will be automatically logged here.</p>
      </div>`;
    return;
  }

  DOM.incidentList.innerHTML = '';
  state.incidentLogs.forEach((inc) => {
    const item = document.createElement('div');
    item.className = 'incident-item';
    item.innerHTML = `
      <img src="${inc.image}" alt="Incident" class="incident-thumb">
      <div class="incident-meta">
        <div class="incident-time">${inc.time}</div>
        <div class="incident-badge">${inc.reason}</div>
      </div>
    `;
    item.addEventListener('click', () => openIncidentModal(inc));
    DOM.incidentList.appendChild(item);
  });
}

function openIncidentModal(incident) {
  DOM.modalImg.src = incident.image;
  DOM.modalTitle.textContent = incident.reason;
  DOM.modalDetails.textContent = `Timestamp: ${incident.time}`;
  DOM.modalDownloadBtn.href = incident.image;
  DOM.modalDownloadBtn.download = `incident_${incident.id}.jpg`;
  DOM.incidentModal.classList.remove('hidden');
}

/**
 * 11. Offline File Upload & Testing
 */
function handleImageUpload(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = async () => {
      // Stop camera if running
      if (state.isStreaming) stopCamera();

      DOM.canvas.width = img.naturalWidth || 640;
      DOM.canvas.height = img.naturalHeight || 480;

      const ctx = DOM.canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, DOM.canvas.width, DOM.canvas.height);

      const { detections, latency } = await runInference(img);
      drawHUD(ctx, detections, DOM.canvas.width, DOM.canvas.height);
      DOM.latencyDisplay.textContent = `${Math.round(latency)} ms`;
      DOM.fpsDisplay.textContent = '--';
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

/**
 * 12. Event Listeners Setup
 */
function setupEventListeners() {
  DOM.startCamBtn.addEventListener('click', () => {
    startCamera(DOM.cameraSelect.value || null);
  });

  DOM.stopCamBtn.addEventListener('click', stopCamera);

  DOM.cameraSelect.addEventListener('change', (e) => {
    if (state.isStreaming) {
      startCamera(e.target.value || null);
    }
  });

  DOM.audioToggleBtn.addEventListener('click', () => {
    state.audioAlertEnabled = !state.audioAlertEnabled;
    if (state.audioAlertEnabled) {
      DOM.audioIconOn.classList.remove('hidden');
      DOM.audioIconOff.classList.add('hidden');
    } else {
      DOM.audioIconOn.classList.add('hidden');
      DOM.audioIconOff.classList.remove('hidden');
    }
  });

  DOM.snapshotBtn.addEventListener('click', () => captureSnapshot('Manual Snapshot'));

  DOM.engineSelect.addEventListener('change', (e) => {
    state.executionEngine = e.target.value;
    if (state.executionEngine === 'client') {
      DOM.engineBadgeText.textContent = 'Engine: Browser AI (Active)';
    } else {
      DOM.engineBadgeText.textContent = 'Engine: Vercel Python API';
    }
  });

  DOM.confRange.addEventListener('input', (e) => {
    state.confThreshold = parseInt(e.target.value, 10) / 100.0;
    DOM.confValue.textContent = `${e.target.value}%`;
  });

  DOM.iouRange.addEventListener('input', (e) => {
    state.iouThreshold = parseInt(e.target.value, 10) / 100.0;
    DOM.iouValue.textContent = `${e.target.value}%`;
  });

  DOM.dropZone.addEventListener('click', () => DOM.imageFileInput.click());
  DOM.imageFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageUpload(e.target.files[0]);
    }
  });

  DOM.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    DOM.dropZone.style.borderColor = 'var(--accent-cyan)';
  });

  DOM.dropZone.addEventListener('dragleave', () => {
    DOM.dropZone.style.borderColor = '';
  });

  DOM.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    DOM.dropZone.style.borderColor = '';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageUpload(e.dataTransfer.files[0]);
    }
  });

  DOM.loadSampleBtn.addEventListener('click', () => {
    fetch('public/sample_test.jpg')
      .then((res) => res.blob())
      .then((blob) => handleImageUpload(blob))
      .catch((err) => console.warn('Could not load sample:', err));
  });

  DOM.clearIncidentsBtn.addEventListener('click', () => {
    state.incidentLogs = [];
    renderIncidentList();
  });

  DOM.modalCloseBtn.addEventListener('click', () => {
    DOM.incidentModal.classList.add('hidden');
  });

  DOM.incidentModal.addEventListener('click', (e) => {
    if (e.target === DOM.incidentModal) {
      DOM.incidentModal.classList.add('hidden');
    }
  });
}

// Bootstrap Application
window.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await listCameras();
  await loadOnnxModel();
});
