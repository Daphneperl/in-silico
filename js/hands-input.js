import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const MP_VERSION = "1.0.1";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" + MP_VERSION + "/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

/** Drop a second detection if it sits this close (px) to another hand. */
const GHOST_PX = 72;
/** Frames of lost tracking allowed before dropping a held microbe. */
const MISS_FRAMES = 18;
/** Frames over the ethanol bottle before it activates. */
const ETHANOL_DWELL = 18;
const THUMB_TIP = 4;
const INDEX_TIP = 8;
const PALM_IDS = [0, 5, 9, 13, 17];

const HAND_LABELS = ["Left", "Right"];

const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];

/** @type {import("@mediapipe/tasks-vision").HandLandmarker | null} */
let handLandmarker = null;
let landmarkerPromise = null;
let video = null;
let debugCanvas = null;
let debugCtx = null;
let statusEl = null;
let running = false;
let debugOn =
  typeof location !== "undefined" &&
  new URLSearchParams(location.search).has("debug");
let detectEveryOther = false;
let frameCount = 0;

const handState = {
  Left: {
    missFrames: 0,
    ethanolFrames: 0,
    lastX: 0,
    lastY: 0,
    hasSmooth: false,
    smoothX: 0,
    smoothY: 0,
  },
  Right: {
    missFrames: 0,
    ethanolFrames: 0,
    lastX: 0,
    lastY: 0,
    hasSmooth: false,
    smoothX: 0,
    smoothY: 0,
  },
};

function setStatus(message) {
  if (!statusEl) statusEl = document.getElementById("camera-status");
  if (statusEl) statusEl.textContent = message || "";
}

function pointer() {
  return window.synthPointer || null;
}

function loadLandmarker() {
  if (landmarkerPromise) return landmarkerPromise;
  landmarkerPromise = (async function () {
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
    return handLandmarker;
  })().catch(function (err) {
    landmarkerPromise = null;
    throw err;
  });
  return landmarkerPromise;
}

function videoToScreen(lmX, lmY) {
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const sw = window.innerWidth;
  const sh = window.innerHeight;
  const scale = Math.max(sw / vw, sh / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const ox = (sw - dw) / 2;
  const oy = (sh - dh) / 2;
  return {
    x: lmX * dw + ox,
    y: lmY * dh + oy,
  };
}

function palmPoint(landmarks) {
  let x = 0;
  let y = 0;
  let n = 0;
  for (let i = 0; i < PALM_IDS.length; i++) {
    const lm = landmarks[PALM_IDS[i]];
    if (!lm) continue;
    x += lm.x;
    y += lm.y;
    n += 1;
  }
  if (!n) return videoToScreen(0.5, 0.5);
  return videoToScreen(x / n, y / n);
}

function handScore(handednesses, index) {
  const group = handednesses && handednesses[index];
  const cat = group && (group[0] || (group.categories && group.categories[0]));
  const score = cat && (cat.score != null ? cat.score : cat.categoryScore);
  return typeof score === "number" ? score : 0;
}

function smoothCursor(state, x, y) {
  if (!state.hasSmooth) {
    state.smoothX = x;
    state.smoothY = y;
    state.hasSmooth = true;
    return { x: x, y: y };
  }
  const dist = Math.hypot(x - state.smoothX, y - state.smoothY);
  const alpha = dist > 80 ? 0.65 : dist > 20 ? 0.4 : 0.18;
  state.smoothX += (x - state.smoothX) * alpha;
  state.smoothY += (y - state.smoothY) * alpha;
  return { x: state.smoothX, y: state.smoothY };
}

const ETHANOL_HIT_PAD = 20;

function hitTestEthanol(x, y) {
  const btn = document.getElementById("ethanol-button");
  if (!btn || !btn.classList.contains("show")) return null;
  const r = btn.getBoundingClientRect();
  if (
    x >= r.left - ETHANOL_HIT_PAD &&
    x <= r.right + ETHANOL_HIT_PAD &&
    y >= r.top - ETHANOL_HIT_PAD &&
    y <= r.bottom + ETHANOL_HIT_PAD
  ) {
    return btn;
  }
  return null;
}

function setCursor(label, x, y, visible, pinched) {
  const el = document.querySelector(
    '.hand-cursor[data-hand="' + label + '"]'
  );
  if (!el) return;
  if (!visible) {
    el.classList.remove("visible", "pinched");
    return;
  }
  el.style.left = x + "px";
  el.style.top = y + "px";
  el.classList.add("visible");
  el.classList.toggle("pinched", !!pinched);
}

function resizeDebugCanvas() {
  if (!debugCanvas) return;
  debugCanvas.width = window.innerWidth;
  debugCanvas.height = window.innerHeight;
}

function drawDebug(hands) {
  if (!debugCtx || !debugCanvas) return;
  debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
  if (!debugOn) return;

  hands.forEach(function (landmarks) {
    debugCtx.strokeStyle = "rgba(255,255,255,0.45)";
    debugCtx.lineWidth = 2;
    HAND_CONNECTIONS.forEach(function (pair) {
      const a = videoToScreen(landmarks[pair[0]].x, landmarks[pair[0]].y);
      const b = videoToScreen(landmarks[pair[1]].x, landmarks[pair[1]].y);
      debugCtx.beginPath();
      debugCtx.moveTo(a.x, a.y);
      debugCtx.lineTo(b.x, b.y);
      debugCtx.stroke();
    });
    landmarks.forEach(function (lm, idx) {
      const p = videoToScreen(lm.x, lm.y);
      debugCtx.fillStyle =
        idx === THUMB_TIP || idx === INDEX_TIP
          ? "rgba(255,255,255,0.95)"
          : "rgba(255,255,255,0.55)";
      debugCtx.beginPath();
      debugCtx.arc(p.x, p.y, idx === THUMB_TIP || idx === INDEX_TIP ? 6 : 3, 0, Math.PI * 2);
      debugCtx.fill();
    });
  });
}

function handLabel(handednesses, index) {
  const group = handednesses && handednesses[index];
  const cat = group && (group[0] || group.categories && group.categories[0]);
  const name = cat && (cat.categoryName || cat.displayName);
  if (name === "Left" || name === "Right") return name;
  return index === 0 ? "Left" : "Right";
}

function processHands(result) {
  const api = pointer();
  const landmarksList = (result && result.landmarks) || [];
  const handednesses = (result && result.handednesses) || result.handedness || [];
  const seen = { Left: false, Right: false };

  const candidates = [];
  for (let i = 0; i < landmarksList.length; i++) {
    const landmarks = landmarksList[i];
    candidates.push({
      landmarks: landmarks,
      label: handLabel(handednesses, i),
      score: handScore(handednesses, i),
      palm: palmPoint(landmarks),
    });
  }
  candidates.sort(function (a, b) {
    return b.score - a.score;
  });
  const kept = [];
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    let ghost = false;
    for (let k = 0; k < kept.length; k++) {
      if (Math.hypot(c.palm.x - kept[k].palm.x, c.palm.y - kept[k].palm.y) < GHOST_PX) {
        ghost = true;
        break;
      }
    }
    if (!ghost) kept.push(c);
  }

  for (let i = 0; i < kept.length; i++) {
    const label = kept[i].label;
    if (seen[label]) continue;
    seen[label] = true;

    const state = handState[label];
    const slot = "hand-" + label;
    const holding = api && api.isGrabbing(slot);
    const cursor = smoothCursor(state, kept[i].palm.x, kept[i].palm.y);
    state.missFrames = 0;
    state.lastX = cursor.x;
    state.lastY = cursor.y;
    setCursor(label, cursor.x, cursor.y, true, holding);

    if (!api) continue;

    if (holding) {
      state.ethanolFrames = 0;
      api.moveGrab(slot, cursor.x, cursor.y);
      if (api.isInCircle && api.isInCircle(cursor.x, cursor.y)) {
        if (api.depositGrab) api.depositGrab(slot, cursor.x, cursor.y);
        else api.endGrab(slot);
      }
      continue;
    }

    const ethanol = hitTestEthanol(cursor.x, cursor.y);
    if (ethanol) {
      state.ethanolFrames += 1;
      if (state.ethanolFrames >= ETHANOL_DWELL) {
        ethanol.click();
        state.ethanolFrames = 0;
      }
      continue;
    }

    state.ethanolFrames = 0;
    const el = api.hitTestMicrobe(cursor.x, cursor.y);
    if (el) api.startGrab(slot, el, cursor.x, cursor.y);
  }

  HAND_LABELS.forEach(function (label) {
    if (seen[label]) return;
    const state = handState[label];
    const slot = "hand-" + label;
    const holding = api && api.isGrabbing(slot);

    if (holding) {
      state.missFrames += 1;
      setCursor(label, state.lastX, state.lastY, true, true);
      if (state.missFrames >= MISS_FRAMES) {
        state.missFrames = 0;
        state.ethanolFrames = 0;
        state.hasSmooth = false;
        api.endGrab(slot);
        setCursor(label, 0, 0, false, false);
      }
      return;
    }

    state.missFrames = 0;
    state.ethanolFrames = 0;
    state.hasSmooth = false;
    setCursor(label, 0, 0, false, false);
    if (api) api.endGrab(slot);
  });

  drawDebug(landmarksList);
}

function detectLoop() {
  if (!running) return;
  requestAnimationFrame(detectLoop);
  if (!handLandmarker || !video || video.readyState < 2) return;

  frameCount += 1;
  if (detectEveryOther && frameCount % 2 === 1) {
    return;
  }

  const t0 = performance.now();
  let result;
  try {
    result = handLandmarker.detectForVideo(video, t0);
  } catch (err) {
    return;
  }
  const elapsed = performance.now() - t0;
  if (elapsed > 32) detectEveryOther = true;
  else if (elapsed < 20) detectEveryOther = false;

  processHands(result);
}

async function startCamera() {
  if (running) return;
  video = document.getElementById("webcam");
  debugCanvas = document.getElementById("hands-debug");
  statusEl = document.getElementById("camera-status");
  if (!video) return;

  if (debugCanvas) {
    debugCtx = debugCanvas.getContext("2d");
    resizeDebugCanvas();
  }

  setStatus("Starting camera…");
  const started = await openCameraStream(savedDeviceId());
  if (!started) return;

  try {
    await loadLandmarker();
  } catch (err) {
    setStatus("Hand tracking failed to load — use mouse or touch");
    return;
  }

  setStatus("");
  running = true;
  detectLoop();
}

function savedDeviceId() {
  if (window.selectedWebcamDevice) return window.selectedWebcamDevice;
  try {
    const parsed = JSON.parse(localStorage.getItem("synthLayout") || "{}");
    return parsed.webcamDevice || "";
  } catch (err) {
    return "";
  }
}

function videoConstraints(deviceId) {
  const constraints = {
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 24, max: 30 },
  };
  if (deviceId) constraints.deviceId = { exact: deviceId };
  else constraints.facingMode = "user";
  return constraints;
}

function stopCameraStream() {
  if (!video || !video.srcObject) return;
  video.srcObject.getTracks().forEach(function (track) {
    track.stop();
  });
  video.srcObject = null;
}

async function openCameraStream(deviceId) {
  if (!video) video = document.getElementById("webcam");
  if (!video) return false;
  stopCameraStream();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints(deviceId),
      audio: false,
    });
    video.srcObject = stream;
    video.classList.add("active");
    await video.play();
    setStatus("");
    return true;
  } catch (err) {
    video.classList.remove("active");
    setStatus("Camera blocked — use mouse or touch");
    return false;
  }
}

async function setHandsCamera(deviceId) {
  window.selectedWebcamDevice = deviceId || "";
  const started = await openCameraStream(deviceId);
  if (!started) return false;
  if (!running) {
    try {
      await loadLandmarker();
    } catch (err) {
      setStatus("Hand tracking failed to load — use mouse or touch");
      return false;
    }
    running = true;
    detectLoop();
  }
  return true;
}

function toggleDebug() {
  debugOn = !debugOn;
  if (!debugOn && debugCtx && debugCanvas) {
    debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
  }
}

window.startHandsCamera = startCamera;
window.setHandsCamera = setHandsCamera;
if (window._handsCameraRequested) startCamera();
window.addEventListener("resize", resizeDebugCanvas);
window.addEventListener("keydown", function (e) {
  if (e.key === "d" || e.key === "D") toggleDebug();
});
