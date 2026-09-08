import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const MP_VERSION = "1.0.1";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" + MP_VERSION + "/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const PINCH_ON = 0.045;
const PINCH_OFF = 0.07;
/** Frames of open pinch required before dropping a held microbe. */
const RELEASE_FRAMES = 10;
/** Frames of lost tracking allowed before dropping a held microbe. */
const MISS_FRAMES = 18;
const THUMB_TIP = 4;
const INDEX_TIP = 8;

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
  Left: { pinched: false, releaseFrames: 0, missFrames: 0, lastX: 0, lastY: 0 },
  Right: { pinched: false, releaseFrames: 0, missFrames: 0, lastX: 0, lastY: 0 },
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
    x: (1 - lmX) * dw + ox,
    y: lmY * dh + oy,
  };
}

function pinchDistance(landmarks) {
  const a = landmarks[THUMB_TIP];
  const b = landmarks[INDEX_TIP];
  if (!a || !b) return 1;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function pinchMidpoint(landmarks) {
  const a = landmarks[THUMB_TIP];
  const b = landmarks[INDEX_TIP];
  return videoToScreen((a.x + b.x) / 2, (a.y + b.y) / 2);
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

  for (let i = 0; i < landmarksList.length; i++) {
    const landmarks = landmarksList[i];
    const label = handLabel(handednesses, i);
    if (seen[label]) continue;
    seen[label] = true;

    const state = handState[label];
    const slot = "hand-" + label;
    const holding = api && api.isGrabbing(slot);
    const dist = pinchDistance(landmarks);

    let justPinched = false;
    if (!state.pinched && dist < PINCH_ON) {
      state.pinched = true;
      state.releaseFrames = 0;
      justPinched = true;
    }
    if (state.pinched && dist > PINCH_OFF) {
      // While holding a microbe, require sustained open-pinch before release.
      if (holding) {
        state.releaseFrames += 1;
        if (state.releaseFrames >= RELEASE_FRAMES) {
          state.pinched = false;
          state.releaseFrames = 0;
        }
      } else {
        state.pinched = false;
        state.releaseFrames = 0;
      }
    } else {
      state.releaseFrames = 0;
    }

    state.missFrames = 0;
    const tip = videoToScreen(landmarks[INDEX_TIP].x, landmarks[INDEX_TIP].y);
    const mid = pinchMidpoint(landmarks);
    const cursor = state.pinched || holding ? mid : tip;
    state.lastX = cursor.x;
    state.lastY = cursor.y;
    setCursor(label, cursor.x, cursor.y, true, state.pinched || holding);

    const ethanol = hitTestEthanol(mid.x, mid.y);
    if (justPinched && ethanol) ethanol.click();

    if (!api) continue;

    if (state.pinched || holding) {
      if (holding) {
        api.moveGrab(slot, mid.x, mid.y);
      }
      if (state.pinched && !api.isGrabbing(slot) && !ethanol) {
        const el = api.hitTestMicrobe(mid.x, mid.y);
        if (el) api.startGrab(slot, el, mid.x, mid.y);
      }
      if (!state.pinched && holding) {
        api.endGrab(slot);
      }
    } else {
      api.endGrab(slot);
    }
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
        state.pinched = false;
        state.releaseFrames = 0;
        state.missFrames = 0;
        api.endGrab(slot);
        setCursor(label, 0, 0, false, false);
      }
      return;
    }

    state.pinched = false;
    state.releaseFrames = 0;
    state.missFrames = 0;
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

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: 1280, height: 720 },
      audio: false,
    });
    video.srcObject = stream;
    video.classList.add("active");
    await video.play();
  } catch (err) {
    video.classList.remove("active");
    setStatus("Camera blocked — use mouse or touch");
    return;
  }

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

function toggleDebug() {
  debugOn = !debugOn;
  if (!debugOn && debugCtx && debugCanvas) {
    debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
  }
}

window.startHandsCamera = startCamera;
if (window._handsCameraRequested) startCamera();
window.addEventListener("resize", resizeDebugCanvas);
window.addEventListener("keydown", function (e) {
  if (e.key === "d" || e.key === "D") toggleDebug();
});

loadLandmarker().catch(function () {});
