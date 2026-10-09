import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const MP_VERSION = "1.0.1";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" + MP_VERSION + "/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

/** Drop a second detection if it sits this close (px) to another wrist. */
const GHOST_PX = 72;
/** Max px a wrist may jump and still keep the same track. */
const MATCH_PX = 280;
/** Frames of lost tracking allowed before dropping a held microbe. */
const MISS_FRAMES = 18;
/** Frames over the ethanol bottle before it activates. */
const ETHANOL_DWELL = 18;
const VISIBILITY_MIN = 0.4;
const FIST_EXTEND = 0.2;
const LEFT_ELBOW = 13;
const RIGHT_ELBOW = 14;
const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;
const TRACK_IDS = ["0", "1", "2", "3"];

/** @type {import("@mediapipe/tasks-vision").PoseLandmarker | null} */
let poseLandmarker = null;
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

function emptyTrackState() {
  return {
    missFrames: 0,
    ethanolFrames: 0,
    lastX: 0,
    lastY: 0,
    hasSmooth: false,
    smoothX: 0,
    smoothY: 0,
  };
}

const handState = {
  0: emptyTrackState(),
  1: emptyTrackState(),
  2: emptyTrackState(),
  3: emptyTrackState(),
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
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 2,
    });
    return poseLandmarker;
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

function landmarkVisible(lm) {
  if (!lm) return false;
  const vis = lm.visibility != null ? lm.visibility : 1;
  return vis >= VISIBILITY_MIN;
}

function fistFromArm(elbow, wrist) {
  const dx = wrist.x - elbow.x;
  const dy = wrist.y - elbow.y;
  return videoToScreen(
    wrist.x + dx * FIST_EXTEND,
    wrist.y + dy * FIST_EXTEND
  );
}

function collectWrists(landmarksList) {
  const detections = [];
  for (let i = 0; i < landmarksList.length; i++) {
    const lm = landmarksList[i];
    if (!lm) continue;
    const pairs = [
      [lm[LEFT_ELBOW], lm[LEFT_WRIST]],
      [lm[RIGHT_ELBOW], lm[RIGHT_WRIST]],
    ];
    for (let p = 0; p < pairs.length; p++) {
      const elbow = pairs[p][0];
      const wrist = pairs[p][1];
      if (!landmarkVisible(wrist)) continue;
      const fist = landmarkVisible(elbow)
        ? fistFromArm(elbow, wrist)
        : videoToScreen(wrist.x, wrist.y);
      detections.push({
        fist: fist,
        elbow: elbow,
        wrist: wrist,
      });
    }
  }
  return detections;
}

function dedupeWrists(detections) {
  const kept = [];
  for (let i = 0; i < detections.length; i++) {
    const c = detections[i];
    let ghost = false;
    for (let k = 0; k < kept.length; k++) {
      if (
        Math.hypot(
          c.fist.x - kept[k].fist.x,
          c.fist.y - kept[k].fist.y
        ) < GHOST_PX
      ) {
        ghost = true;
        break;
      }
    }
    if (!ghost) kept.push(c);
  }
  return kept;
}

function matchTracks(detections) {
  const assigned = {};
  const used = {};
  TRACK_IDS.forEach(function (id) {
    assigned[id] = null;
  });

  const scored = [];
  for (let i = 0; i < detections.length; i++) {
    for (let t = 0; t < TRACK_IDS.length; t++) {
      const id = TRACK_IDS[t];
      const state = handState[id];
      if (!state.hasSmooth && state.missFrames === 0 && state.lastX === 0 && state.lastY === 0) {
        continue;
      }
      const dist = Math.hypot(
        detections[i].fist.x - state.lastX,
        detections[i].fist.y - state.lastY
      );
      if (dist > MATCH_PX) continue;
      scored.push({ i: i, id: id, dist: dist });
    }
  }
  scored.sort(function (a, b) {
    return a.dist - b.dist;
  });
  for (let s = 0; s < scored.length; s++) {
    const pair = scored[s];
    if (used[pair.i] || assigned[pair.id]) continue;
    used[pair.i] = true;
    assigned[pair.id] = detections[pair.i];
  }

  for (let i = 0; i < detections.length; i++) {
    if (used[i]) continue;
    for (let t = 0; t < TRACK_IDS.length; t++) {
      const id = TRACK_IDS[t];
      if (assigned[id]) continue;
      assigned[id] = detections[i];
      used[i] = true;
      break;
    }
  }
  return assigned;
}

function smoothCursor(state, x, y) {
  if (!state.hasSmooth) {
    state.smoothX = x;
    state.smoothY = y;
    state.hasSmooth = true;
    return { x: x, y: y };
  }
  const dist = Math.hypot(x - state.smoothX, y - state.smoothY);
  if (dist < 6) return { x: state.smoothX, y: state.smoothY };
  const alpha = dist > 140 ? 0.42 : dist > 50 ? 0.18 : 0.07;
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

function drawDebug(detections) {
  if (!debugCtx || !debugCanvas) return;
  debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
  if (!debugOn) return;

  detections.forEach(function (det) {
    const wrist = videoToScreen(det.wrist.x, det.wrist.y);
    debugCtx.strokeStyle = "rgba(255,255,255,0.45)";
    debugCtx.lineWidth = 2;
    debugCtx.beginPath();
    if (landmarkVisible(det.elbow)) {
      const elbow = videoToScreen(det.elbow.x, det.elbow.y);
      debugCtx.moveTo(elbow.x, elbow.y);
      debugCtx.lineTo(wrist.x, wrist.y);
    } else {
      debugCtx.moveTo(wrist.x, wrist.y);
    }
    debugCtx.lineTo(det.fist.x, det.fist.y);
    debugCtx.stroke();
    debugCtx.fillStyle = "rgba(255,255,255,0.95)";
    debugCtx.beginPath();
    debugCtx.arc(det.fist.x, det.fist.y, 6, 0, Math.PI * 2);
    debugCtx.fill();
  });
}

function processPose(result) {
  const api = pointer();
  const landmarksList = (result && result.landmarks) || [];
  const detections = dedupeWrists(collectWrists(landmarksList));
  const assigned = matchTracks(detections);

  TRACK_IDS.forEach(function (id) {
    const det = assigned[id];
    const state = handState[id];
    const slot = "hand-" + id;

    if (det) {
      const holding = api && api.isGrabbing(slot);
      const cursor = smoothCursor(state, det.fist.x, det.fist.y);
      state.missFrames = 0;
      state.lastX = cursor.x;
      state.lastY = cursor.y;
      setCursor(id, cursor.x, cursor.y, true, holding);

      if (!api) return;

      if (holding) {
        state.ethanolFrames = 0;
        api.moveGrab(slot, cursor.x, cursor.y);
        if (api.isInCircle && api.isInCircle(cursor.x, cursor.y)) {
          if (api.depositGrab) api.depositGrab(slot, cursor.x, cursor.y);
          else api.endGrab(slot);
        }
        return;
      }

      const ethanol = hitTestEthanol(cursor.x, cursor.y);
      if (ethanol) {
        state.ethanolFrames += 1;
        if (state.ethanolFrames >= ETHANOL_DWELL) {
          ethanol.click();
          state.ethanolFrames = 0;
        }
        return;
      }

      state.ethanolFrames = 0;
      const el = api.hitTestMicrobe(cursor.x, cursor.y);
      if (el) api.startGrab(slot, el, cursor.x, cursor.y);
      return;
    }

    const holding = api && api.isGrabbing(slot);
    if (holding) {
      state.missFrames += 1;
      setCursor(id, state.lastX, state.lastY, true, true);
      if (state.missFrames >= MISS_FRAMES) {
        state.missFrames = 0;
        state.ethanolFrames = 0;
        state.hasSmooth = false;
        api.endGrab(slot);
        setCursor(id, 0, 0, false, false);
      }
      return;
    }

    state.missFrames = 0;
    state.ethanolFrames = 0;
    state.hasSmooth = false;
    state.lastX = 0;
    state.lastY = 0;
    setCursor(id, 0, 0, false, false);
    if (api) api.endGrab(slot);
  });

  drawDebug(detections);
}

function detectLoop() {
  if (!running) return;
  requestAnimationFrame(detectLoop);
  if (!poseLandmarker || !video || video.readyState < 2) return;

  frameCount += 1;
  if (detectEveryOther && frameCount % 2 === 1) {
    return;
  }

  const t0 = performance.now();
  let result;
  try {
    result = poseLandmarker.detectForVideo(video, t0);
  } catch (err) {
    return;
  }
  const elapsed = performance.now() - t0;
  if (elapsed > 32) detectEveryOther = true;
  else if (elapsed < 20) detectEveryOther = false;

  processPose(result);
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
    setStatus("Tracking failed to load — use mouse or touch");
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
      setStatus("Tracking failed to load — use mouse or touch");
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
