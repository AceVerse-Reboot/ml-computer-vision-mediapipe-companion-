import {
  FilesetResolver,
  GestureRecognizer,
  FaceLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const video = document.querySelector("#webcam");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");

const startButton = document.querySelector("#startButton");
const statusText = document.querySelector("#status");
const companionEmoji = document.querySelector("#companionEmoji");
const gestureValue = document.querySelector("#gestureValue");
const gestureConfidence = document.querySelector("#gestureConfidence");
const fingerValue = document.querySelector("#fingerValue");
const expressionValue = document.querySelector("#expressionValue");
const expressionConfidence = document.querySelector("#expressionConfidence");

let gestureRecognizer;
let faceLandmarker;
let drawingUtils;
let running = false;
let lastVideoTime = -1;

// Stabilises results so the UI does not flicker every frame.
const fingerHistory = [];
const expressionHistory = [];
const HISTORY_LENGTH = 7;

const GESTURE_LABELS = {
  Closed_Fist: "Closed fist",
  Open_Palm: "Open palm",
  Pointing_Up: "Pointing up",
  Thumb_Down: "Thumbs down",
  Thumb_Up: "Thumbs up",
  Victory: "Victory",
  ILoveYou: "I love you",
  None: "Unclear gesture"
};

const GESTURE_EMOJIS = {
  Closed_Fist: "✊",
  Open_Palm: "🖐️",
  Pointing_Up: "☝️",
  Thumb_Down: "👎",
  Thumb_Up: "👍",
  Victory: "✌️",
  ILoveYou: "🤟",
  None: "🤖"
};

async function createModels() {
  statusText.textContent = "Loading MediaPipe models…";

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task"
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
    cannedGesturesClassifierOptions: {
      scoreThreshold: 0.55
    }
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.6,
    minFacePresenceConfidence: 0.6,
    minTrackingConfidence: 0.6
  });

  drawingUtils = new DrawingUtils(ctx);
  statusText.textContent = "MediaPipe ready";
}

async function startCamera() {
  try {
    startButton.disabled = true;

    if (!gestureRecognizer || !faceLandmarker) {
      await createModels();
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    running = true;
    startButton.hidden = true;
    statusText.textContent = "Show your face and one hand";
    requestAnimationFrame(renderLoop);
  } catch (error) {
    console.error(error);
    statusText.textContent =
      "Could not start. Check camera permission and internet connection.";
    startButton.disabled = false;
  }
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));
}

function fingerIsExtended(points, tip, pip, mcp) {
  // Rotation-tolerant test: the fingertip should be farther from the wrist
  // than both the middle and base joints.
  const wrist = points[0];
  return (
    distance(points[tip], wrist) >
      distance(points[pip], wrist) * 1.12 &&
    distance(points[tip], wrist) >
      distance(points[mcp], wrist) * 1.28
  );
}

function estimateFingerCount(points, recognisedGesture) {
  // Use MediaPipe's trained gesture classes whenever they directly identify
  // a numeric hand pose.
  if (recognisedGesture === "Closed_Fist") return 0;
  if (recognisedGesture === "Pointing_Up") return 1;
  if (recognisedGesture === "Victory") return 2;
  if (recognisedGesture === "Open_Palm") return 5;

  let count = 0;

  // Thumb: compare its spread from the palm rather than x direction,
  // which makes it less sensitive to left/right hands and mirroring.
  const thumbSpread = distance(points[4], points[5]);
  const palmWidth = distance(points[5], points[17]);
  if (thumbSpread > palmWidth * 0.62) count++;

  if (fingerIsExtended(points, 8, 6, 5)) count++;
  if (fingerIsExtended(points, 12, 10, 9)) count++;
  if (fingerIsExtended(points, 16, 14, 13)) count++;
  if (fingerIsExtended(points, 20, 18, 17)) count++;

  return Math.max(0, Math.min(5, count));
}

function stableValue(history, nextValue) {
  history.push(nextValue);
  if (history.length > HISTORY_LENGTH) history.shift();

  const counts = new Map();
  for (const value of history) {
    counts.set(value, (counts.get(value) || 0) + 1);
  }

  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

function categoryMap(faceResult) {
  const categories =
    faceResult.faceBlendshapes?.[0]?.categories || [];

  return Object.fromEntries(
    categories.map(item => [item.categoryName, item.score])
  );
}

function average(...values) {
  return values.reduce((sum, value) => sum + (value || 0), 0) / values.length;
}

function analyseVisibleExpression(faceResult) {
  const b = categoryMap(faceResult);

  const smile = average(b.mouthSmileLeft, b.mouthSmileRight);
  const jawOpen = b.jawOpen || 0;
  const browUp = average(
    b.browInnerUp,
    b.browOuterUpLeft,
    b.browOuterUpRight
  );
  const blink = average(b.eyeBlinkLeft, b.eyeBlinkRight);
  const frown = average(b.mouthFrownLeft, b.mouthFrownRight);
  const browDown = average(b.browDownLeft, b.browDownRight);

  const candidates = [
    { key: "Smiling", emoji: "🙂", score: smile, threshold: 0.45 },
    { key: "Mouth open", emoji: "😮", score: jawOpen, threshold: 0.48 },
    { key: "Brows raised", emoji: "😲", score: browUp, threshold: 0.42 },
    { key: "Eyes closed", emoji: "😌", score: blink, threshold: 0.58 },
    {
      key: "Frown-like movement",
      emoji: "🙁",
      score: Math.max(frown, browDown),
      threshold: 0.48
    }
  ].sort((a, b) => b.score - a.score);

  const best = candidates[0];

  if (!best || best.score < best.threshold) {
    return { key: "Neutral-looking", emoji: "😐", score: 1 - (best?.score || 0) };
  }

  return best;
}

function processGesture(result) {
  if (!result.landmarks?.length) {
    gestureValue.textContent = "No hand detected";
    gestureConfidence.textContent = "–";
    fingerValue.textContent = "–";
    return null;
  }

  const gesture = result.gestures?.[0]?.[0];
  const gestureName = gesture?.categoryName || "None";
  const score = gesture?.score || 0;

  const estimatedCount = estimateFingerCount(
    result.landmarks[0],
    gestureName
  );

  const stableCount = stableValue(fingerHistory, estimatedCount);

  gestureValue.textContent =
    GESTURE_LABELS[gestureName] || gestureName;
  gestureConfidence.textContent =
    `${Math.round(score * 100)}% confidence`;
  fingerValue.textContent = stableCount;

  companionEmoji.textContent =
    GESTURE_EMOJIS[gestureName] || "🤖";

  return result.landmarks[0];
}

function processFace(result) {
  if (!result.faceLandmarks?.length) {
    expressionValue.textContent = "No face detected";
    expressionConfidence.textContent = "–";
    return null;
  }

  const expression = analyseVisibleExpression(result);
  const stableExpression = stableValue(
    expressionHistory,
    expression.key
  );

  expressionValue.textContent = stableExpression;
  expressionConfidence.textContent =
    `${Math.round(expression.score * 100)}% signal`;

  // Use the face emoji only when the hand is absent.
  if (gestureValue.textContent === "No hand detected") {
    companionEmoji.textContent = expression.emoji;
  }

  return result.faceLandmarks[0];
}

function drawResults(handPoints, facePoints) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (handPoints) {
    drawingUtils.drawConnectors(
      handPoints,
      GestureRecognizer.HAND_CONNECTIONS,
      { lineWidth: 4 }
    );
    drawingUtils.drawLandmarks(handPoints, { radius: 4 });
  }

  if (facePoints) {
    drawingUtils.drawConnectors(
      facePoints,
      FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
      { lineWidth: 2 }
    );
    drawingUtils.drawConnectors(
      facePoints,
      FaceLandmarker.FACE_LANDMARKS_LIPS,
      { lineWidth: 2 }
    );
    drawingUtils.drawConnectors(
      facePoints,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
      { lineWidth: 2 }
    );
    drawingUtils.drawConnectors(
      facePoints,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
      { lineWidth: 2 }
    );
  }
}

function renderLoop() {
  if (!running) return;

  if (
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    video.currentTime !== lastVideoTime
  ) {
    lastVideoTime = video.currentTime;
    const now = performance.now();

    const gestureResult =
      gestureRecognizer.recognizeForVideo(video, now);
    const faceResult =
      faceLandmarker.detectForVideo(video, now);

    const handPoints = processGesture(gestureResult);
    const facePoints = processFace(faceResult);

    drawResults(handPoints, facePoints);
  }

  requestAnimationFrame(renderLoop);
}

startButton.addEventListener("click", startCamera);
