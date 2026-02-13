import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById("webcam");
const debugCanvas = document.getElementById("debugCanvas");
const progressText = document.getElementById("progressText");
const progressFill = document.getElementById("progressFill");
const statusText = document.getElementById("statusText");
const closeButton = document.getElementById("closeButton");
const floatingHearts = document.getElementById("floatingHearts");
const cupidArrows = document.getElementById("cupidArrows");
const sparkles = document.getElementById("sparkles");
const confettiContainer = document.getElementById("confettiContainer");
const cuteMessage = document.getElementById("cuteMessage");

const HOLD_RATE = 25;
const DECAY_RATE = 10;
const MAX_PROGRESS = 100;
const HEART_SCORE_THRESHOLD = 2.6;
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17],
];

let handLandmarker;
let progressValue = 0;
let lastFrameTime = performance.now();
let accepted = false;
let cameraStream;
let audioContext;

const debugCtx = debugCanvas.getContext("2d");

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

function scoreCloseness(distance, max) {
  return clamp(1 - distance / max, 0, 1);
}

function getHandMap(landmarks, handednesses) {
  const handMap = {};
  if (handednesses && handednesses.length === landmarks.length) {
    handednesses.forEach((handedness, index) => {
      const label = handedness[0]?.categoryName?.toLowerCase();
      if (label) {
        handMap[label] = landmarks[index];
      }
    });
  }
  if (!handMap.left || !handMap.right) {
    handMap.left = landmarks[0];
    handMap.right = landmarks[1];
  }
  return handMap;
}

function getCenter(hand) {
  const wrist = hand[0];
  const mid = hand[9];
  return { x: (wrist.x + mid.x) / 2, y: (wrist.y + mid.y) / 2 };
}

function isHeartPose(landmarks, handednesses) {
  if (!landmarks || landmarks.length < 2) return false;

  const { left, right } = getHandMap(landmarks, handednesses);
  if (!left || !right) return false;

  const leftThumb = left[4];
  const leftIndex = left[8];
  const rightThumb = right[4];
  const rightIndex = right[8];

  const leftCenter = getCenter(left);
  const rightCenter = getCenter(right);
  const centerDistance = dist(leftCenter, rightCenter);

  if (centerDistance > 0.65) return false;

  const crossThumbIndex =
    scoreCloseness(dist(leftThumb, rightIndex), 0.14) +
    scoreCloseness(dist(rightThumb, leftIndex), 0.14);

  const indexNear = scoreCloseness(dist(leftIndex, rightIndex), 0.18);
  const thumbNear = scoreCloseness(dist(leftThumb, rightThumb), 0.2);

  const centerBalance = scoreCloseness(
    Math.abs((leftCenter.x + rightCenter.x) / 2 - 0.5),
    0.18
  );

  const score = crossThumbIndex + indexNear + thumbNear + centerBalance;
  return score >= HEART_SCORE_THRESHOLD;
}

function updateProgress(isHolding, deltaSeconds) {
  if (accepted) return;

  if (isHolding) {
    progressValue += HOLD_RATE * deltaSeconds;
    // Add sparkles when holding
    if (Math.random() < 0.3) {
      createSparkle();
    }
  } else {
    progressValue -= DECAY_RATE * deltaSeconds;
  }

  progressValue = clamp(progressValue, 0, MAX_PROGRESS);

  progressFill.style.width = `${progressValue.toFixed(0)}%`;
  progressText.textContent = `HOLDING ${progressValue.toFixed(0)}%`;

  // Show cute messages at milestones
  if (progressValue >= 25 && progressValue < 30) {
    showCuteMessage("You're doing great! 💕");
  } else if (progressValue >= 50 && progressValue < 55) {
    showCuteMessage("Halfway there! ✨");
  } else if (progressValue >= 75 && progressValue < 80) {
    showCuteMessage("Almost there! 💖");
  }

  if (progressValue >= MAX_PROGRESS) {
    accepted = true;
    document.body.classList.add("accepted");
    playPewSound();
    stopCamera();
    createConfetti();
    showCuteMessage("Yay! You did it! 🎉💕");
    // Add sparkles to gallery
    setTimeout(() => {
      for (let i = 0; i < 20; i++) {
        setTimeout(() => createSparkle(), i * 100);
      }
    }, 500);
  }
}

function drawDebug(landmarks) {
  if (!debugCtx) return;
  debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
  if (!landmarks) return;
  debugCtx.save();
  debugCtx.strokeStyle = "rgba(255, 79, 172, 0.8)";
  debugCtx.lineWidth = 3;
  landmarks.forEach((hand) => {
    HAND_CONNECTIONS.forEach(([start, end]) => {
      const a = hand[start];
      const b = hand[end];
      debugCtx.beginPath();
      debugCtx.moveTo(a.x * debugCanvas.width, a.y * debugCanvas.height);
      debugCtx.lineTo(b.x * debugCanvas.width, b.y * debugCanvas.height);
      debugCtx.stroke();
    });
    hand.forEach((point) => {
      debugCtx.beginPath();
      debugCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
      debugCtx.arc(
        point.x * debugCanvas.width,
        point.y * debugCanvas.height,
        4,
        0,
        Math.PI * 2
      );
      debugCtx.fill();
    });
  });
  debugCtx.restore();
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

async function setupHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

function updateStatus(text) {
  statusText.textContent = text;
}

function startMagicalEffects() {
  // Create floating hearts periodically
  setInterval(() => {
    if (!accepted) {
      createFloatingHeart();
    }
  }, 2000);

  // Create cupid arrows periodically
  setInterval(() => {
    if (!accepted) {
      createCupidArrow();
    }
  }, 4000);

  // Create initial sparkles
  for (let i = 0; i < 10; i++) {
    setTimeout(() => createSparkle(), i * 300);
  }
}

async function init() {
  if (!navigator.mediaDevices?.getUserMedia) {
    updateStatus("Camera not supported in this browser.");
    return;
  }

  startMagicalEffects();

  try {
    updateStatus("Loading hand tracker...");
    cameraStream = await setupCamera();
    handLandmarker = await setupHandLandmarker();
    updateStatus("Make a hand heart to accept.");
    requestAnimationFrame(loop);
  } catch (error) {
    updateStatus("Unable to access camera.");
  }
}

async function resetExperience() {
  accepted = false;
  progressValue = 0;
  progressFill.style.width = "0%";
  progressText.textContent = "HOLDING 0%";
  document.body.classList.remove("accepted");
  stopCamera();
  // Clear confetti
  if (confettiContainer) {
    confettiContainer.innerHTML = "";
  }
  updateStatus("Resetting camera...");
  try {
    cameraStream = await setupCamera();
    updateStatus("Make a hand heart to accept.");
    // Add some welcome sparkles
    for (let i = 0; i < 5; i++) {
      setTimeout(() => createSparkle(), i * 200);
    }
  } catch (error) {
    updateStatus("Unable to access camera.");
  }
}

function stopCamera() {
  if (!cameraStream) return;
  cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
}

function createFloatingHeart() {
  if (!floatingHearts) return;
  const heart = document.createElement("div");
  heart.className = "floating-heart";
  heart.textContent = ["❤", "💕", "💖", "💗", "💝"][Math.floor(Math.random() * 5)];
  heart.style.left = `${Math.random() * 100}%`;
  heart.style.animationDuration = `${6 + Math.random() * 4}s`;
  heart.style.animationDelay = `${Math.random() * 2}s`;
  floatingHearts.appendChild(heart);
  setTimeout(() => heart.remove(), 10000);
}

function createCupidArrow() {
  if (!cupidArrows) return;
  const arrow = document.createElement("div");
  arrow.className = "cupid-arrow";
  arrow.textContent = "💘";
  arrow.style.top = `${20 + Math.random() * 60}%`;
  arrow.style.animationDuration = `${3 + Math.random() * 2}s`;
  arrow.style.animationDelay = `${Math.random() * 1}s`;
  cupidArrows.appendChild(arrow);
  setTimeout(() => arrow.remove(), 6000);
}

function createSparkle() {
  if (!sparkles) return;
  const sparkle = document.createElement("div");
  sparkle.className = "sparkle";
  sparkle.style.left = `${Math.random() * 100}%`;
  sparkle.style.top = `${Math.random() * 100}%`;
  sparkle.style.animationDelay = `${Math.random() * 0.5}s`;
  sparkles.appendChild(sparkle);
  setTimeout(() => sparkle.remove(), 2000);
}

function createConfetti() {
  if (!confettiContainer) return;
  for (let i = 0; i < 50; i++) {
    const conf = document.createElement("div");
    conf.className = "confetti";
    conf.style.left = `${Math.random() * 100}%`;
    conf.style.animationDelay = `${Math.random() * 0.5}s`;
    conf.style.width = `${8 + Math.random() * 8}px`;
    conf.style.height = conf.style.width;
    confettiContainer.appendChild(conf);
    setTimeout(() => conf.remove(), 4000);
  }
}

function showCuteMessage(text) {
  if (!cuteMessage) return;
  cuteMessage.textContent = text;
  cuteMessage.classList.add("show");
  setTimeout(() => {
    cuteMessage.classList.remove("show");
  }, 2000);
}

function playPewSound() {
  try {
    audioContext =
      audioContext ||
      new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === "suspended") {
      audioContext.resume();
    }

    const osc = audioContext.createOscillator();
    const gain = audioContext.createGain();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(880, audioContext.currentTime);
    osc.frequency.exponentialRampToValueAtTime(
      420,
      audioContext.currentTime + 0.2
    );
    gain.gain.setValueAtTime(0.0001, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(
      0.25,
      audioContext.currentTime + 0.02
    );
    gain.gain.exponentialRampToValueAtTime(
      0.0001,
      audioContext.currentTime + 0.26
    );
    osc.connect(gain).connect(audioContext.destination);
    osc.start();
    osc.stop(audioContext.currentTime + 0.27);
  } catch (error) {
    // Ignore audio errors to avoid blocking the UI.
  }
}

function loop(now) {
  const deltaSeconds = Math.max(0.016, (now - lastFrameTime) / 1000);
  lastFrameTime = now;

  let isHolding = false;

  if (handLandmarker && video.readyState >= 2) {
    const results = handLandmarker.detectForVideo(video, now);
    isHolding = isHeartPose(results.landmarks, results.handednesses);
    drawDebug(results.landmarks);
  }

  updateProgress(isHolding, deltaSeconds);
  requestAnimationFrame(loop);
}

init();

if (closeButton) {
  closeButton.addEventListener("click", () => {
    resetExperience();
  });
}

