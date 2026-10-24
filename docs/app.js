const COLORS = {
  bg: "#08090b",
  panel: "#111215",
  inner: "#0c0d10",
  border: "#2a2c31",
  text: "#ebedf1",
  muted: "#858a94",
  dim: "#454a54",
  accent: "#4abeff",
  bright: "#7fdaff",
  green: "#6fecad",
};

const drawCanvas = document.getElementById("draw-canvas");
const drawCtx = drawCanvas.getContext("2d", { willReadFrequently: true });
const previewCanvas = document.getElementById("input-preview");
const previewCtx = previewCanvas.getContext("2d");
const networkCanvas = document.getElementById("network-canvas");
const networkCtx = networkCanvas.getContext("2d");
const statusEl = document.getElementById("model-status");
const digitEl = document.getElementById("predicted-digit");
const captionEl = document.getElementById("result-caption");
const scoresEl = document.getElementById("scores");

let model = null;
let input = new Float32Array(784);
let activations = [];
let prediction = null;
let scores = new Float64Array(10);
let isDrawing = false;
let lastPoint = null;
let animationFrame = null;

function setStatus(text, ok = true) {
  statusEl.innerHTML = `<span class="status-dot"></span><span>${text}</span>`;
  statusEl.style.color = ok ? COLORS.green : "#f0a0a8";
  statusEl.querySelector(".status-dot").style.background = ok
    ? COLORS.green
    : "#f0a0a8";
}

function sigmoid(x) {
  x = Math.max(-60, Math.min(60, x));
  return 1 / (1 + Math.exp(-x));
}

async function loadModel() {
  try {
    const response = await fetch("model/weights.json", { cache: "no-store" });
    if (!response.ok) throw new Error("weights.json not found");
    const data = await response.json();
    const weights = data.weights;
    const biases = data.biases;
    if (!weights || !biases || weights.length !== 3 || biases.length !== 3) {
      throw new Error("Unexpected model format");
    }
    if (
      weights[0].length !== 784 ||
      weights[0][0].length !== 16 ||
      weights[1].length !== 16 ||
      weights[1][0].length !== 16 ||
      weights[2].length !== 16 ||
      weights[2][0].length !== 10
    ) {
      throw new Error("Model shapes do not match 784 → 16 → 16 → 10");
    }
    model = { weights, biases };
    setStatus("MODEL LOADED");
    updateFromDrawing();
  } catch (error) {
    setStatus("MODEL FILE NEEDED", false);
    captionEl.textContent = "Export model.npz to enable predictions";
    console.warn("Could not load pretrained model:", error);
  }
}

function forwardPass(values) {
  const all = [Array.from(values)];
  let current = Array.from(values);
  for (let layer = 0; layer < model.weights.length; layer++) {
    const w = model.weights[layer];
    const b = model.biases[layer];
    const next = new Array(b.length);
    for (let j = 0; j < b.length; j++) {
      let z = b[j];
      for (let i = 0; i < current.length; i++) z += current[i] * w[i][j];
      next[j] = sigmoid(z);
    }
    current = next;
    all.push(current);
  }
  return all;
}

function updatePrediction(values) {
  input = values;
  if (!model) {
    activations = [];
    drawInputPreview();
    drawNetwork();
    return;
  }
  activations = forwardPass(input);
  const outputs = activations[activations.length - 1];
  prediction = outputs.reduce(
    (best, v, i) => (v > outputs[best] ? i : best),
    0,
  );
  const total = outputs.reduce((a, b) => a + b, 0);
  scores = Float64Array.from(outputs, (v) => (total > 1e-12 ? v / total : 0.1));
  digitEl.textContent = String(prediction);
  captionEl.textContent = `${(scores[prediction] * 100).toFixed(1)}% relative score`;
  renderScores();
  drawInputPreview();
  drawNetwork();
}

function renderScores() {
  scoresEl.innerHTML = "";
  for (let i = 0; i < 10; i++) {
    const best = i === prediction;
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `
      <span class="score-digit ${best ? "best" : ""}">${i}</span>
      <div class="score-track"><div class="score-fill ${best ? "best" : ""}" style="width:${Math.max(0, Math.min(1, scores[i])) * 100}%"></div></div>
      <span class="score-value ${best ? "best" : ""}">${(scores[i] * 100).toFixed(1)}%</span>`;
    scoresEl.appendChild(row);
  }
}

function getProcessedInput() {
  const raw = drawCtx.getImageData(0, 0, 280, 280);
  const data = raw.data;
  let minX = 280,
    minY = 280,
    maxX = -1,
    maxY = -1;
  for (let y = 0; y < 280; y++) {
    for (let x = 0; x < 280; x++) {
      const k = (y * 280 + x) * 4;
      const intensity = (data[k] + data[k + 1] + data[k + 2]) / (3 * 255);
      if (intensity > 0.05) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }
  if (maxX < minX || maxY < minY) return new Float32Array(784);

  const cropW = maxX - minX + 1,
    cropH = maxY - minY + 1;
  const crop = document.createElement("canvas");
  crop.width = cropW;
  crop.height = cropH;
  crop
    .getContext("2d")
    .putImageData(drawCtx.getImageData(minX, minY, cropW, cropH), 0, 0);

  const scale = 20 / Math.max(cropW, cropH);
  const newW = Math.max(1, Math.round(cropW * scale));
  const newH = Math.max(1, Math.round(cropH * scale));
  const resized = document.createElement("canvas");
  resized.width = newW;
  resized.height = newH;
  const rctx = resized.getContext("2d", { willReadFrequently: true });
  rctx.imageSmoothingEnabled = true;
  rctx.imageSmoothingQuality = "high";
  rctx.filter = "blur(0.25px)";
  rctx.drawImage(crop, 0, 0, newW, newH);
  rctx.filter = "none";
  const pixels = rctx.getImageData(0, 0, newW, newH).data;

  const image = new Float32Array(28 * 28);
  const xOff = Math.floor((28 - newW) / 2);
  const yOff = Math.floor((28 - newH) / 2);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const k = (y * newW + x) * 4;
      image[(y + yOff) * 28 + x + xOff] = pixels[k] / 255;
    }
  }

  let total = 0,
    cx = 0,
    cy = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = image[y * 28 + x];
      total += v;
      cx += x * v;
      cy += y * v;
    }
  }
  if (total > 1e-8) {
    cx /= total;
    cy /= total;
    const shiftX = Math.round(13.5 - cx);
    const shiftY = Math.round(13.5 - cy);
    const shifted = new Float32Array(784);
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const dx = x + shiftX,
          dy = y + shiftY;
        if (dx >= 0 && dx < 28 && dy >= 0 && dy < 28) {
          shifted[dy * 28 + dx] = image[y * 28 + x];
        }
      }
    }
    return shifted;
  }
  return image;
}

function drawInputPreview() {
  const image = previewCtx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = Math.round(Math.max(0, Math.min(1, input[i])) * 255);
    const k = i * 4;
    image.data[k] = image.data[k + 1] = image.data[k + 2] = v;
    image.data[k + 3] = 255;
  }
  previewCtx.putImageData(image, 0, 0);
}

function updateFromDrawing() {
  updatePrediction(getProcessedInput());
}

function clearDrawing() {
  drawCtx.fillStyle = "#000";
  drawCtx.fillRect(0, 0, 280, 280);
  input = new Float32Array(784);
  activations = [];
  prediction = null;
  scores = new Float64Array(10);
  digitEl.textContent = "—";
  captionEl.textContent = "Draw a digit to begin";
  renderScores();
  drawInputPreview();
  drawNetwork();
}
function pointerPosition(event) {
  const r = drawCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - r.left) * 280) / r.width,
    y: ((event.clientY - r.top) * 280) / r.height,
  };
}
function startDraw(event) {
  isDrawing = true;
  drawCanvas.setPointerCapture?.(event.pointerId);
  lastPoint = pointerPosition(event);
  drawStroke(lastPoint, lastPoint);
  updateFromDrawing();
}
function moveDraw(event) {
  if (!isDrawing) return;
  const point = pointerPosition(event);
  drawStroke(lastPoint || point, point);
  lastPoint = point;
  updateFromDrawing();
}
function stopDraw() {
  isDrawing = false;
  lastPoint = null;
}
function drawStroke(a, b) {
  drawCtx.strokeStyle = "#fff";
  drawCtx.fillStyle = "#fff";
  drawCtx.lineWidth = 20;
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.beginPath();
  drawCtx.moveTo(a.x, a.y);
  drawCtx.lineTo(b.x, b.y);
  drawCtx.stroke();
  drawCtx.beginPath();
  drawCtx.arc(b.x, b.y, 10, 0, Math.PI * 2);
  drawCtx.fill();
}

function fitNetworkCanvas() {
  const rect = networkCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(rect.width * dpr));
  const h = Math.max(1, Math.floor(rect.height * dpr));
  if (networkCanvas.width !== w || networkCanvas.height !== h) {
    networkCanvas.width = w;
    networkCanvas.height = h;
  }
  networkCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawNetwork();
}
function mix(a, b, t) {
  const ah = a.replace("#", ""),
    bh = b.replace("#", "");
  const ar = parseInt(ah.slice(0, 2), 16),
    ag = parseInt(ah.slice(2, 4), 16),
    ab = parseInt(ah.slice(4, 6), 16);
  const br = parseInt(bh.slice(0, 2), 16),
    bg = parseInt(bh.slice(2, 4), 16),
    bb = parseInt(bh.slice(4, 6), 16);
  t = Math.max(0, Math.min(1, t));
  return `rgb(${Math.round(ar + (br - ar) * t)},${Math.round(ag + (bg - ag) * t)},${Math.round(ab + (bb - ab) * t)})`;
}
function drawGlow(ctx, x, y, r, color, strength) {
  if (strength < 0.12) return;
  ctx.save();
  ctx.globalAlpha = Math.min(0.6, strength * 0.45);
  ctx.shadowColor = color;
  ctx.shadowBlur = 10 + strength * 10;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r + 1, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}
function drawNetwork() {
  const rect = networkCanvas.getBoundingClientRect();
  const w = rect.width,
    h = rect.height;
  if (!w || !h) return;
  const ctx = networkCtx;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const left = 18,
    right = w - 18,
    top = 12,
    bottom = h - 12;
  const xs = [
    left,
    left + (right - left) * 0.31,
    left + (right - left) * 0.66,
    right,
  ];
  const counts = [28, 16, 16, 10];
  const sampleIndices = Array.from({ length: 28 }, (_, i) =>
    Math.floor((i * 783) / 27),
  );
  const vals =
    activations.length === 4
      ? [
          sampleIndices.map((i) => input[i]),
          activations[1],
          activations[2],
          activations[3],
        ]
      : [
          sampleIndices.map((i) => input[i]),
          Array(16).fill(0),
          Array(16).fill(0),
          Array(10).fill(0),
        ];
  const positions = counts.map((count, layer) => {
    const pad = Math.max(10, (bottom - top) * 0.025);
    return Array.from({ length: count }, (_, i) => ({
      x: xs[layer],
      y:
        count === 1
          ? (top + bottom) / 2
          : top + pad + (i * (bottom - top - 2 * pad)) / (count - 1),
    }));
  });

  // Connections use the exact weights associated with sampled input pixels.
  if (model) {
    for (let l = 0; l < 3; l++) {
      for (let i = 0; i < counts[l]; i++) {
        const sourceIdx = l === 0 ? sampleIndices[i] : i;
        const sv = vals[l][i] || 0;
        for (let j = 0; j < counts[l + 1]; j++) {
          const weight = model.weights[l][sourceIdx][j];
          const tv = vals[l + 1][j] || 0;
          const brightness =
            Math.min(Math.abs(weight) / 2, 1) * (0.15 + 0.45 * sv + 0.4 * tv);
          const base = weight >= 0 ? "#1b232c" : "#261d23";
          const lit = weight >= 0 ? "#46aee1" : "#d66a7d";
          ctx.strokeStyle = mix(base, lit, brightness);
          ctx.lineWidth = brightness > 0.72 ? 1.8 : 0.8;
          ctx.beginPath();
          ctx.moveTo(positions[l][i].x, positions[l][i].y);
          ctx.lineTo(positions[l + 1][j].x, positions[l + 1][j].y);
          ctx.stroke();
        }
      }
    }
  } else {
    // Subtle inactive connections preserve the visual layout until weights are exported.
    for (let l = 0; l < 3; l++)
      for (const a of positions[l])
        for (const b of positions[l + 1]) {
          ctx.strokeStyle = "#1a2027";
          ctx.lineWidth = 0.7;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
  }

  for (let l = 0; l < counts.length; l++) {
    for (let i = 0; i < counts[l]; i++) {
      const p = positions[l][i];
      const value = Math.max(0, Math.min(1, vals[l][i] || 0));
      const radius = l === 0 ? 3.6 : l === 3 ? 7 : 5.8;
      const color = l === 3 && prediction === i ? COLORS.green : COLORS.bright;
      drawGlow(ctx, p.x, p.y, radius, color, value);
      ctx.fillStyle = mix("#17191e", COLORS.accent, value * 0.8);
      ctx.strokeStyle = mix("#3f434c", color, value);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
}

drawCtx.fillStyle = "#000";
drawCtx.fillRect(0, 0, 280, 280);
renderScores();
drawInputPreview();
drawNetwork();
loadModel();

drawCanvas.addEventListener("pointerdown", startDraw);
drawCanvas.addEventListener("pointermove", moveDraw);
window.addEventListener("pointerup", stopDraw);
window.addEventListener("pointercancel", stopDraw);
document.getElementById("clear-button").addEventListener("click", clearDrawing);
document
  .getElementById("predict-button")
  .addEventListener("click", updateFromDrawing);
window.addEventListener("keydown", (event) => {
  if (
    event.key.toLowerCase() === "c" &&
    !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)
  )
    clearDrawing();
  if (event.key === "Enter") updateFromDrawing();
});
window.addEventListener("resize", () =>
  requestAnimationFrame(fitNetworkCanvas),
);
if ("ResizeObserver" in window)
  new ResizeObserver(fitNetworkCanvas).observe(networkCanvas);
requestAnimationFrame(fitNetworkCanvas);
