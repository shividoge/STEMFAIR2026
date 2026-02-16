/* Lettuce Yield Predictor (STEM-fair robust version)
   - Video: playback sampling (no repeated seeking -> no freezes)
   - Segmentation: Excess Green Index (ExG) -> works under blue grow lights
   - Prediction:
       * Area (cm^2) = last_frame_area_px * (cm^2/pixel)
         - auto-calibrates cm^2/pixel when you enter an Actual final area
         - stores calibration in browser localStorage
       * Mass (g) = alpha * predicted_area_cm^2
         - alpha (g/cm^2) learned from training CSV (median mass/area)
   - Training CSV requirements (minimum):
       trial_id, final_area_cm2, final_mass_g
     (Other columns allowed but not required)
*/

const els = {
  videoInput: document.getElementById("videoInput"),
  trainCsvInput: document.getElementById("trainCsvInput"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  resetBtn: document.getElementById("resetBtn"),
  fitBtn: document.getElementById("fitBtn"),
  exportFeaturesBtn: document.getElementById("exportFeaturesBtn"),
  predictBtn: document.getElementById("predictBtn"),

  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),

  framesSampled: document.getElementById("framesSampled"),
  meanAreaPx: document.getElementById("meanAreaPx"),
  maxAreaPx: document.getElementById("maxAreaPx"),
  slopePxPerS: document.getElementById("slopePxPerS"),

  trialId: document.getElementById("trialId"),
  actualArea: document.getElementById("actualArea"),
  actualMass: document.getElementById("actualMass"),

  predArea: document.getElementById("predArea"),
  predMass: document.getElementById("predMass"),

  areaAbsErr: document.getElementById("areaAbsErr"),
  areaMape: document.getElementById("areaMape"),
  massAbsErr: document.getElementById("massAbsErr"),
  massMape: document.getElementById("massMape"),

  modelStatus: document.getElementById("modelStatus"),
  log: document.getElementById("log"),
};

const ctx = els.canvas.getContext("2d", { willReadFrequently: true });

// ---------------------------
// Fixed parameters (no UI prompts)
// ---------------------------
const SAMPLE_FPS = 2;       // 2 frames/sec
const DOWNSCALE_W = 320;    // speed + consistency

// Default fallback calibration (will be auto-calibrated if you enter an actual final area once)
const CM2_PER_PIXEL = 0.0012;

// ExG threshold (tune ONCE if needed)
const EXG_THRESHOLD = 15;

// Morphology (keeps mask cleaner)
const MORPH = { erodeIters: 1, dilateIters: 2 };

// ---------------------------
// Calibration storage (browser)
// ---------------------------
const CAL_KEY = "lettuce_cm2_per_pixel_v1";

function getCm2PerPixel() {
  const v = Number(localStorage.getItem(CAL_KEY));
  return isFinite(v) && v > 0 ? v : CM2_PER_PIXEL;
}
function setCm2PerPixel(v) {
  if (isFinite(v) && v > 0) localStorage.setItem(CAL_KEY, String(v));
}

// ---------------------------
// State
// ---------------------------
let alphaMassPerArea = null; // g/cm^2 (learned from CSV)
let current = {
  ok: false,
  durationS: 0,
  nFrames: 0,
  areasPx: [],
  timesS: [],
  features: null,
};

// ---------------------------
// Small utilities
// ---------------------------
function logln(s) {
  els.log.textContent += s + "\n";
  els.log.scrollTop = els.log.scrollHeight;
}
function setEnabled(el, enabled) {
  el.disabled = !enabled;
}
function fmt(x, d = 3) {
  if (!isFinite(x)) return "—";
  return Number(x).toFixed(d);
}
function parseCSV(text) {
  const lines = text.replace(/\r/g, "").trim().split("\n").filter(l => l.length);
  if (lines.length < 2) return [];
  const header = lines[0].split(",").map(s => s.trim());
  return lines.slice(1).map(line => {
    const cols = line.split(","); // assumes no quoted commas
    const row = {};
    header.forEach((h, i) => row[h] = (cols[i] ?? "").trim());
    return row;
  });
}
function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ---------------------------
// Segmentation: ExG (robust under blue grow LEDs)
// ---------------------------
function buildMask(imageData) {
  const { data, width, height } = imageData;
  const mask = new Uint8Array(width * height);

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];

    // Excess Green Index
    const exg = 2 * g - r - b;

    mask[p] = exg > EXG_THRESHOLD ? 1 : 0;
  }
  return mask;
}

function erode(mask, w, h) {
  const out = new Uint8Array(mask.length);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const keep =
        mask[i] &&
        mask[i - 1] && mask[i + 1] &&
        mask[i - w] && mask[i + w];
      out[i] = keep ? 1 : 0;
    }
  }
  return out;
}

function dilate(mask, w, h) {
  const out = new Uint8Array(mask.length);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const any =
        mask[i] ||
        mask[i - 1] || mask[i + 1] ||
        mask[i - w] || mask[i + w];
      out[i] = any ? 1 : 0;
    }
  }
  return out;
}

function applyMorph(mask, w, h) {
  let m = mask;
  for (let i = 0; i < MORPH.erodeIters; i++) m = erode(m, w, h);
  for (let i = 0; i < MORPH.dilateIters; i++) m = dilate(m, w, h);
  return m;
}

function maskArea(mask) {
  let s = 0;
  for (let i = 0; i < mask.length; i++) s += mask[i];
  return s;
}

function drawMaskOverlay(mask, w, h) {
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  for (let p = 0; p < mask.length; p++) {
    if (mask[p]) {
      const i = p * 4;
      d[i] = Math.min(255, d[i] + 40);
      d[i + 1] = Math.min(255, d[i + 1] + 120);
      d[i + 2] = Math.min(255, d[i + 2] + 40);
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ---------------------------
// Features (for export + diagnostics)
// ---------------------------
function linearFitSlope(xs, ys) {
  const n = xs.length;
  if (n < 2) return 0;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i];
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return 0;
  return (n * sxy - sx * sy) / denom;
}

function computeFeatures(timesS, areasPx) {
  const n = areasPx.length;
  const mean = areasPx.reduce((a, b) => a + b, 0) / Math.max(1, n);
  const max = Math.max(...areasPx);
  const min = Math.min(...areasPx);
  const slope = linearFitSlope(timesS, areasPx);
  const delta = areasPx[n - 1] - areasPx[0];

  return {
    area_mean_px: mean,
    area_max_px: max,
    area_slope_px_per_s: slope,
    area_delta_px: delta,
    area_min_px: min
  };
}

// ---------------------------
// Video loading + canvas sizing
// ---------------------------
async function loadVideoFile(file) {
  revokeLoadedVideoUrlIfAny();
  const url = URL.createObjectURL(file);
  els.video.src = url;

  await new Promise((res, rej) => {
    els.video.onloadedmetadata = () => res();
    els.video.onerror = () => rej(new Error("Video failed to load metadata."));
  });

  els.video._objectUrl = url;
}

function revokeLoadedVideoUrlIfAny() {
  if (els.video && els.video._objectUrl) {
    URL.revokeObjectURL(els.video._objectUrl);
    els.video._objectUrl = null;
  }
}

function setCanvasSizeFromVideo() {
  const vw = els.video.videoWidth;
  const vh = els.video.videoHeight;
  if (!vw || !vh) return;

  const scale = DOWNSCALE_W / vw;
  const w = Math.max(1, Math.round(vw * scale));
  const h = Math.max(1, Math.round(vh * scale));

  els.canvas.width = w;
  els.canvas.height = h;
}

// ---------------------------
// Video analysis (ROBUST: playback sampling)
// ---------------------------
async function analyzeVideo() {
  els.log.textContent = "";
  logln("Starting analysis… (playback sampling)");
  setEnabled(els.analyzeBtn, false);
  setEnabled(els.predictBtn, false);
  setEnabled(els.exportFeaturesBtn, false);
  setEnabled(els.fitBtn, false);

  current.ok = false;

  const duration = els.video.duration;
  if (!isFinite(duration) || duration <= 0) {
    logln("ERROR: video duration unavailable.");
    setEnabled(els.analyzeBtn, true);
    return;
  }

  setCanvasSizeFromVideo();
  const w = els.canvas.width;
  const h = els.canvas.height;

  if (!w || !h) {
    logln("ERROR: canvas size is 0. Video metadata may not be ready.");
    setEnabled(els.analyzeBtn, true);
    return;
  }

  current.durationS = duration;
  current.timesS = [];
  current.areasPx = [];
  current.features = null;
  current.nFrames = 0;

  // Helps autoplay in Chrome/Safari
  els.video.muted = true;
  els.video.playsInline = true;

  const dt = 1 / SAMPLE_FPS;
  let nextSampleT = 0;
  let frameCount = 0;

  try { els.video.currentTime = 0; } catch (_) {}

  const sampleFrame = () => {
    ctx.drawImage(els.video, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h);

    const rawMask = buildMask(img);
    const mask = applyMorph(rawMask, w, h);
    const areaPx = maskArea(mask);

    const t = els.video.currentTime;
    current.timesS.push(t);
    current.areasPx.push(areaPx);
    frameCount++;

    const expected = Math.max(1, Math.floor(duration * SAMPLE_FPS));
    if (frameCount === Math.floor(expected * 0.6)) {
      ctx.drawImage(els.video, 0, 0, w, h);
      drawMaskOverlay(mask, w, h);
    }

    if (frameCount === 1 || frameCount % 10 === 0) {
      logln(`Frame ${frameCount}: t=${t.toFixed(2)}s area_px=${areaPx}`);
    }
  };

  let stopped = false;
  const finish = () => {
    if (stopped) return;
    stopped = true;

    els.video.pause();

    current.nFrames = current.areasPx.length;
    if (current.nFrames < 3) {
      logln("ERROR: Too few samples collected. Try a different encoding or longer clip.");
      setEnabled(els.analyzeBtn, true);
      return;
    }

    const feats = computeFeatures(current.timesS, current.areasPx);
    current.features = feats;
    current.ok = true;

    els.framesSampled.textContent = String(current.nFrames);
    els.meanAreaPx.textContent = fmt(feats.area_mean_px, 1);
    els.maxAreaPx.textContent = fmt(feats.area_max_px, 1);
    els.slopePxPerS.textContent = fmt(feats.area_slope_px_per_s, 4);

    setEnabled(els.predictBtn, true);
    setEnabled(els.exportFeaturesBtn, true);
    setEnabled(els.resetBtn, true);
    setEnabled(els.fitBtn, !!els.trainCsvInput.files?.length);

    logln("Analysis complete.");
    logln("Extracted features:");
    logln(JSON.stringify(feats, null, 2));

    // Helpful diagnostic: baseline cm2 using current calibration
    const lastPx = current.areasPx[current.areasPx.length - 1];
    const baseline = lastPx * getCm2PerPixel();
    logln(`Diagnostic: last_frame_area_cm2_baseline = ${baseline.toFixed(3)} (cm2PerPixel=${getCm2PerPixel()})`);

    setEnabled(els.analyzeBtn, true);
  };

  const hardTimeoutMs = Math.max(15000, Math.ceil(duration * 1000) + 10000);
  const timeoutId = setTimeout(() => {
    logln("WARNING: Analysis timed out. Finishing with collected samples.");
    finish();
  }, hardTimeoutMs);

  try {
    await els.video.play();
  } catch (e) {
    clearTimeout(timeoutId);
    logln("ERROR: Browser blocked autoplay. Click play once on the video, then click Analyze again.");
    setEnabled(els.analyzeBtn, true);
    return;
  }

  const useRVFC = typeof els.video.requestVideoFrameCallback === "function";

  if (useRVFC) {
    const onFrame = () => {
      if (stopped) return;
      const t = els.video.currentTime;

      if (t + 1e-6 >= nextSampleT) {
        sampleFrame();
        nextSampleT += dt;
      }

      if (t >= duration - 0.05 || els.video.ended) {
        clearTimeout(timeoutId);
        finish();
        return;
      }

      els.video.requestVideoFrameCallback(onFrame);
    };
    els.video.requestVideoFrameCallback(onFrame);
  } else {
    const interval = setInterval(() => {
      if (stopped) return;
      const t = els.video.currentTime;

      if (t + 1e-6 >= nextSampleT) {
        sampleFrame();
        nextSampleT += dt;
      }

      if (t >= duration - 0.05 || els.video.ended) {
        clearInterval(interval);
        clearTimeout(timeoutId);
        finish();
      }
    }, 20);
  }
}

// ---------------------------
// Training: learn alpha = median(mass/area)
// ---------------------------
async function fitFromCsv(file) {
  els.modelStatus.textContent = "Model status: fitting…";
  logln("Reading training CSV…");

  const text = await file.text();
  const rows = parseCSV(text);

  const required = ["trial_id", "final_area_cm2", "final_mass_g"];
  for (const col of required) {
    if (!rows[0] || !(col in rows[0])) {
      els.modelStatus.textContent = "Model status: fit failed (missing columns).";
      logln(`ERROR: Missing required column in training CSV: ${col}`);
      return;
    }
  }

  const ratios = [];
  for (const r of rows) {
    const area = Number(r.final_area_cm2);
    const mass = Number(r.final_mass_g);
    if (isFinite(area) && isFinite(mass) && area > 0) ratios.push(mass / area);
  }

  if (ratios.length < 3) {
    els.modelStatus.textContent = "Model status: fit failed (need more rows).";
    logln("ERROR: Need at least 3 valid rows to compute mass/area ratio.");
    return;
  }

  ratios.sort((a, b) => a - b);
  const mid = Math.floor(ratios.length / 2);
  const median = ratios.length % 2 ? ratios[mid] : 0.5 * (ratios[mid - 1] + ratios[mid]);

  alphaMassPerArea = median;

  els.modelStatus.textContent =
    `Model status: fit OK (n=${ratios.length}). Using mass≈alpha·area (alpha=${alphaMassPerArea.toFixed(5)} g/cm²).`;
  logln(`Fit OK: alphaMassPerArea = ${alphaMassPerArea} g/cm^2`);

  // Also log the current calibration state
  logln(`Current cm2PerPixel (stored) = ${getCm2PerPixel()}`);
}

// ---------------------------
// Prediction + evaluation (aimed for <10% once calibrated)
// ---------------------------
function computeMape(pred, actual) {
  if (!isFinite(actual) || actual === 0) return NaN;
  return Math.abs((pred - actual) / actual) * 100;
}

function predictNow() {
  if (!current.ok || !current.features || current.areasPx.length < 2) return;

  const lastAreaPx = current.areasPx[current.areasPx.length - 1];

  // If user provided actual final area, auto-calibrate cm2PerPixel from THIS video
  const actualArea = Number(els.actualArea.value);
  if (isFinite(actualArea) && actualArea > 0 && lastAreaPx > 0) {
    const newCm2PerPixel = actualArea / lastAreaPx;
    setCm2PerPixel(newCm2PerPixel);
    logln(`Auto-calibration: cm2PerPixel set to ${newCm2PerPixel} using actualArea=${actualArea} and lastAreaPx=${lastAreaPx}`);
  }

  const cm2PerPixel = getCm2PerPixel();

  // Area prediction: direct conversion from pixel area
  let predAreaCm2 = Math.max(0, lastAreaPx * cm2PerPixel);

  // Mass prediction: alpha * area (alpha learned from CSV). If not fit, fallback.
  const alpha = (isFinite(alphaMassPerArea) && alphaMassPerArea > 0) ? alphaMassPerArea : 0.02;
  let predMassG = Math.max(0, alpha * predAreaCm2);

  els.predArea.textContent = fmt(predAreaCm2, 2);
  els.predMass.textContent = fmt(predMassG, 2);

  // Errors if actuals present
  const actualMass = Number(els.actualMass.value);

  if (isFinite(actualArea)) {
    els.areaAbsErr.textContent = fmt(Math.abs(predAreaCm2 - actualArea), 2);
    els.areaMape.textContent = fmt(computeMape(predAreaCm2, actualArea), 2) + "%";
  } else {
    els.areaAbsErr.textContent = "—";
    els.areaMape.textContent = "—";
  }

  if (isFinite(actualMass)) {
    els.massAbsErr.textContent = fmt(Math.abs(predMassG - actualMass), 2);
    els.massMape.textContent = fmt(computeMape(predMassG, actualMass), 2) + "%";
  } else {
    els.massAbsErr.textContent = "—";
    els.massMape.textContent = "—";
  }

  // Diagnostics
  logln(`Diagnostic: lastAreaPx=${lastAreaPx}, cm2PerPixel=${cm2PerPixel}, predAreaCm2=${predAreaCm2}, alpha=${alpha}, predMassG=${predMassG}`);
}

// ---------------------------
// Export features for your dataset-building workflow
// ---------------------------
function exportCurrentFeaturesCsv() {
  if (!current.ok || !current.features) return;

  const tid = (els.trialId.value || "").trim() || "UNLABELED";
  const f = current.features;

  // Also include last_frame_area_px since prediction uses it
  const lastFrameAreaPx = current.areasPx[current.areasPx.length - 1];

  const header = [
    "trial_id",
    "area_mean_px",
    "area_max_px",
    "area_slope_px_per_s",
    "area_delta_px",
    "last_frame_area_px",
    "final_area_cm2",
    "final_mass_g"
  ].join(",");

  const line = [
    tid,
    f.area_mean_px,
    f.area_max_px,
    f.area_slope_px_per_s,
    f.area_delta_px,
    lastFrameAreaPx,
    "", ""
  ].join(",");

  downloadText(`${tid}_features.csv`, header + "\n" + line + "\n");
}

// ---------------------------
// UI wiring
// ---------------------------
els.videoInput.addEventListener("change", async () => {
  const file = els.videoInput.files?.[0];
  if (!file) return;

  setEnabled(els.analyzeBtn, false);
  setEnabled(els.predictBtn, false);
  setEnabled(els.exportFeaturesBtn, false);

  try {
    await loadVideoFile(file);
    logln(`Loaded video: ${file.name}`);
    logln(`Video metadata: ${els.video.videoWidth}x${els.video.videoHeight}, duration=${els.video.duration.toFixed(2)}s`);
    setEnabled(els.analyzeBtn, true);
    setEnabled(els.resetBtn, true);
  } catch (e) {
    logln("ERROR: " + (e?.message || String(e)));
    setEnabled(els.analyzeBtn, false);
  }
});

els.analyzeBtn.addEventListener("click", analyzeVideo);

els.trainCsvInput.addEventListener("change", () => {
  setEnabled(els.fitBtn, !!els.trainCsvInput.files?.length);
});

els.fitBtn.addEventListener("click", async () => {
  const file = els.trainCsvInput.files?.[0];
  if (!file) return;
  await fitFromCsv(file);
});

els.exportFeaturesBtn.addEventListener("click", exportCurrentFeaturesCsv);

els.predictBtn.addEventListener("click", predictNow);

els.resetBtn.addEventListener("click", () => {
  els.log.textContent = "";
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);

  revokeLoadedVideoUrlIfAny();
  els.video.removeAttribute("src");
  els.video.load();

  current = { ok: false, durationS: 0, nFrames: 0, areasPx: [], timesS: [], features: null };

  els.framesSampled.textContent = "—";
  els.meanAreaPx.textContent = "—";
  els.maxAreaPx.textContent = "—";
  els.slopePxPerS.textContent = "—";

  els.predArea.textContent = "—";
  els.predMass.textContent = "—";

  els.areaAbsErr.textContent = "—";
  els.areaMape.textContent = "—";
  els.massAbsErr.textContent = "—";
  els.massMape.textContent = "—";

  setEnabled(els.analyzeBtn, false);
  setEnabled(els.predictBtn, false);
  setEnabled(els.fitBtn, false);
  setEnabled(els.exportFeaturesBtn, false);
  setEnabled(els.resetBtn, false);

  els.modelStatus.textContent = `Model status: not fit. Current cm2PerPixel=${getCm2PerPixel()}`;
  logln("Reset complete.");
});

// Initial UI state
setEnabled(els.analyzeBtn, false);
setEnabled(els.resetBtn, false);
setEnabled(els.fitBtn, false);
setEnabled(els.exportFeaturesBtn, false);
setEnabled(els.predictBtn, false);

els.modelStatus.textContent = `Model status: not fit. Current cm2PerPixel=${getCm2PerPixel()}`;

// Helpful startup note
logln("Ready. Tip: After analyzing, enter Actual final area once and click Predict to auto-calibrate cm²/pixel.");
logln("If segmentation grabs background, increase EXG_THRESHOLD; if it misses lettuce, decrease EXG_THRESHOLD.");
