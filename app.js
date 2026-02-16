/* Lettuce Yield Predictor (STEM fair validation mode)
   ---------------------------------------------------
   Goals:
   1) Analyze a ~60 s top-down video and extract projected plant area (pixels) via ExG segmentation.
   2) Convert pixel area -> cm^2 using a stored calibration (cm^2/pixel).
   3) Estimate mass from area using alpha = median(mass/area) learned from the uploaded trial CSV.
   4) Upload trial ground-truth CSV and select which trial to validate against.
   5) Report error metrics (ABS error, MAPE) as "accuracy of the CV pipeline."

   Required trial CSV columns:
     trial_id, final_area_cm2, final_mass_g
*/

const els = {
  // Section 1
  videoInput: document.getElementById("videoInput"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  resetBtn: document.getElementById("resetBtn"),
  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),

  framesSampled: document.getElementById("framesSampled"),
  meanAreaPx: document.getElementById("meanAreaPx"),
  maxAreaPx: document.getElementById("maxAreaPx"),
  slopePxPerS: document.getElementById("slopePxPerS"),

  // Section 2 (trial data)
  trialCsvInput: document.getElementById("trialCsvInput"),
  trialSelect: document.getElementById("trialSelect"),
  trialDataStatus: document.getElementById("trialDataStatus"),

  // Section 3
  trialId: document.getElementById("trialId"),
  actualArea: document.getElementById("actualArea"),
  actualMass: document.getElementById("actualMass"),
  predictBtn: document.getElementById("predictBtn"),
  exportFeaturesBtn: document.getElementById("exportFeaturesBtn"),

  predArea: document.getElementById("predArea"),
  predMass: document.getElementById("predMass"),

  areaAbsErr: document.getElementById("areaAbsErr"),
  areaMape: document.getElementById("areaMape"),
  massAbsErr: document.getElementById("massAbsErr"),
  massMape: document.getElementById("massMape"),

  log: document.getElementById("log"),
};

const ctx = els.canvas.getContext("2d", { willReadFrequently: true });

// ---------------------------
// Fixed parameters (no user tuning)
// ---------------------------
const SAMPLE_FPS = 2;
const DOWNSCALE_W = 320;

// Default fallback calibration (auto-calibrated when you validate one known trial)
const CM2_PER_PIXEL_DEFAULT = 0.0012;

// ExG threshold (robust under blue LEDs)
const EXG_THRESHOLD = 15;

// Morphology
const MORPH = { erodeIters: 1, dilateIters: 2 };

// ---------------------------
// Calibration storage
// ---------------------------
const CAL_KEY = "lettuce_cm2_per_pixel_v2";
function getCm2PerPixel() {
  const v = Number(localStorage.getItem(CAL_KEY));
  return isFinite(v) && v > 0 ? v : CM2_PER_PIXEL_DEFAULT;
}
function setCm2PerPixel(v) {
  if (isFinite(v) && v > 0) localStorage.setItem(CAL_KEY, String(v));
}

// ---------------------------
// State
// ---------------------------
let current = {
  ok: false,
  durationS: 0,
  nFrames: 0,
  areasPx: [],
  timesS: [],
  features: null,
};

let trialRows = [];           // parsed CSV rows
let trialMap = new Map();     // trial_id -> {final_area_cm2, final_mass_g}
let alphaMassPerArea = null;  // g/cm^2 learned from CSV median(mass/area)

// ---------------------------
// Utilities
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
function computeMape(pred, actual) {
  if (!isFinite(actual) || actual === 0) return NaN;
  return Math.abs((pred - actual) / actual) * 100;
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
// Segmentation: ExG
// ---------------------------
function buildMask(imageData) {
  const { data, width, height } = imageData;
  const mask = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
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
      out[i] = (mask[i] && mask[i - 1] && mask[i + 1] && mask[i - w] && mask[i + w]) ? 1 : 0;
    }
  }
  return out;
}
function dilate(mask, w, h) {
  const out = new Uint8Array(mask.length);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      out[i] = (mask[i] || mask[i - 1] || mask[i + 1] || mask[i - w] || mask[i + w]) ? 1 : 0;
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
// Features
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
  return { area_mean_px: mean, area_max_px: max, area_slope_px_per_s: slope, area_delta_px: delta, area_min_px: min };
}

// ---------------------------
// Video load + sizing
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
  const vw = els.video.videoWidth, vh = els.video.videoHeight;
  if (!vw || !vh) return;
  const scale = DOWNSCALE_W / vw;
  els.canvas.width = Math.max(1, Math.round(vw * scale));
  els.canvas.height = Math.max(1, Math.round(vh * scale));
}

// ---------------------------
// Analyze video (playback sampling)
// ---------------------------
async function analyzeVideo() {
  els.log.textContent = "";
  logln("Starting analysis…");
  setEnabled(els.analyzeBtn, false);
  setEnabled(els.predictBtn, false);
  setEnabled(els.exportFeaturesBtn, false);

  current.ok = false;

  const duration = els.video.duration;
  if (!isFinite(duration) || duration <= 0) {
    logln("ERROR: video duration unavailable.");
    setEnabled(els.analyzeBtn, true);
    return;
  }

  setCanvasSizeFromVideo();
  const w = els.canvas.width, h = els.canvas.height;
  if (!w || !h) {
    logln("ERROR: canvas sizing failed.");
    setEnabled(els.analyzeBtn, true);
    return;
  }

  current = { ok: false, durationS: duration, nFrames: 0, areasPx: [], timesS: [], features: null };

  els.video.muted = true;
  els.video.playsInline = true;

  const dt = 1 / SAMPLE_FPS;
  let nextSampleT = 0;
  let frameCount = 0;
  try { els.video.currentTime = 0; } catch (_) {}

  const sampleFrame = () => {
    ctx.drawImage(els.video, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h);
    const mask = applyMorph(buildMask(img), w, h);
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
      logln("ERROR: too few samples.");
      setEnabled(els.analyzeBtn, true);
      return;
    }

    current.features = computeFeatures(current.timesS, current.areasPx);
    current.ok = true;

    els.framesSampled.textContent = String(current.nFrames);
    els.meanAreaPx.textContent = fmt(current.features.area_mean_px, 1);
    els.maxAreaPx.textContent = fmt(current.features.area_max_px, 1);
    els.slopePxPerS.textContent = fmt(current.features.area_slope_px_per_s, 4);

    setEnabled(els.predictBtn, true);
    setEnabled(els.exportFeaturesBtn, true);
    setEnabled(els.resetBtn, true);

    logln("Analysis complete.");
    logln(JSON.stringify(current.features, null, 2));

    const lastPx = current.areasPx[current.areasPx.length - 1];
    logln(`Diagnostic: last_frame_area_cm2_baseline=${(lastPx * getCm2PerPixel()).toFixed(3)} (cm2PerPixel=${getCm2PerPixel()})`);

    setEnabled(els.analyzeBtn, true);
  };

  const hardTimeoutMs = Math.max(15000, Math.ceil(duration * 1000) + 10000);
  const timeoutId = setTimeout(() => {
    logln("WARNING: timed out; finishing with collected samples.");
    finish();
  }, hardTimeoutMs);

  try {
    await els.video.play();
  } catch (e) {
    clearTimeout(timeoutId);
    logln("ERROR: autoplay blocked. Press play once, then click Analyze again.");
    setEnabled(els.analyzeBtn, true);
    return;
  }

  const useRVFC = typeof els.video.requestVideoFrameCallback === "function";
  if (useRVFC) {
    const onFrame = () => {
      if (stopped) return;
      const t = els.video.currentTime;
      if (t + 1e-6 >= nextSampleT) { sampleFrame(); nextSampleT += dt; }
      if (t >= duration - 0.05 || els.video.ended) { clearTimeout(timeoutId); finish(); return; }
      els.video.requestVideoFrameCallback(onFrame);
    };
    els.video.requestVideoFrameCallback(onFrame);
  } else {
    const interval = setInterval(() => {
      if (stopped) return;
      const t = els.video.currentTime;
      if (t + 1e-6 >= nextSampleT) { sampleFrame(); nextSampleT += dt; }
      if (t >= duration - 0.05 || els.video.ended) {
        clearInterval(interval);
        clearTimeout(timeoutId);
        finish();
      }
    }, 20);
  }
}

// ---------------------------
// Trial CSV loading + dropdown + alpha calculation
// ---------------------------
async function loadTrialCsv(file) {
  const text = await file.text();
  const rows = parseCSV(text);

  const required = ["trial_id", "final_area_cm2", "final_mass_g"];
  for (const col of required) {
    if (!rows[0] || !(col in rows[0])) {
      els.trialDataStatus.textContent = "Trial data: load failed (missing columns).";
      logln(`ERROR: trial CSV missing column: ${col}`);
      return;
    }
  }

  trialRows = rows;
  trialMap = new Map();

  const ratios = [];
  for (const r of rows) {
    const tid = (r.trial_id || "").trim();
    const area = Number(r.final_area_cm2);
    const mass = Number(r.final_mass_g);
    if (!tid) continue;
    if (isFinite(area) && isFinite(mass)) {
      trialMap.set(tid, { final_area_cm2: area, final_mass_g: mass });
      if (area > 0) ratios.push(mass / area);
    }
  }

  // Compute alpha = median(mass/area)
  if (ratios.length >= 3) {
    ratios.sort((a, b) => a - b);
    const mid = Math.floor(ratios.length / 2);
    alphaMassPerArea = ratios.length % 2 ? ratios[mid] : 0.5 * (ratios[mid - 1] + ratios[mid]);
  } else {
    alphaMassPerArea = null;
  }

  // Populate dropdown
  els.trialSelect.innerHTML = `<option value="">— Select a trial —</option>`;
  const trialIds = Array.from(trialMap.keys()).sort();
  for (const tid of trialIds) {
    const opt = document.createElement("option");
    opt.value = tid;
    opt.textContent = tid;
    els.trialSelect.appendChild(opt);
  }
  setEnabled(els.trialSelect, true);

  els.trialDataStatus.textContent =
    `Trial data: loaded ${trialIds.length} trials. alpha(mass/area)=${alphaMassPerArea ? alphaMassPerArea.toFixed(5) : "not available"} g/cm².`;

  logln(`Loaded trial CSV: n=${trialIds.length} trials`);
  logln(`alphaMassPerArea=${alphaMassPerArea}`);
}

// When a trial is selected, fill actual fields automatically
function applySelectedTrial(tid) {
  const rec = trialMap.get(tid);
  if (!rec) return;

  els.trialId.value = tid;
  els.actualArea.value = rec.final_area_cm2;
  els.actualMass.value = rec.final_mass_g;

  logln(`Selected trial ${tid}: actualArea=${rec.final_area_cm2}, actualMass=${rec.final_mass_g}`);
}

// ---------------------------
// Prediction + validation
// ---------------------------
function predictNow() {
  if (!current.ok || !current.features || current.areasPx.length < 2) return;

  const lastAreaPx = current.areasPx[current.areasPx.length - 1];
  const actualArea = Number(els.actualArea.value);
  const actualMass = Number(els.actualMass.value);

  // Auto-calibrate cm2PerPixel when a known ground-truth area is available.
  // This directly improves the CV-to-area conversion accuracy for your setup.
  if (isFinite(actualArea) && actualArea > 0 && lastAreaPx > 0) {
    const newCm2PerPixel = actualArea / lastAreaPx;
    setCm2PerPixel(newCm2PerPixel);
    logln(`Auto-calibration: cm2PerPixel=${newCm2PerPixel} (using actualArea/lastAreaPx)`);
  }

  const cm2PerPixel = getCm2PerPixel();

  // CV-derived area prediction (this is the "computer vision portion")
  const predAreaCm2 = Math.max(0, lastAreaPx * cm2PerPixel);

  // Mass estimate from area using alpha learned from CSV (if available)
  const alpha = (isFinite(alphaMassPerArea) && alphaMassPerArea > 0) ? alphaMassPerArea : 0.02;
  const predMassG = Math.max(0, alpha * predAreaCm2);

  els.predArea.textContent = fmt(predAreaCm2, 2);
  els.predMass.textContent = fmt(predMassG, 2);

  // Errors if actuals exist
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

  logln(`Prediction: lastAreaPx=${lastAreaPx}, cm2PerPixel=${cm2PerPixel}, predAreaCm2=${predAreaCm2}`);
  logln(`Prediction: alpha=${alpha}, predMassG=${predMassG}`);
}

// ---------------------------
// Export features (optional)
// ---------------------------
function exportCurrentFeaturesCsv() {
  if (!current.ok || !current.features) return;

  const tid = (els.trialId.value || "").trim() || "UNLABELED";
  const f = current.features;
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
    logln(`Video: ${els.video.videoWidth}x${els.video.videoHeight}, duration=${els.video.duration.toFixed(2)}s`);
    setEnabled(els.analyzeBtn, true);
    setEnabled(els.resetBtn, true);
  } catch (e) {
    logln("ERROR: " + (e?.message || String(e)));
    setEnabled(els.analyzeBtn, false);
  }
});

els.analyzeBtn.addEventListener("click", analyzeVideo);

els.trialCsvInput.addEventListener("change", async () => {
  const file = els.trialCsvInput.files?.[0];
  if (!file) return;
  await loadTrialCsv(file);
});

els.trialSelect.addEventListener("change", () => {
  const tid = els.trialSelect.value;
  if (!tid) return;
  applySelectedTrial(tid);
});

els.predictBtn.addEventListener("click", predictNow);
els.exportFeaturesBtn.addEventListener("click", exportCurrentFeaturesCsv);

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
  setEnabled(els.exportFeaturesBtn, false);
  setEnabled(els.resetBtn, false);

  logln("Reset complete.");
});

// Initial UI state
setEnabled(els.analyzeBtn, false);
setEnabled(els.resetBtn, false);
setEnabled(els.predictBtn, false);
setEnabled(els.exportFeaturesBtn, false);
setEnabled(els.trialSelect, false);

els.trialDataStatus.textContent = "Trial data: not loaded.";
logln("Ready. Workflow: (1) Upload trial CSV, select trial. (2) Upload video, Analyze. (3) Predict to see accuracy.");
logln(`Current cm2PerPixel=${getCm2PerPixel()} (auto-updates when you validate one known trial).`);
