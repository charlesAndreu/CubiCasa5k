const $ = (id) => document.getElementById(id);

const sourcePreset = $("sourcePreset");
const sourceUpload = $("sourceUpload");
const planLabel = $("planLabel");
const uploadLabel = $("uploadLabel");
const planSelect = $("planSelect");
const fileInput = $("fileInput");
const modelSelect = $("modelSelect");
const runBtn = $("runBtn");
const statusEl = $("status");

const baseLayer = $("baseLayer");
const segAlphaLabel = $("segAlphaLabel");
const segAlpha = $("segAlpha");
const segAlphaVal = $("segAlphaVal");

const methodSelect = $("method");
const axisBiasLabel = $("axisBiasLabel");
const threshold = $("threshold");
const thresholdVal = $("thresholdVal");
const axisBias = $("axisBias");
const axisBiasVal = $("axisBiasVal");
const snapAlign = $("snapAlign");
const snapAlignVal = $("snapAlignVal");
const wallEvidence = $("wallEvidence");
const wallEvidenceVal = $("wallEvidenceVal");
const minWallFraction = $("minWallFraction");
const minWallFractionVal = $("minWallFractionVal");

const imgOverlay = $("imgOverlay");
const imgWrap = imgOverlay.parentElement;
const overlayCaption = $("overlayCaption");
const dlOverlayPng = $("dlOverlayPng");
const dlSkeletonJson = $("dlSkeletonJson");
const resetViewBtn = $("resetViewBtn");

const criteriaInputs = [threshold, axisBias, snapAlign, wallEvidence, minWallFraction];
const BASE_CAPTIONS = {
  map: "Skeleton on map",
  segmentation: "Skeleton on segmentation",
  both: "Skeleton on segmentation + map",
};

let postprocTimer = null;
let uploadId = null;
let hasRun = false;

function isUploadMode() {
  return sourceUpload.checked;
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg || "";
  statusEl.classList.toggle("error", isError);
}

const statusSpinner = $("statusSpinner");
let busyCount = 0;

function setBusy(busy) {
  busyCount = Math.max(0, busyCount + (busy ? 1 : -1));
  const isBusy = busyCount > 0;
  statusSpinner.classList.toggle("hidden", !isBusy);
  imgOverlay.classList.toggle("loading", isBusy);
}

function cacheBust(url) {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}t=${Date.now()}`;
}

function planId() {
  return parseInt(planSelect.value, 10);
}

function modelId() {
  return modelSelect.value;
}

function qs(extra = {}) {
  const base = { model_id: modelId() };
  if (isUploadMode()) {
    if (!uploadId) throw new Error("Upload an image first");
    base.upload_id = uploadId;
  } else {
    base.plan_id = String(planId());
  }
  return new URLSearchParams({ ...base, ...extra }).toString();
}

function runBody() {
  const body = { model_id: modelId() };
  if (isUploadMode()) {
    if (!uploadId) throw new Error("Upload an image first");
    body.upload_id = uploadId;
  } else {
    body.plan_id = planId();
  }
  return body;
}

function overlayQuery() {
  const q = {
    threshold: Number(threshold.value).toFixed(2),
    method: methodSelect.value,
    axis_bias: Number(axisBias.value).toFixed(2),
    snap_align: Number(snapAlign.value).toFixed(0),
    wall_evidence: Number(wallEvidence.value).toFixed(2),
    min_wall_fraction: Number(minWallFraction.value).toFixed(2),
    base: baseLayer.value,
  };
  if (baseLayer.value === "both") {
    q.seg_alpha = Number(segAlpha.value).toFixed(2);
  }
  return q;
}

function updateMethodUI() {
  axisBiasLabel.classList.toggle("hidden", methodSelect.value === "evidence");
  updateButtons();
}

function syncCriteriaLabels() {
  thresholdVal.textContent = Number(threshold.value).toFixed(2);
  axisBiasVal.textContent = Number(axisBias.value).toFixed(2);
  const snap = Number(snapAlign.value);
  snapAlignVal.textContent = snap === 0 ? "0 (off)" : `${snap.toFixed(0)}px`;
  wallEvidenceVal.textContent = Number(wallEvidence.value).toFixed(2);
  minWallFractionVal.textContent = Number(minWallFraction.value).toFixed(2);
  segAlphaVal.textContent = Number(segAlpha.value).toFixed(2);
}

function updateBaseLayerUI() {
  segAlphaLabel.classList.toggle("hidden", baseLayer.value !== "both");
  overlayCaption.textContent = BASE_CAPTIONS[baseLayer.value] || "Skeleton";
}

function updateSourceUI() {
  const upload = isUploadMode();
  planLabel.classList.toggle("hidden", upload);
  uploadLabel.classList.toggle("hidden", !upload);
  clearOutputs();
  if (upload) {
    uploadId = null;
  } else {
    fileInput.value = "";
  }
  updateButtons();
}

function clearOutputs() {
  imgOverlay.removeAttribute("src");
  hasRun = false;
  resetView();
  updateButtons();
}

// --- Pan/zoom: mouse wheel to zoom (toward the cursor), drag to pan. ---
const view = { scale: 1, panX: 0, panY: 0 };
const ZOOM_MIN = 1;
const ZOOM_MAX = 12;

function applyView() {
  imgOverlay.style.transform = `translate(${view.panX}px, ${view.panY}px) scale(${view.scale})`;
}

function resetView() {
  view.scale = 1;
  view.panX = 0;
  view.panY = 0;
  applyView();
}

imgWrap.addEventListener(
  "wheel",
  (e) => {
    if (!imgOverlay.getAttribute("src")) return;
    e.preventDefault();
    const rect = imgWrap.getBoundingClientRect();
    // cursor position relative to the wrap's center (transform-origin is center by default)
    const cx = e.clientX - rect.left - rect.width / 2;
    const cy = e.clientY - rect.top - rect.height / 2;
    const prevScale = view.scale;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    view.scale = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, view.scale * factor));
    if (view.scale === prevScale) return;
    // keep the point under the cursor stationary while scaling around it
    view.panX = cx - (cx - view.panX) * (view.scale / prevScale);
    view.panY = cy - (cy - view.panY) * (view.scale / prevScale);
    applyView();
  },
  { passive: false }
);

let dragging = false;
let dragStart = { x: 0, y: 0, panX: 0, panY: 0 };

imgWrap.addEventListener("mousedown", (e) => {
  if (!imgOverlay.getAttribute("src")) return;
  dragging = true;
  imgWrap.classList.add("dragging");
  dragStart = { x: e.clientX, y: e.clientY, panX: view.panX, panY: view.panY };
});

window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  view.panX = dragStart.panX + (e.clientX - dragStart.x);
  view.panY = dragStart.panY + (e.clientY - dragStart.y);
  applyView();
});

window.addEventListener("mouseup", () => {
  dragging = false;
  imgWrap.classList.remove("dragging");
});

imgWrap.addEventListener("dblclick", resetView);

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function downloadBlob(url, filename) {
  const res = await fetch(url);
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || res.statusText);
  }
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objectUrl);
}

async function handleFileSelect() {
  const file = fileInput.files?.[0];
  if (!file) return;
  setStatus(`Uploading ${file.name}…`);
  const form = new FormData();
  form.append("image", file);
  try {
    const meta = await fetchJson("/api/upload", { method: "POST", body: form });
    uploadId = meta.upload_id;
    setStatus(`${file.name} (${meta.width}×${meta.height})`);
    updateButtons();
  } catch (e) {
    uploadId = null;
    setStatus(e.message, true);
  }
}

async function init() {
  const [plans, models] = await Promise.all([
    fetchJson("/api/plans"),
    fetchJson("/api/models"),
  ]);

  planSelect.innerHTML = plans
    .map((p) => `<option value="${p.id}">#${p.id} — ${p.folder.replace(/^\//, "")}</option>`)
    .join("");

  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    modelSelect.appendChild(opt);
  });

  if (models.length === 0) {
    setStatus("No checkpoints found under runs_cubi*", true);
  }

  sourcePreset.addEventListener("change", updateSourceUI);
  sourceUpload.addEventListener("change", updateSourceUI);
  planSelect.addEventListener("change", clearOutputs);
  fileInput.addEventListener("change", handleFileSelect);
  modelSelect.addEventListener("change", () => {
    clearOutputs();
    updateButtons();
  });

  runBtn.addEventListener("click", runAll);

  methodSelect.addEventListener("change", () => {
    updateMethodUI();
    if (hasRun) scheduleOverlayUpdate();
  });

  baseLayer.addEventListener("change", () => {
    updateBaseLayerUI();
    updateButtons();
    if (hasRun) scheduleOverlayUpdate();
  });
  segAlpha.addEventListener("input", () => {
    segAlphaVal.textContent = Number(segAlpha.value).toFixed(2);
    if (hasRun) scheduleOverlayUpdate();
  });

  criteriaInputs.forEach((el) => {
    el.addEventListener("input", () => {
      syncCriteriaLabels();
      if (hasRun) scheduleOverlayUpdate();
    });
  });

  resetViewBtn.addEventListener("click", resetView);

  dlOverlayPng.addEventListener("click", () => {
    downloadBlob(`/api/overlay.png?${qs(overlayQuery())}`, "skeleton_overlay.png").catch((e) =>
      setStatus(e.message, true)
    );
  });
  dlSkeletonJson.addEventListener("click", () => {
    downloadBlob(`/api/skeleton.json?${qs(overlayQuery())}`, "skeleton.json").catch((e) =>
      setStatus(e.message, true)
    );
  });

  syncCriteriaLabels();
  updateBaseLayerUI();
  updateMethodUI();
}

function updateButtons() {
  const hasModel = Boolean(modelId());
  const hasInput = isUploadMode() ? Boolean(uploadId) : true;
  const ready = hasModel && hasInput;
  runBtn.disabled = !ready;
  criteriaInputs.forEach((el) => (el.disabled = !ready));
  axisBias.disabled = !ready || methodSelect.value === "evidence";
  methodSelect.disabled = !ready;
  baseLayer.disabled = !ready;
  segAlpha.disabled = !ready || baseLayer.value !== "both";
  dlOverlayPng.disabled = !hasRun;
  dlSkeletonJson.disabled = !hasRun;
  resetViewBtn.disabled = !hasRun;
}

async function runAll() {
  if (!modelId()) return;
  if (isUploadMode() && !uploadId) return;
  runBtn.disabled = true;
  setBusy(true);
  setStatus("Running inference (first load may take a minute)…");

  try {
    const meta = await fetchJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(runBody()),
    });
    setStatus(meta.folder || "Done");
    hasRun = true;
    await loadOverlay();
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    setBusy(false);
    updateButtons();
  }
}

function scheduleOverlayUpdate() {
  clearTimeout(postprocTimer);
  postprocTimer = setTimeout(loadOverlay, 200);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    imgOverlay.onload = () => resolve();
    imgOverlay.onerror = () => reject(new Error("Failed to load overlay image"));
    imgOverlay.src = src;
  });
}

async function loadOverlay() {
  if (!hasRun) return;
  setBusy(true);
  setStatus("Computing wall skeleton…");
  try {
    await loadImage(cacheBust(`/api/overlay.png?${qs(overlayQuery())}`));
    setStatus("Ready");
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    setBusy(false);
  }
}

init().catch((e) => setStatus(e.message, true));
