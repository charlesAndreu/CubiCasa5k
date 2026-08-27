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
const overlayCaption = $("overlayCaption");
const dlOverlayPng = $("dlOverlayPng");
const dlSkeletonJson = $("dlSkeletonJson");

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
  updateButtons();
}

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
  updateButtons();
}

function updateButtons() {
  const hasModel = Boolean(modelId());
  const hasInput = isUploadMode() ? Boolean(uploadId) : true;
  const ready = hasModel && hasInput;
  runBtn.disabled = !ready;
  criteriaInputs.forEach((el) => (el.disabled = !ready));
  baseLayer.disabled = !ready;
  segAlpha.disabled = !ready || baseLayer.value !== "both";
  dlOverlayPng.disabled = !hasRun;
  dlSkeletonJson.disabled = !hasRun;
}

async function runAll() {
  if (!modelId()) return;
  if (isUploadMode() && !uploadId) return;
  runBtn.disabled = true;
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
    updateButtons();
  }
}

function scheduleOverlayUpdate() {
  clearTimeout(postprocTimer);
  postprocTimer = setTimeout(loadOverlay, 200);
}

async function loadOverlay() {
  if (!hasRun) return;
  setStatus("Computing wall skeleton…");
  try {
    imgOverlay.src = cacheBust(`/api/overlay.png?${qs(overlayQuery())}`);
    setStatus("Ready");
  } catch (e) {
    setStatus(e.message, true);
  }
}

init().catch((e) => setStatus(e.message, true));
