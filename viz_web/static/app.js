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
const threshold = $("threshold");
const thresholdVal = $("thresholdVal");

const images = {
  input: $("imgInput"),
  wall: $("imgWall"),
  opening: $("imgOpening"),
  iconHm: $("imgIconHm"),
  room: $("imgRoom"),
  iconSeg: $("imgIconSeg"),
  roomEntropy: $("imgRoomEntropy"),
  iconEntropy: $("imgIconEntropy"),
  postproc: $("imgPostproc"),
};

let postprocTimer = null;
let uploadId = null;

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
  const p = new URLSearchParams({ ...base, ...extra });
  return p.toString();
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

function updateSourceUI() {
  const upload = isUploadMode();
  planLabel.classList.toggle("hidden", upload);
  uploadLabel.classList.toggle("hidden", !upload);
  clearModelOutputs();
  if (upload) {
    uploadId = null;
    images.input.removeAttribute("src");
  } else {
    fileInput.value = "";
    loadInput();
  }
  updateButtons();
}

function loadInput() {
  if (isUploadMode()) {
    if (!uploadId) return;
    images.input.src = cacheBust(`/api/input.png?upload_id=${uploadId}`);
    return;
  }
  images.input.src = cacheBust(`/api/input/${planId()}.png`);
}

function clearModelOutputs() {
  for (const key of [
    "wall",
    "opening",
    "iconHm",
    "room",
    "iconSeg",
    "roomEntropy",
    "iconEntropy",
    "postproc",
  ]) {
    images[key].removeAttribute("src");
  }
}

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
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
    loadInput();
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
    .map(
      (p) =>
        `<option value="${p.id}">#${p.id} — ${p.folder.replace(/^\//, "")}</option>`
    )
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
  planSelect.addEventListener("change", () => {
    loadInput();
    clearModelOutputs();
    updateButtons();
  });
  fileInput.addEventListener("change", handleFileSelect);
  modelSelect.addEventListener("change", () => {
    clearModelOutputs();
    updateButtons();
  });

  runBtn.addEventListener("click", runInference);
  threshold.addEventListener("input", () => {
    thresholdVal.textContent = Number(threshold.value).toFixed(2);
  });
  threshold.addEventListener("change", schedulePostproc);
  threshold.addEventListener("input", () => {
    if (modelId() && images.room.src) schedulePostproc();
  });

  loadInput();
  updateButtons();
}

function updateButtons() {
  const hasModel = Boolean(modelId());
  const hasInput = isUploadMode() ? Boolean(uploadId) : true;
  runBtn.disabled = !hasModel || !hasInput;
  threshold.disabled = !hasModel || !hasInput;
}

async function runInference() {
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

    const q = qs();
    images.wall.src = cacheBust(`/api/artifact/wall_hm.png?${q}`);
    images.opening.src = cacheBust(`/api/artifact/opening_hm.png?${q}`);
    images.iconHm.src = cacheBust(`/api/artifact/icon_hm.png?${q}`);
    images.room.src = cacheBust(`/api/artifact/room_seg.png?${q}`);
    images.iconSeg.src = cacheBust(`/api/artifact/icon_seg.png?${q}`);
    images.roomEntropy.src = cacheBust(`/api/artifact/room_entropy.png?${q}`);
    images.iconEntropy.src = cacheBust(`/api/artifact/icon_entropy.png?${q}`);

    await loadPostproc();
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    updateButtons();
  }
}

function schedulePostproc() {
  if (!modelId() || !images.room.src) return;
  clearTimeout(postprocTimer);
  postprocTimer = setTimeout(loadPostproc, 200);
}

async function loadPostproc() {
  const thr = Number(threshold.value).toFixed(2);
  thresholdVal.textContent = thr;
  setStatus(`Post-processing @ ${thr}…`);
  try {
    images.postproc.src = cacheBust(
      `/api/postproc.png?${qs({ threshold: thr })}`
    );
    const label = isUploadMode()
      ? fileInput.files?.[0]?.name || "upload"
      : planSelect.selectedOptions[0]?.textContent || "";
    setStatus(`Ready — ${label}`);
  } catch (e) {
    setStatus(e.message, true);
  }
}

init().catch((e) => setStatus(e.message, true));
