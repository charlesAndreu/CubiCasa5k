const $ = (id) => document.getElementById(id);

const planSelect = $("planSelect");
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
  postproc: $("imgPostproc"),
};

let postprocTimer = null;

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
  const p = new URLSearchParams({
    plan_id: String(planId()),
    model_id: modelId(),
    ...extra,
  });
  return p.toString();
}

function loadInput() {
  const id = planId();
  images.input.src = cacheBust(`/api/input/${id}.png`);
}

function clearModelOutputs() {
  for (const key of ["wall", "opening", "iconHm", "room", "iconSeg", "postproc"]) {
    images[key].removeAttribute("src");
  }
}

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
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

  planSelect.addEventListener("change", () => {
    loadInput();
    clearModelOutputs();
    updateButtons();
  });

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
  runBtn.disabled = !hasModel;
  threshold.disabled = !hasModel;
}

async function runInference() {
  if (!modelId()) return;
  runBtn.disabled = true;
  setStatus("Running inference (first load may take a minute)…");

  try {
    const meta = await fetchJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan_id: planId(), model_id: modelId() }),
    });
    setStatus(meta.folder || "Done");

    const q = qs();
    images.wall.src = cacheBust(`/api/artifact/wall_hm.png?${q}`);
    images.opening.src = cacheBust(`/api/artifact/opening_hm.png?${q}`);
    images.iconHm.src = cacheBust(`/api/artifact/icon_hm.png?${q}`);
    images.room.src = cacheBust(`/api/artifact/room_seg.png?${q}`);
    images.iconSeg.src = cacheBust(`/api/artifact/icon_seg.png?${q}`);

    await loadPostproc();
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    runBtn.disabled = !modelId();
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
    setStatus(`Ready — ${planSelect.selectedOptions[0]?.textContent || ""}`);
  } catch (e) {
    setStatus(e.message, true);
  }
}

init().catch((e) => setStatus(e.message, true));
