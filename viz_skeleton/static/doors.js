const SVGNS = "http://www.w3.org/2000/svg";
const ZOOM_MIN_SCALE = 1 / 12;
const ZOOM_MAX_SCALE = 1;
const CLICK_DRAG_THRESHOLD_PX = 3;
const DEFAULT_DOOR_WIDTH_PX = 40;
// How far a detected opening may sit from a wall of the EDITED graph and
// still attach to it (fraction of the image width). The openings were
// detected against the original post-process geometry, so a wall that was
// nudged in the editor still has to collect its own doors.
const OPENING_SNAP_IMG_FRACTION = 0.02;
const JAMB_VIEW_FRACTION = 0.004; // half-length of the tick drawn across each door end
const HANDLE_VIEW_FRACTION = 0.006; // radius of the drag-to-resize handle at each door end
const MIN_DOOR_WIDTH_PX = 2;

const svg = document.getElementById("canvas");
const canvasWrap = document.getElementById("canvasWrap");
const statusEl = document.getElementById("status");
const statusSpinner = document.getElementById("statusSpinner");
const selectionCaptionEl = document.getElementById("selectionCaption");

const undoBtn = document.getElementById("undoBtn");
const redoBtn = document.getElementById("redoBtn");
const deleteBtn = document.getElementById("deleteBtn");
const widthInput = document.getElementById("widthInput");
const resetDoorsBtn = document.getElementById("resetDoorsBtn");
const resetViewBtn = document.getElementById("resetViewBtn");
const continueToGeoBtn = document.getElementById("continueToGeoBtn");
const backToEditLink = document.getElementById("backToEditLink");

// --- State ----------------------------------------------------------------
// The wall graph is FIXED here (that's the premise of this step: walls are
// already OK). A door is stored against the wall it belongs to -- edge id,
// `t` = where its centre sits along that edge (0..1), `width` in plan pixels
// -- not as a free-floating pair of coordinates, so it stays glued to its
// wall and its span is always exactly collinear with it. That's what lets the
// export cut it cleanly out of the wall band.
const graph = { points: [], edges: [] };
const pointById = new Map();
let doors = [];
let nextDoorId = 0;
let detectedDoors = null; // what the post-process found, kept for "Reset to detected"
let selection = new Set();
let undoStack = [];
let redoStack = [];
let defaultDoorWidth = DEFAULT_DOOR_WIDTH_PX;

let bgImageEl, wallsLayer, doorsLayer;
let imgW = 0;
let imgH = 0;
let view = { x: 0, y: 0, w: 0, h: 0 };
let query = "";

let dragMode = null; // 'move' | 'pan' | null
let dragData = {};

function setStatus(msg, isError = false) {
  statusEl.textContent = msg || "";
  statusEl.classList.toggle("error", isError);
}

function setBusy(busy) {
  statusSpinner.classList.toggle("hidden", !busy);
}

async function fetchJson(url) {
  const res = await fetch(url);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

// --- Geometry --------------------------------------------------------------

function edgeGeom(edgeId) {
  const edge = graph.edges.find((e) => e.id === edgeId);
  if (!edge) return null;
  const a = pointById.get(edge.a);
  const b = pointById.get(edge.b);
  if (!a || !b) return null;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy);
  if (len < 1e-9) return null;
  return { a, b, dx, dy, len, ux: dx / len, uy: dy / len };
}

// Keeps a door's whole span on its own wall: `t` is its centre, so it can
// travel between half a door from each end. A door wider than the wall it's
// on just centres there rather than being refused -- the wall is what it is,
// and the export's cut is clipped to the wall band anyway.
function clampT(t, width, len) {
  const half = width / 2 / len;
  if (half >= 0.5) return 0.5;
  return Math.min(1 - half, Math.max(half, t));
}

// The door's span along its wall: the two jambs, plus the centre.
function doorSpan(door) {
  const g = edgeGeom(door.edgeId);
  if (!g) return null;
  const cx = g.a.x + g.ux * door.t * g.len;
  const cy = g.a.y + g.uy * door.t * g.len;
  const h = door.width / 2;
  return {
    cx,
    cy,
    ux: g.ux,
    uy: g.uy,
    x1: cx - g.ux * h,
    y1: cy - g.uy * h,
    x2: cx + g.ux * h,
    y2: cy + g.uy * h,
  };
}

// Drag one jamb, keep the other one still: the door's width and its centre
// both follow from where the two ends are, so this works in distance ALONG
// the wall (0..len) rather than in the stored centre/width pair. The dragged
// end is clamped to the wall's own extent and can't cross the fixed end.
function resizeDoorTo(door, end, distanceAlongWall) {
  const g = edgeGeom(door.edgeId);
  if (!g) return;
  const centre = door.t * g.len;
  const half = door.width / 2;
  const fixed = end === "a" ? centre + half : centre - half;
  let moved = Math.min(g.len, Math.max(0, distanceAlongWall));
  if (end === "a") moved = Math.min(moved, fixed - MIN_DOOR_WIDTH_PX);
  else moved = Math.max(moved, fixed + MIN_DOOR_WIDTH_PX);
  door.width = Math.abs(fixed - moved);
  door.t = (fixed + moved) / 2 / g.len;
}

// The wall nearest (x, y), with where along it the point falls -- used both
// to drop a new door on a wall and to keep a dragged one on one.
function nearestEdgeAt(x, y) {
  let best = null;
  graph.edges.forEach((edge) => {
    const g = edgeGeom(edge.id);
    if (!g) return;
    const raw = ((x - g.a.x) * g.dx + (y - g.a.y) * g.dy) / (g.len * g.len);
    const t = Math.min(1, Math.max(0, raw));
    const dist = Math.hypot(x - (g.a.x + t * g.dx), y - (g.a.y + t * g.dy));
    if (!best || dist < best.dist) best = { edgeId: edge.id, t, dist, len: g.len };
  });
  return best;
}

function getDoor(id) {
  return doors.find((d) => d.id === id);
}

// --- History ---------------------------------------------------------------

function snapshot() {
  return doors.map((d) => ({ ...d }));
}

function restore(snap) {
  doors = snap.map((d) => ({ ...d }));
}

function commitIfChanged(before) {
  const after = snapshot();
  if (JSON.stringify(before) === JSON.stringify(after)) {
    render();
    return;
  }
  undoStack.push(before);
  if (undoStack.length > 100) undoStack.shift();
  redoStack = [];
  render();
}

function undo() {
  if (!undoStack.length) return;
  const prev = undoStack.pop();
  redoStack.push(snapshot());
  restore(prev);
  selection.clear();
  render();
}

function redo() {
  if (!redoStack.length) return;
  const next = redoStack.pop();
  undoStack.push(snapshot());
  restore(next);
  selection.clear();
  render();
}

// --- View (pan/zoom via SVG viewBox) ---------------------------------------

function applyView() {
  svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.h}`);
}

function resetView() {
  if (!imgW) return;
  view = { x: 0, y: 0, w: imgW, h: imgH };
  applyView();
  render();
}

function svgFromClient(clientX, clientY) {
  const pt = svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}

function svgFromClientUsingMatrix(clientX, clientY, invMatrix) {
  const pt = svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  return pt.matrixTransform(invMatrix);
}

svg.addEventListener(
  "wheel",
  (e) => {
    if (!imgW) return;
    e.preventDefault();
    const cursor = svgFromClient(e.clientX, e.clientY);
    const factor = e.deltaY < 0 ? 1 / 1.15 : 1.15;
    const curScale = view.w / imgW;
    const newScale = Math.min(ZOOM_MAX_SCALE, Math.max(ZOOM_MIN_SCALE, curScale * factor));
    if (newScale === curScale) return;
    const fracX = (cursor.x - view.x) / view.w;
    const fracY = (cursor.y - view.y) / view.h;
    view.w = imgW * newScale;
    view.h = imgH * newScale;
    view.x = cursor.x - fracX * view.w;
    view.y = cursor.y - fracY * view.h;
    applyView();
    render(); // the jamb ticks are sized in view units, so they follow the zoom
  },
  { passive: false }
);

// --- Rendering -------------------------------------------------------------

function buildSvgSkeleton() {
  svg.innerHTML = "";

  bgImageEl = document.createElementNS(SVGNS, "image");
  bgImageEl.setAttribute("x", 0);
  bgImageEl.setAttribute("y", 0);
  bgImageEl.setAttribute("width", imgW);
  bgImageEl.setAttribute("height", imgH);
  bgImageEl.setAttribute("preserveAspectRatio", "none");
  svg.appendChild(bgImageEl);

  wallsLayer = document.createElementNS(SVGNS, "g");
  svg.appendChild(wallsLayer);

  doorsLayer = document.createElementNS(SVGNS, "g");
  svg.appendChild(doorsLayer);

  // Walls never change on this page, so they're drawn once: a wide invisible
  // hit line (the double-click target that adds a door) under the visible one.
  graph.edges.forEach((edge) => {
    const g = edgeGeom(edge.id);
    if (!g) return;
    ["gw-wall-hit", "gw-wall"].forEach((cls) => {
      const line = document.createElementNS(SVGNS, "line");
      line.setAttribute("class", cls);
      line.setAttribute("x1", g.a.x);
      line.setAttribute("y1", g.a.y);
      line.setAttribute("x2", g.b.x);
      line.setAttribute("y2", g.b.y);
      line.dataset.edgeId = edge.id;
      wallsLayer.appendChild(line);
    });
  });
}

function render() {
  doorsLayer.innerHTML = "";
  const jamb = Math.max(1, view.w * JAMB_VIEW_FRACTION);
  const handleR = Math.max(2, view.w * HANDLE_VIEW_FRACTION);

  doors.forEach((door) => {
    const span = doorSpan(door);
    if (!span) return;
    const selected = selection.has(door.id);

    const hit = document.createElementNS(SVGNS, "line");
    hit.setAttribute("class", "gw-door-hit");
    hit.setAttribute("x1", span.x1);
    hit.setAttribute("y1", span.y1);
    hit.setAttribute("x2", span.x2);
    hit.setAttribute("y2", span.y2);
    hit.dataset.doorId = door.id;
    doorsLayer.appendChild(hit);

    const line = document.createElementNS(SVGNS, "line");
    line.setAttribute("class", "gw-door" + (selected ? " selected" : ""));
    line.setAttribute("x1", span.x1);
    line.setAttribute("y1", span.y1);
    line.setAttribute("x2", span.x2);
    line.setAttribute("y2", span.y2);
    line.dataset.doorId = door.id;
    doorsLayer.appendChild(line);

    // A tick across each end, so a narrow door still reads as an opening
    // with two jambs instead of a blob.
    [
      ["a", span.x1, span.y1],
      ["b", span.x2, span.y2],
    ].forEach(([end, x, y]) => {
      const tick = document.createElementNS(SVGNS, "line");
      tick.setAttribute("class", "gw-door-jamb" + (selected ? " selected" : ""));
      tick.setAttribute("x1", x - span.uy * jamb);
      tick.setAttribute("y1", y + span.ux * jamb);
      tick.setAttribute("x2", x + span.uy * jamb);
      tick.setAttribute("y2", y - span.ux * jamb);
      doorsLayer.appendChild(tick);

      // Resize handles, on the selected door only -- otherwise every door on
      // the plan sprouts two grab targets and moving one becomes a gamble.
      if (!selected) return;
      const handle = document.createElementNS(SVGNS, "circle");
      handle.setAttribute("class", "gw-door-handle");
      handle.setAttribute("cx", x);
      handle.setAttribute("cy", y);
      handle.setAttribute("r", handleR);
      handle.dataset.doorId = door.id;
      handle.dataset.end = end;
      doorsLayer.appendChild(handle);
    });
  });

  updateSelectionCaption();
  updateToolbarState();
}

function updateSelectionCaption() {
  const total = `${doors.length} door${doors.length === 1 ? "" : "s"}`;
  selectionCaptionEl.textContent =
    selection.size === 0 ? `${total} — no selection` : `${total} — ${selection.size} selected`;
}

function updateToolbarState() {
  undoBtn.disabled = undoStack.length === 0;
  redoBtn.disabled = redoStack.length === 0;
  deleteBtn.disabled = selection.size === 0;
  resetDoorsBtn.disabled = !detectedDoors;
  resetViewBtn.disabled = imgW === 0;
  continueToGeoBtn.disabled = graph.edges.length === 0;
}

// --- Editing ---------------------------------------------------------------

function addDoorAt(x, y) {
  const near = nearestEdgeAt(x, y);
  if (!near) return;
  const before = snapshot();
  const door = {
    id: nextDoorId++,
    edgeId: near.edgeId,
    t: clampT(near.t, defaultDoorWidth, near.len),
    width: defaultDoorWidth,
  };
  doors.push(door);
  selection.clear();
  selection.add(door.id);
  commitIfChanged(before);
}

function deleteSelected() {
  if (selection.size === 0) return;
  const before = snapshot();
  doors = doors.filter((d) => !selection.has(d.id));
  selection.clear();
  commitIfChanged(before);
}

function applyWidthToSelection(width) {
  if (!Number.isFinite(width) || width < 1 || selection.size === 0) return;
  const before = snapshot();
  doors.forEach((d) => {
    if (!selection.has(d.id)) return;
    const g = edgeGeom(d.edgeId);
    d.width = width;
    if (g) d.t = clampT(d.t, width, g.len);
  });
  commitIfChanged(before);
}

// --- Mouse -----------------------------------------------------------------

function movedEnough(e) {
  return (
    Math.abs(e.clientX - dragData.startClientX) > CLICK_DRAG_THRESHOLD_PX ||
    Math.abs(e.clientY - dragData.startClientY) > CLICK_DRAG_THRESHOLD_PX
  );
}

svg.addEventListener("mousedown", (e) => {
  if (e.button !== 0 || !imgW) return;
  const svgPos = svgFromClient(e.clientX, e.clientY);
  const handleEl = e.target.closest(".gw-door-handle");
  const doorEl = e.target.closest("[data-door-id]");

  // Checked before the door body: a handle carries a door id too, and
  // grabbing one has to resize rather than move.
  if (handleEl) {
    dragMode = "resize";
    dragData = {
      doorId: Number(handleEl.dataset.doorId),
      end: handleEl.dataset.end,
      beforeSnapshot: snapshot(),
      startClientX: e.clientX,
      startClientY: e.clientY,
    };
    return;
  }

  if (doorEl) {
    const id = Number(doorEl.dataset.doorId);
    if (e.shiftKey) {
      if (selection.has(id)) selection.delete(id);
      else selection.add(id);
      render();
      return;
    }
    if (!selection.has(id)) {
      selection.clear();
      selection.add(id);
    }
    dragMode = "move";
    dragData = {
      doorId: id,
      beforeSnapshot: snapshot(),
      startClientX: e.clientX,
      startClientY: e.clientY,
    };
    render();
    return;
  }

  dragMode = "pan";
  dragData = {
    startClientX: e.clientX,
    startClientY: e.clientY,
    startView: { ...view },
    invCtm0: svg.getScreenCTM().inverse(),
    startSvg: svgPos,
  };
  canvasWrap.classList.add("panning");
});

window.addEventListener("mousemove", (e) => {
  if (!dragMode) return;

  if (dragMode === "move") {
    // The cursor doesn't place the door directly: it picks the nearest wall
    // and the position along it, so a door can be slid along its wall or
    // dropped onto a different one, but never ends up floating off the walls.
    const svgPos = svgFromClient(e.clientX, e.clientY);
    const near = nearestEdgeAt(svgPos.x, svgPos.y);
    const door = getDoor(dragData.doorId);
    if (near && door) {
      door.edgeId = near.edgeId;
      door.t = clampT(near.t, door.width, near.len);
      render();
    }
    return;
  }

  if (dragMode === "resize") {
    const svgPos = svgFromClient(e.clientX, e.clientY);
    const door = getDoor(dragData.doorId);
    const g = door && edgeGeom(door.edgeId);
    if (door && g) {
      // Projected onto the door's OWN wall, so dragging sideways off the wall
      // slides the jamb along it instead of snapping the door elsewhere.
      const along = (svgPos.x - g.a.x) * g.ux + (svgPos.y - g.a.y) * g.uy;
      resizeDoorTo(door, dragData.end, along);
      defaultDoorWidth = Math.round(door.width);
      widthInput.value = defaultDoorWidth;
      render();
    }
    return;
  }

  if (dragMode === "pan") {
    const cur = svgFromClientUsingMatrix(e.clientX, e.clientY, dragData.invCtm0);
    const start = svgFromClientUsingMatrix(dragData.startClientX, dragData.startClientY, dragData.invCtm0);
    view.x = dragData.startView.x - (cur.x - start.x);
    view.y = dragData.startView.y - (cur.y - start.y);
    applyView();
  }
});

window.addEventListener("mouseup", (e) => {
  if (!dragMode) return;
  const mode = dragMode;
  dragMode = null;
  canvasWrap.classList.remove("panning");

  if (mode === "move" || mode === "resize") {
    commitIfChanged(dragData.beforeSnapshot);
  } else if (mode === "pan" && !movedEnough(e) && selection.size) {
    selection.clear();
    render();
  }
});

svg.addEventListener("dblclick", (e) => {
  if (e.target.closest("[data-door-id]")) return; // don't drop a door on top of a door
  const wallEl = e.target.closest("[data-edge-id]");
  if (wallEl) {
    const svgPos = svgFromClient(e.clientX, e.clientY);
    addDoorAt(svgPos.x, svgPos.y);
    return;
  }
  resetView();
});

window.addEventListener("keydown", (e) => {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;

  if (e.key === "Delete" || e.key === "Backspace") {
    e.preventDefault();
    deleteSelected();
    return;
  }
  if (e.key.toLowerCase() === "z" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    if (e.shiftKey) redo();
    else undo();
    return;
  }
  if (e.key.toLowerCase() === "y" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    redo();
    return;
  }
  if (e.key === "Escape" && selection.size) {
    selection.clear();
    render();
  }
});

// --- Toolbar ---------------------------------------------------------------

undoBtn.addEventListener("click", undo);
redoBtn.addEventListener("click", redo);
deleteBtn.addEventListener("click", deleteSelected);
resetViewBtn.addEventListener("click", resetView);

widthInput.addEventListener("change", () => {
  const width = Number(widthInput.value);
  if (!Number.isFinite(width) || width < 1) return;
  defaultDoorWidth = width;
  applyWidthToSelection(width);
});

resetDoorsBtn.addEventListener("click", () => {
  if (!detectedDoors) return;
  const before = snapshot();
  restore(detectedDoors);
  selection.clear();
  commitIfChanged(before);
});

continueToGeoBtn.addEventListener("click", () => {
  // Doors are handed on as plain spans in plan-pixel space -- the two ends of
  // the opening along its wall -- next to the graph itself. The export step
  // has no use for which edge a door belongs to; it just needs to know what
  // to cut out of the wall band, and a span is exactly that. Same key and
  // same mechanism as the wall editor's own handoff.
  const spans = doors
    .map((d) => doorSpan(d))
    .filter(Boolean)
    .map((s) => [
      [s.x1, s.y1],
      [s.x2, s.y2],
    ]);
  try {
    sessionStorage.setItem(
      "cubi-geo-graph",
      JSON.stringify({
        points: graph.points.map((p) => ({ ...p })),
        edges: graph.edges.map((e) => ({ ...e })),
        doors: spans,
        doorDefs: doors.map((d) => ({ ...d })), // so coming back here restores the edit, not the detection
      })
    );
  } catch (err) {
    setStatus("Could not hand off the doors to the export step: " + err.message, true);
    return;
  }
  window.location.href = "/geo";
});

// --- Loading ---------------------------------------------------------------

function loadBackgroundImage(q) {
  return new Promise((resolve, reject) => {
    const url = `/api/base.png?${q}`;
    const img = new Image();
    img.onload = () => {
      imgW = img.naturalWidth;
      imgH = img.naturalHeight;
      resolve(url);
    };
    img.onerror = () => reject(new Error("Failed to load base image"));
    img.src = url;
  });
}

// The post-process reports an opening as the gap between two detected opening
// points ON one of ITS wall segments. Those segments are the pre-editing ones,
// so rather than trusting the edge index, each gap is re-attached to whichever
// wall of the EDITED graph its midpoint is closest to -- which also drops the
// openings whose wall was deleted in the editor, instead of leaving them
// floating in the middle of nothing.
//
// Only the openings the model called doors are kept. The wall post-process
// can't tell a door from a window (one opening class), but the model's icon
// head can, and /api/openings.json reads the label off it -- so windows stay
// out of here instead of being cut out of the walls on export.
function doorsFromOpenings(openings) {
  const tolerance = Math.max(8, imgW * OPENING_SNAP_IMG_FRACTION);
  const seen = new Set();
  const out = [];
  openings.forEach((op) => {
    if (op.kind !== "door") return;
    const gap = op.gap;
    if (!gap || gap.length < 2) return;
    const [[x1, y1], [x2, y2]] = gap;
    const width = Math.hypot(x2 - x1, y2 - y1);
    if (width < 1) return;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const near = nearestEdgeAt(cx, cy);
    if (!near || near.dist > tolerance) return;
    // attach_openings pairs points per wall segment, so two collinear
    // segments can report the same opening twice.
    const key = `${near.edgeId}_${Math.round(cx)}_${Math.round(cy)}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push({
      id: nextDoorId++,
      edgeId: near.edgeId,
      t: clampT(near.t, width, near.len),
      width,
    });
  });
  return out;
}

function medianWidth(list) {
  if (!list.length) return DEFAULT_DOOR_WIDTH_PX;
  const sorted = list.map((d) => d.width).sort((a, b) => a - b);
  return Math.round(sorted[Math.floor(sorted.length / 2)]);
}

async function init() {
  query = window.location.search.replace(/^\?/, "");
  backToEditLink.href = query ? `/edit?${query}` : "/edit";
  if (!query) {
    setStatus("Missing model/plan parameters — open this page from the wall editor.", true);
    return;
  }

  let handoff = null;
  try {
    handoff = JSON.parse(sessionStorage.getItem("cubi-geo-graph") || "null");
  } catch (e) {
    handoff = null;
  }
  if (!handoff || !handoff.points || !handoff.points.length) {
    setStatus('No wall graph found — open this page via the wall editor\'s "Continue" button.', true);
    return;
  }
  graph.points = handoff.points;
  graph.edges = handoff.edges || [];
  pointById.clear();
  graph.points.forEach((p) => pointById.set(p.id, p));

  setStatus("Loading detected doors…");
  setBusy(true);
  try {
    const [detected, imgUrl] = await Promise.all([
      fetchJson(`/api/openings.json?${query}`),
      loadBackgroundImage(query),
    ]);
    const openings = detected.openings || [];
    const skipped = openings.filter((op) => op.kind !== "door");
    detectedDoors = doorsFromOpenings(openings);

    // Coming back from the export step restores the doors as they were left,
    // not the raw detection.
    const previous = Array.isArray(handoff.doorDefs) ? handoff.doorDefs : null;
    doors = previous ? previous.map((d) => ({ ...d })) : detectedDoors.map((d) => ({ ...d }));
    nextDoorId = Math.max(nextDoorId, ...doors.map((d) => d.id + 1), 0);

    defaultDoorWidth = medianWidth(doors.length ? doors : detectedDoors);
    widthInput.value = defaultDoorWidth;

    buildSvgSkeleton();
    bgImageEl.setAttributeNS("http://www.w3.org/1999/xlink", "href", imgUrl);
    bgImageEl.setAttribute("href", imgUrl);
    view = { x: 0, y: 0, w: imgW, h: imgH };
    applyView();
    render();
    const windows = skipped.filter((op) => op.kind === "window").length;
    const unknown = skipped.length - windows;
    setStatus(
      `${doors.length} door${doors.length === 1 ? "" : "s"} on ${graph.edges.length} walls` +
        (previous ? " (restored)" : " (detected)") +
        (skipped.length ? ` — ignored ${windows} window${windows === 1 ? "" : "s"}` +
          (unknown ? ` and ${unknown} unclassified opening${unknown === 1 ? "" : "s"}` : "") : "")
    );
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    setBusy(false);
    updateToolbarState();
  }
}

init();
