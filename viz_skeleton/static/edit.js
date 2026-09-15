const SVGNS = "http://www.w3.org/2000/svg";
const CLICK_DRAG_THRESHOLD_PX = 3;
const ZOOM_MIN_SCALE = 1 / 12;
const ZOOM_MAX_SCALE = 1;
const ALIGN_SNAP_VIEW_FRACTION = 0.01; // point-to-point x/y alignment tolerance, relative to view width
const ANGLE_SNAP_DEG = 4; // edge-angle tolerance around each 45deg step (so 90deg corners included)
const ANGLE_STEP_DEG = 45;
const ALIGN_SELECT_VIEW_FRACTION = 0.006; // alt-double-click-on-edge "aligned points" perpendicular-distance delta
const NUDGE_PX = 1;
const NUDGE_COARSE_PX = 10; // shift+arrow, the usual 10x step

const svg = document.getElementById("canvas");
const canvasWrap = document.getElementById("canvasWrap");
const statusEl = document.getElementById("status");
const statusSpinner = document.getElementById("statusSpinner");
const selectionCaptionEl = document.getElementById("selectionCaption");

const undoBtn = document.getElementById("undoBtn");
const redoBtn = document.getElementById("redoBtn");
const deleteBtn = document.getElementById("deleteBtn");
const resetGraphBtn = document.getElementById("resetGraphBtn");
const dlEditedJsonBtn = document.getElementById("dlEditedJson");
const resetViewBtn = document.getElementById("resetViewBtn");
const magnetismBtn = document.getElementById("magnetismBtn");
const continueToGeoBtn = document.getElementById("continueToGeoBtn");
const helpBtn = document.getElementById("helpBtn");

// --- State: points/edges reference each other by stable integer id, not array
// index, so deletions/fusions never have to renumber anything they don't touch. ---
const state = { points: [], edges: [] };
let selection = new Set();
let fuseTargetId = null;
let nextPointId = 0;
let nextEdgeId = 0;
let originalSnapshot = null;
let undoStack = [];
let redoStack = [];
let magnetismEnabled = false;
try {
  magnetismEnabled = localStorage.getItem("cubi-edit-magnetism") === "1";
} catch (e) {
  // localStorage unavailable (e.g. private-browsing restrictions) -- default off.
}

let bgImageEl, edgesLayer, pointsLayer, newEdgePreviewEl, marqueeEl, guideXEl, guideYEl;
let imgW = 0;
let imgH = 0;
let view = { x: 0, y: 0, w: 0, h: 0 };

let dragMode = null; // 'move' | 'pan' | 'marquee' | 'new-edge' | null
let dragData = {};
let spaceHeld = false; // space = temporary hand tool, as in every design app

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

// --- Graph helpers -----------------------------------------------------

function getPoint(id) {
  return state.points.find((p) => p.id === id);
}

function nearestPoint(x, y, excludeId) {
  let best = null;
  let bestDist = Infinity;
  for (const p of state.points) {
    if (p.id === excludeId) continue;
    const d = Math.hypot(p.x - x, p.y - y);
    if (d < bestDist) {
      bestDist = d;
      best = p;
    }
  }
  return best ? { id: best.id, dist: bestDist } : null;
}

// Perpendicular distance from (px,py) to the infinite line through (x1,y1)-(x2,y2).
function pointLineDistance(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len < 1e-9) return Math.hypot(px - x1, py - y1);
  return Math.abs((px - x1) * dy - (py - y1) * dx) / len;
}

// Double-click-on-edge selection: grows outward from the clicked edge along
// actual edge connections only (BFS), not by scanning every point in the
// plan for geometric alignment -- a point that merely happens to sit on the
// same infinite line elsewhere in the building, with no path of edges back
// to the clicked one, is never included. A candidate is added only once it's
// reached from an already-selected point via a real edge, and only if it's
// still within `tol` of the ORIGINAL clicked edge's line.
function alignedConnectedPoints(edge, tol) {
  const a = getPoint(edge.a);
  const b = getPoint(edge.b);
  const visited = new Set([edge.a, edge.b]);
  const queue = [edge.a, edge.b];
  while (queue.length) {
    const pid = queue.shift();
    state.edges.forEach((ed) => {
      const otherId = ed.a === pid ? ed.b : ed.b === pid ? ed.a : null;
      if (otherId == null || visited.has(otherId)) return;
      const other = getPoint(otherId);
      if (other && pointLineDistance(other.x, other.y, a.x, a.y, b.x, b.y) <= tol) {
        visited.add(otherId);
        queue.push(otherId);
      }
    });
  }
  return visited;
}

function angleDiffDeg(a, b) {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

// Group drags (multiple selected points moved together) don't have a single
// "edge to a neighbor" to snap an angle against the way a single-point drag
// does -- instead, magnetism here snaps the drag's own direction onto
// horizontal or vertical when it's already close, so nudging a whole selected
// wall run sideways doesn't accidentally skew it a degree or two off-axis.
function snapAxisDelta(dx, dy) {
  if (!magnetismEnabled) return { dx, dy };
  const dist = Math.hypot(dx, dy);
  if (dist < 1e-6) return { dx, dy };
  const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
  const step = Math.round(angle / 90) * 90;
  if (angleDiffDeg(angle, step) > ANGLE_SNAP_DEG) return { dx, dy };
  const rad = (step * Math.PI) / 180;
  return { dx: dist * Math.cos(rad), dy: dist * Math.sin(rad) };
}

// Snaps a tentative point position to (a) line up with any other existing
// point's x or y (classic alignment guides) and (b) put each edge to a fixed
// neighbor exactly on a 45deg step (0/45/90/.../315 -- so cardinal AND
// perpendicular corners are both covered, since two edges each independently
// snapped to a cardinal direction are automatically perpendicular to each
// other). Returns { x, y, guideX, guideY } -- guideX/guideY are the matched
// coordinate to draw an alignment guide line at, or null when that axis didn't
// snap. No-op (guides null) when magnetism is off.
function applyMagnetism(pos, excludeId, neighborIds) {
  if (!magnetismEnabled) return { x: pos.x, y: pos.y, guideX: null, guideY: null };

  let x = pos.x;
  let y = pos.y;
  let guideX = null;
  let guideY = null;
  const tol = Math.max(2, view.w * ALIGN_SNAP_VIEW_FRACTION);
  let bestDX = tol;
  let bestDY = tol;
  for (const p of state.points) {
    if (p.id === excludeId) continue;
    const dx = Math.abs(p.x - pos.x);
    if (dx < bestDX) {
      bestDX = dx;
      guideX = p.x;
    }
    const dy = Math.abs(p.y - pos.y);
    if (dy < bestDY) {
      bestDY = dy;
      guideY = p.y;
    }
  }
  if (guideX != null) x = guideX;
  if (guideY != null) y = guideY;

  for (const nid of neighborIds || []) {
    const n = getPoint(nid);
    if (!n) continue;
    const dx = x - n.x;
    const dy = y - n.y;
    const dist = Math.hypot(dx, dy);
    if (dist < 1e-6) continue;
    const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
    const step = Math.round(angle / ANGLE_STEP_DEG) * ANGLE_STEP_DEG;
    if (angleDiffDeg(angle, step) <= ANGLE_SNAP_DEG) {
      const rad = (step * Math.PI) / 180;
      x = n.x + dist * Math.cos(rad);
      y = n.y + dist * Math.sin(rad);
    }
  }

  return { x, y, guideX, guideY };
}

function showGuides(guideX, guideY) {
  if (guideX != null) {
    guideXEl.setAttribute("x1", guideX);
    guideXEl.setAttribute("x2", guideX);
    guideXEl.setAttribute("y1", 0);
    guideXEl.setAttribute("y2", imgH);
    guideXEl.style.display = "";
  } else {
    guideXEl.style.display = "none";
  }
  if (guideY != null) {
    guideYEl.setAttribute("y1", guideY);
    guideYEl.setAttribute("y2", guideY);
    guideYEl.setAttribute("x1", 0);
    guideYEl.setAttribute("x2", imgW);
    guideYEl.style.display = "";
  } else {
    guideYEl.style.display = "none";
  }
}

function hideGuides() {
  guideXEl.style.display = "none";
  guideYEl.style.display = "none";
}

function addPoint(x, y, arity = 2, conf = 1.0) {
  const p = { id: nextPointId++, x, y, arity, conf };
  state.points.push(p);
  return p;
}

function addEdge(aId, bId) {
  if (aId === bId) return null;
  const exists = state.edges.some(
    (ed) => (ed.a === aId && ed.b === bId) || (ed.a === bId && ed.b === aId)
  );
  if (exists) return null;
  const e = { id: nextEdgeId++, a: aId, b: bId };
  state.edges.push(e);
  return e;
}

// Splits a wall at `pos` (projected onto the wall, so the new point lands ON
// it rather than wherever the cursor happened to be) and replaces it with the
// two halves. The standard double-click-a-segment gesture of every vertex
// editor; the new point comes out selected, ready to be dragged.
function insertPointOnEdge(edge, pos) {
  const a = getPoint(edge.a);
  const b = getPoint(edge.b);
  if (!a || !b) return;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 1e-9) return;
  const t = Math.min(1, Math.max(0, ((pos.x - a.x) * dx + (pos.y - a.y) * dy) / lenSq));
  const before = snapshot();
  const p = addPoint(a.x + t * dx, a.y + t * dy, 2, 1.0);
  state.edges = state.edges.filter((ed) => ed.id !== edge.id);
  addEdge(edge.a, p.id);
  addEdge(p.id, edge.b);
  selection.clear();
  selection.add(p.id);
  commitIfChanged(before);
}

function fusePoints(targetId, mergeId) {
  if (targetId === mergeId) return;
  state.edges.forEach((ed) => {
    if (ed.a === mergeId) ed.a = targetId;
    if (ed.b === mergeId) ed.b = targetId;
  });
  const seen = new Set();
  state.edges = state.edges.filter((ed) => {
    if (ed.a === ed.b) return false;
    const key = ed.a < ed.b ? `${ed.a}_${ed.b}` : `${ed.b}_${ed.a}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  state.points = state.points.filter((p) => p.id !== mergeId);
}

// Deleting a selection removes the segments *between* selected points, then
// drops only those selected points left with nothing attached. A point the
// deleted line shares with the rest of the plan -- a wall junction, a corner
// where another wall continues -- still has an edge that survives the action,
// so it stays, and the walls hanging off it aren't shredded along with the
// one being removed.
//
// A selected point with no selected neighbour has no segment of its own to
// remove, so there the only possible intent is "delete this point": its
// incident edges go with it, as they always did.
function deleteSelected() {
  if (selection.size === 0) return;
  const before = snapshot();
  const ids = new Set(selection);

  const hasSelectedEdge = new Set();
  state.edges.forEach((ed) => {
    if (ids.has(ed.a) && ids.has(ed.b)) {
      hasSelectedEdge.add(ed.a);
      hasSelectedEdge.add(ed.b);
    }
  });
  const isLone = (id) => ids.has(id) && !hasSelectedEdge.has(id);

  state.edges = state.edges.filter((ed) => {
    if (ids.has(ed.a) && ids.has(ed.b)) return false;
    return !isLone(ed.a) && !isLone(ed.b);
  });

  const stillConnected = new Set();
  state.edges.forEach((ed) => {
    stillConnected.add(ed.a);
    stillConnected.add(ed.b);
  });
  state.points = state.points.filter((p) => !ids.has(p.id) || stillConnected.has(p.id));

  selection.clear();
  commitIfChanged(before);
}

// --- History -------------------------------------------------------------

function snapshot() {
  return {
    points: state.points.map((p) => ({ ...p })),
    edges: state.edges.map((e) => ({ ...e })),
  };
}

function restore(snap) {
  state.points = snap.points.map((p) => ({ ...p }));
  state.edges = snap.edges.map((e) => ({ ...e }));
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

// --- View (pan/zoom via SVG viewBox) --------------------------------------

function applyView() {
  svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.h}`);
}

function resetView() {
  if (!imgW) return;
  view = { x: 0, y: 0, w: imgW, h: imgH };
  applyView();
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

// Zooms by `factor` about a fixed point -- the cursor for the wheel, the
// middle of the view for the keyboard. render() afterwards because point
// radius is derived from view.w: without it, zooming in leaves the circles at
// their old size in user units and they balloon on screen.
function zoomBy(factor, about) {
  if (!imgW) return;
  const curScale = view.w / imgW;
  const newScale = Math.min(ZOOM_MAX_SCALE, Math.max(ZOOM_MIN_SCALE, curScale * factor));
  if (newScale === curScale) return;
  const cx = about ? about.x : view.x + view.w / 2;
  const cy = about ? about.y : view.y + view.h / 2;
  const fracX = (cx - view.x) / view.w;
  const fracY = (cy - view.y) / view.h;
  view.w = imgW * newScale;
  view.h = imgH * newScale;
  view.x = cx - fracX * view.w;
  view.y = cy - fracY * view.h;
  applyView();
  render();
}

svg.addEventListener(
  "wheel",
  (e) => {
    if (!imgW) return;
    e.preventDefault();
    zoomBy(e.deltaY < 0 ? 1 / 1.15 : 1.15, svgFromClient(e.clientX, e.clientY));
  },
  { passive: false }
);

svg.addEventListener("dblclick", (e) => {
  if (e.target.closest(".gw-point")) return; // avoid an accidental view reset on a point

  const edgeEl = e.target.closest("[data-edge-id]");
  if (edgeEl) {
    const edge = state.edges.find((ed) => ed.id === Number(edgeEl.dataset.edgeId));
    if (!edge) return;
    // Alt+double-click keeps the wall-specific gesture: select the whole
    // aligned run this wall belongs to. Plain double-click does what a
    // double-click on a segment does everywhere else -- insert a point there.
    if (e.altKey) {
      const tol = Math.max(2, view.w * ALIGN_SELECT_VIEW_FRACTION);
      if (!e.shiftKey) selection.clear();
      alignedConnectedPoints(edge, tol).forEach((id) => selection.add(id));
      render();
      return;
    }
    insertPointOnEdge(edge, svgFromClient(e.clientX, e.clientY));
    return;
  }

  resetView();
});

// --- Rendering -------------------------------------------------------------

function buildSvgSkeleton() {
  svg.innerHTML = "";
  edgeEls.clear();
  pointEls.clear();

  bgImageEl = document.createElementNS(SVGNS, "image");
  bgImageEl.setAttribute("x", 0);
  bgImageEl.setAttribute("y", 0);
  bgImageEl.setAttribute("width", imgW);
  bgImageEl.setAttribute("height", imgH);
  bgImageEl.setAttribute("preserveAspectRatio", "none");
  svg.appendChild(bgImageEl);

  edgesLayer = document.createElementNS(SVGNS, "g");
  svg.appendChild(edgesLayer);

  pointsLayer = document.createElementNS(SVGNS, "g");
  svg.appendChild(pointsLayer);

  guideXEl = document.createElementNS(SVGNS, "line");
  guideXEl.setAttribute("class", "gw-guide");
  guideXEl.style.display = "none";
  svg.appendChild(guideXEl);

  guideYEl = document.createElementNS(SVGNS, "line");
  guideYEl.setAttribute("class", "gw-guide");
  guideYEl.style.display = "none";
  svg.appendChild(guideYEl);

  newEdgePreviewEl = document.createElementNS(SVGNS, "line");
  newEdgePreviewEl.setAttribute("class", "gw-new-edge-preview");
  newEdgePreviewEl.style.display = "none";
  svg.appendChild(newEdgePreviewEl);

  marqueeEl = document.createElementNS(SVGNS, "rect");
  marqueeEl.setAttribute("class", "gw-marquee");
  marqueeEl.style.display = "none";
  svg.appendChild(marqueeEl);
}

// render() reconciles by id (persistent DOM elements, reused across calls)
// rather than tearing down and rebuilding everything -- NOT just an
// optimization: replacing the element under the cursor between the two clicks
// of a double-click (which a naive innerHTML="" + rebuild does on every
// selection change) makes Chromium refuse to synthesize a native 'dblclick'
// event at all, breaking double-click-to-select-aligned-points entirely.
// Elements untouched by whatever changed keep their identity.
const edgeEls = new Map(); // edge id -> { hit, line }
const pointEls = new Map(); // point id -> circle

function render() {
  const currentEdgeIds = new Set(state.edges.map((ed) => ed.id));
  for (const [id, els] of edgeEls) {
    if (!currentEdgeIds.has(id)) {
      els.hit.remove();
      els.line.remove();
      edgeEls.delete(id);
    }
  }
  state.edges.forEach((ed) => {
    const a = getPoint(ed.a);
    const b = getPoint(ed.b);
    if (!a || !b) return;
    const selected = selection.has(ed.a) && selection.has(ed.b);

    let els = edgeEls.get(ed.id);
    if (!els) {
      const hit = document.createElementNS(SVGNS, "line");
      hit.setAttribute("class", "gw-edge-hit");
      hit.dataset.edgeId = ed.id;
      edgesLayer.appendChild(hit);

      // The visible line is drawn on top of (after) the wider invisible hit
      // line, so a click landing on the visible stroke itself -- the most
      // likely place to click -- hits *this* element, not .gw-edge-hit. It
      // needs its own edgeId so hit-testing (which matches on [data-edge-id],
      // not a specific class) finds the edge regardless of which of the two
      // lines the click actually landed on.
      const line = document.createElementNS(SVGNS, "line");
      line.dataset.edgeId = ed.id;
      edgesLayer.appendChild(line);

      els = { hit, line };
      edgeEls.set(ed.id, els);
    }
    els.hit.setAttribute("x1", a.x);
    els.hit.setAttribute("y1", a.y);
    els.hit.setAttribute("x2", b.x);
    els.hit.setAttribute("y2", b.y);
    els.line.setAttribute("x1", a.x);
    els.line.setAttribute("y1", a.y);
    els.line.setAttribute("x2", b.x);
    els.line.setAttribute("y2", b.y);
    els.line.setAttribute("class", "gw-edge" + (selected ? " selected" : ""));
  });

  const currentPointIds = new Set(state.points.map((p) => p.id));
  for (const [id, el] of pointEls) {
    if (!currentPointIds.has(id)) {
      el.remove();
      pointEls.delete(id);
    }
  }
  const r = Math.max(2, view.w * 0.006);
  state.points.forEach((p) => {
    let c = pointEls.get(p.id);
    if (!c) {
      c = document.createElementNS(SVGNS, "circle");
      c.dataset.pointId = p.id;
      pointsLayer.appendChild(c);
      pointEls.set(p.id, c);
    }
    let cls = "gw-point";
    if (selection.has(p.id)) cls += " selected";
    if (fuseTargetId === p.id) cls += " fuse-target";
    c.setAttribute("class", cls);
    c.setAttribute("cx", p.x);
    c.setAttribute("cy", p.y);
    c.setAttribute("r", fuseTargetId === p.id ? r * 1.6 : r);
  });

  updateSelectionCaption();
  updateToolbarState();
}

function updateSelectionCaption() {
  selectionCaptionEl.textContent =
    selection.size === 0 ? "No selection" : `${selection.size} point${selection.size > 1 ? "s" : ""} selected`;
}

function updateToolbarState() {
  undoBtn.disabled = undoStack.length === 0;
  redoBtn.disabled = redoStack.length === 0;
  deleteBtn.disabled = selection.size === 0;
  resetGraphBtn.disabled = !originalSnapshot;
  dlEditedJsonBtn.disabled = state.points.length === 0;
  resetViewBtn.disabled = imgW === 0;
  continueToGeoBtn.disabled = state.points.length === 0;
}

// Live marquee selection, applied as the band is dragged rather than only on
// release: plain drag replaces the selection, shift adds to it, alt takes
// away -- the combination every vector editor uses. `base` is the selection
// as it was when the drag started, so growing and shrinking the band stays
// reversible within the same gesture.
function applyMarquee(p1, p2, mode, base) {
  const x1 = Math.min(p1.x, p2.x);
  const x2 = Math.max(p1.x, p2.x);
  const y1 = Math.min(p1.y, p2.y);
  const y2 = Math.max(p1.y, p2.y);
  const inside = new Set();
  state.points.forEach((p) => {
    if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) inside.add(p.id);
  });
  if (mode === "add") selection = new Set([...base, ...inside]);
  else if (mode === "subtract") selection = new Set([...base].filter((id) => !inside.has(id)));
  else selection = inside;
}

function updateMarquee(p1, p2) {
  const x1 = Math.min(p1.x, p2.x);
  const y1 = Math.min(p1.y, p2.y);
  marqueeEl.setAttribute("x", x1);
  marqueeEl.setAttribute("y", y1);
  marqueeEl.setAttribute("width", Math.abs(p2.x - p1.x));
  marqueeEl.setAttribute("height", Math.abs(p2.y - p1.y));
  marqueeEl.style.display = "";
}

function hideMarquee() {
  marqueeEl.style.display = "none";
}

function updateNewEdgePreview(from, to) {
  newEdgePreviewEl.setAttribute("x1", from.x);
  newEdgePreviewEl.setAttribute("y1", from.y);
  newEdgePreviewEl.setAttribute("x2", to.x);
  newEdgePreviewEl.setAttribute("y2", to.y);
  newEdgePreviewEl.style.display = "";
}

function hideNewEdgePreview() {
  newEdgePreviewEl.style.display = "none";
}

// --- Mouse interaction -----------------------------------------------------

function movedEnough(e) {
  return Math.hypot(e.clientX - dragData.startClientX, e.clientY - dragData.startClientY) > CLICK_DRAG_THRESHOLD_PX;
}

// Starts a "move" drag for whatever's currently in `selection` -- shared by
// clicking a point and clicking an edge, so re-dragging an already-selected
// point OR an already-selected edge both move the whole current selection
// together, not just the two points that were clicked on.
function beginMoveDrag(svgPos, e) {
  const ids = [...selection];
  const singleId = ids.length === 1 ? ids[0] : null;
  dragMode = "move";
  dragData = {
    beforeSnapshot: snapshot(),
    startSvg: svgPos,
    startClientX: e.clientX,
    startClientY: e.clientY,
    startPositions: new Map(ids.map((id) => [id, { x: getPoint(id).x, y: getPoint(id).y }])),
    neighborIds:
      singleId != null
        ? state.edges
            .filter((ed) => ed.a === singleId || ed.b === singleId)
            .map((ed) => (ed.a === singleId ? ed.b : ed.a))
        : [],
  };
}

function beginPanDrag(svgPos, e) {
  dragMode = "pan";
  dragData = {
    invCtm0: svg.getScreenCTM().inverse(),
    startSvg: svgPos,
    startView: { ...view },
    startClientX: e.clientX,
    startClientY: e.clientY,
  };
  canvasWrap.classList.add("panning");
}

// Right-drag is one of the pan gestures, so the canvas must not answer it with
// a context menu.
svg.addEventListener("contextmenu", (e) => e.preventDefault());

svg.addEventListener("mousedown", (e) => {
  if (!imgW) return;
  const svgPos = svgFromClient(e.clientX, e.clientY);

  // Pan: middle-drag (CAD/GIS), right-drag, or space+drag (design apps). The
  // left button is left free for selecting, which is what it does everywhere
  // else in this class of editor.
  if (e.button === 1 || e.button === 2 || (e.button === 0 && spaceHeld)) {
    beginPanDrag(svgPos, e);
    e.preventDefault();
    return;
  }
  if (e.button !== 0) return;

  const pointEl = e.target.closest(".gw-point");
  const edgeEl = !pointEl ? e.target.closest("[data-edge-id]") : null;

  // Draw: ctrl/cmd-drag. NOT alt-drag -- nearly every Linux window manager
  // grabs alt-drag to move the window itself, so the gesture never reaches
  // the page. Alt is used here only for "subtract from selection", which is
  // a modifier on a drag that starts on empty canvas, where a stolen gesture
  // costs nothing.
  if (e.ctrlKey || e.metaKey) {
    const fromId = pointEl ? Number(pointEl.dataset.pointId) : null;
    dragMode = "new-edge";
    dragData = {
      fromId,
      startSvg: fromId != null ? { x: getPoint(fromId).x, y: getPoint(fromId).y } : svgPos,
      startClientX: e.clientX,
      startClientY: e.clientY,
    };
    updateNewEdgePreview(dragData.startSvg, svgPos);
    e.preventDefault();
    return;
  }

  if (edgeEl) {
    const edgeId = Number(edgeEl.dataset.edgeId);
    const edge = state.edges.find((ed) => ed.id === edgeId);
    if (!edge) return;
    if (e.shiftKey) {
      // Toggles, like shift-click on a point -- shift-clicking a wall that's
      // already selected has to be able to take it back out again.
      if (selection.has(edge.a) && selection.has(edge.b)) {
        selection.delete(edge.a);
        selection.delete(edge.b);
      } else {
        selection.add(edge.a);
        selection.add(edge.b);
      }
      render();
      return;
    }
    // Clicking an edge that's already (fully) part of the current selection
    // starts dragging the whole selection, same as re-dragging a selected
    // point -- only resets to just this edge's own two points when it wasn't
    // already selected.
    if (!(selection.has(edge.a) && selection.has(edge.b))) {
      selection.clear();
      selection.add(edge.a);
      selection.add(edge.b);
      render();
    }
    beginMoveDrag(svgPos, e);
    e.preventDefault();
    return;
  }

  if (pointEl) {
    const hitId = Number(pointEl.dataset.pointId);
    if (e.shiftKey) {
      if (selection.has(hitId)) selection.delete(hitId);
      else selection.add(hitId);
      render();
      return;
    }
    if (!selection.has(hitId)) {
      selection.clear();
      selection.add(hitId);
      render();
    }
    beginMoveDrag(svgPos, e);
    e.preventDefault();
    return;
  }

  // Empty background: rubber-band select. A plain click that never turns into
  // a drag is just an empty band, which clears the selection -- the same
  // click-away-to-deselect it always did, now falling out of the selection
  // rule instead of being a special case of the pan gesture.
  dragMode = "marquee";
  dragData = {
    startSvg: svgPos,
    startClientX: e.clientX,
    startClientY: e.clientY,
    mode: e.shiftKey ? "add" : e.altKey ? "subtract" : "replace",
    baseSelection: new Set(selection),
  };
  applyMarquee(svgPos, svgPos, dragData.mode, dragData.baseSelection);
  updateMarquee(svgPos, svgPos);
  render();
  e.preventDefault();
});

window.addEventListener("mousemove", (e) => {
  if (!dragMode) return;
  const svgPos = svgFromClient(e.clientX, e.clientY);

  if (dragMode === "move") {
    let dx = svgPos.x - dragData.startSvg.x;
    let dy = svgPos.y - dragData.startSvg.y;
    if (dragData.startPositions.size > 1) {
      ({ dx, dy } = snapAxisDelta(dx, dy));
    }
    for (const [id, start] of dragData.startPositions) {
      const p = getPoint(id);
      p.x = start.x + dx;
      p.y = start.y + dy;
    }
    fuseTargetId = null;
    hideGuides();
    if (dragData.startPositions.size === 1) {
      const [draggedId] = [...dragData.startPositions.keys()];
      const dragged = getPoint(draggedId);
      const snapped = applyMagnetism(dragged, draggedId, dragData.neighborIds);
      dragged.x = snapped.x;
      dragged.y = snapped.y;
      showGuides(snapped.guideX, snapped.guideY);
      const near = nearestPoint(dragged.x, dragged.y, draggedId);
      const radius = view.w * 0.015;
      if (near && near.dist <= radius) fuseTargetId = near.id;
    }
    render();
    return;
  }

  if (dragMode === "pan") {
    const cur = svgFromClientUsingMatrix(e.clientX, e.clientY, dragData.invCtm0);
    const startCur = svgFromClientUsingMatrix(dragData.startClientX, dragData.startClientY, dragData.invCtm0);
    view.x = dragData.startView.x - (cur.x - startCur.x);
    view.y = dragData.startView.y - (cur.y - startCur.y);
    applyView();
    return;
  }

  if (dragMode === "marquee") {
    updateMarquee(dragData.startSvg, svgPos);
    applyMarquee(dragData.startSvg, svgPos, dragData.mode, dragData.baseSelection);
    render();
    return;
  }

  if (dragMode === "new-edge") {
    const neighborIds = dragData.fromId != null ? [dragData.fromId] : [];
    const snapped = applyMagnetism(svgPos, null, neighborIds);
    dragData.lastSnapped = snapped;
    showGuides(snapped.guideX, snapped.guideY);
    updateNewEdgePreview(dragData.startSvg, snapped);
    return;
  }
});

window.addEventListener("mouseup", (e) => {
  if (!dragMode) return;
  const svgPos = svgFromClient(e.clientX, e.clientY);
  const mode = dragMode;
  dragMode = null;

  if (mode === "move") {
    if (dragData.startPositions.size === 1 && fuseTargetId != null) {
      const [draggedId] = [...dragData.startPositions.keys()];
      fusePoints(fuseTargetId, draggedId);
      selection.clear();
      selection.add(fuseTargetId);
      commitIfChanged(dragData.beforeSnapshot);
    } else if (movedEnough(e)) {
      commitIfChanged(dragData.beforeSnapshot);
    } else {
      render();
    }
    fuseTargetId = null;
  } else if (mode === "pan") {
    canvasWrap.classList.remove("panning");
  } else if (mode === "marquee") {
    hideMarquee();
    applyMarquee(dragData.startSvg, svgPos, dragData.mode, dragData.baseSelection);
    render();
  } else if (mode === "new-edge") {
    hideNewEdgePreview();
    hideGuides();
    const before = snapshot();
    const releasePos = dragData.lastSnapped || svgPos;
    const targetEl = document.elementFromPoint(e.clientX, e.clientY);
    const targetPointEl = targetEl && targetEl.closest && targetEl.closest(".gw-point");
    let targetId = targetPointEl ? Number(targetPointEl.dataset.pointId) : null;
    if (targetId == null) {
      const near = nearestPoint(releasePos.x, releasePos.y, dragData.fromId);
      const radius = view.w * 0.015;
      if (near && near.dist <= radius) targetId = near.id;
    }
    if (dragData.fromId == null) {
      addPoint(releasePos.x, releasePos.y);
    } else if (targetId != null && targetId !== dragData.fromId) {
      addEdge(dragData.fromId, targetId);
    } else if (targetId == null) {
      const p = addPoint(releasePos.x, releasePos.y);
      addEdge(dragData.fromId, p.id);
    }
    commitIfChanged(before);
  }

  hideGuides();
  dragData = {};
});

const NUDGE_DIRECTIONS = {
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
};

window.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;

  // Space held = hand tool, for as long as it's down.
  if (e.code === "Space" && !e.repeat) {
    spaceHeld = true;
    canvasWrap.classList.add("pan-ready");
    e.preventDefault(); // space would otherwise scroll the page
    return;
  }

  // Arrow keys nudge the selection, x10 with shift -- the usual step pair.
  const nudge = NUDGE_DIRECTIONS[e.key];
  if (nudge && !e.ctrlKey && !e.metaKey && !e.altKey) {
    if (!selection.size) return;
    e.preventDefault();
    const step = e.shiftKey ? NUDGE_COARSE_PX : NUDGE_PX;
    const before = snapshot();
    selection.forEach((id) => {
      const p = getPoint(id);
      if (!p) return;
      p.x += nudge[0] * step;
      p.y += nudge[1] * step;
    });
    commitIfChanged(before);
    return;
  }

  // F8 toggles snapping -- the CAD convention (F8 ortho / F3 osnap), and the
  // one thing here you'd otherwise have to leave the canvas to reach.
  if (e.key === "F8") {
    e.preventDefault();
    toggleMagnetism();
    return;
  }

  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "a") {
    e.preventDefault();
    if (e.shiftKey) selection.clear();
    else selection = new Set(state.points.map((p) => p.id));
    render();
    return;
  }

  // Zoom from the keyboard, about the middle of the view. Ctrl+0 fits, which
  // is the same thing as the reset the toolbar button does.
  if ((e.ctrlKey || e.metaKey) && e.key === "0") {
    e.preventDefault();
    resetView();
    render();
    return;
  }
  if (e.key === "+" || e.key === "=") {
    e.preventDefault();
    zoomBy(1 / 1.3);
    return;
  }
  if (e.key === "-" || e.key === "_") {
    e.preventDefault();
    zoomBy(1.3);
    return;
  }

  if (e.key === "Delete" || e.key === "Backspace") {
    e.preventDefault();
    deleteSelected();
  } else if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === "z") {
    e.preventDefault();
    undo();
  } else if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === "y" || (e.shiftKey && e.key.toLowerCase() === "z"))) {
    e.preventDefault();
    redo();
  } else if (e.key === "Escape") {
    if (dragMode === "marquee") {
      hideMarquee();
      selection = new Set(dragData.baseSelection);
      dragMode = null;
      dragData = {};
      render();
    } else if (dragMode === "new-edge") {
      hideNewEdgePreview();
      hideGuides();
      dragMode = null;
      dragData = {};
    } else if (dragMode === "move") {
      restore(dragData.beforeSnapshot);
      hideGuides();
      dragMode = null;
      dragData = {};
      fuseTargetId = null;
      render();
    } else if (selection.size) {
      selection.clear();
      render();
    }
  }
});

window.addEventListener("keyup", (e) => {
  if (e.code !== "Space") return;
  spaceHeld = false;
  canvasWrap.classList.remove("pan-ready");
});

// Alt-tabbing away with space down would otherwise leave the hand tool stuck
// on, since the keyup lands in another window.
window.addEventListener("blur", () => {
  spaceHeld = false;
  canvasWrap.classList.remove("pan-ready");
});

// The single place the bindings are written down for the user -- kept next to
// the handlers above so the two can't drift apart unnoticed.
initShortcutHelp(helpBtn, [
  {
    title: "Select",
    rows: [
      ["click", "Select a point, or a wall (both its points)"],
      ["Shift + click", "Add to / remove from the selection"],
      ["drag", "Rubber-band select, replacing the selection"],
      ["Shift + drag", "Rubber-band, adding to the selection"],
      ["Alt + drag", "Rubber-band, removing from the selection"],
      ["Alt + double-click", "On a wall: select the whole aligned wall run"],
      ["Ctrl + A", "Select every point"],
      ["Ctrl + Shift + A", "Deselect everything"],
      ["Esc", "Clear the selection, or cancel the gesture under way"],
    ],
  },
  {
    title: "Edit",
    rows: [
      ["drag", "On a selected point or wall: move the whole selection"],
      ["drag onto a point", "Drop a dragged point on another to fuse the two"],
      ["Ctrl + drag", "From a point: draw a wall to another point, or to a new one"],
      ["Ctrl + drag", "From empty canvas: drop a new free point"],
      ["double-click", "On a wall: insert a point there, splitting it"],
      ["arrows", "Nudge the selection by 1 px"],
      ["Shift + arrows", "Nudge the selection by 10 px"],
      ["Del / Backspace", "Delete the selection"],
      ["Ctrl + Z", "Undo"],
      ["Ctrl + Shift + Z", "Redo (Ctrl + Y also works)"],
    ],
  },
  {
    title: "View",
    rows: [
      ["scroll", "Zoom, centred on the cursor"],
      ["+ / -", "Zoom in / out, centred on the view"],
      ["Ctrl + 0", "Fit the whole plan in the view"],
      ["Space + drag", "Pan"],
      ["middle-drag", "Pan (right-drag works too)"],
      ["double-click", "On empty canvas: reset the view"],
      ["F8", "Turn magnetism (snapping) on or off"],
      ["?", "Show this list (F1 too)"],
    ],
  },
]);

// --- Loading -----------------------------------------------------------

function loadBackgroundImage(query) {
  return new Promise((resolve, reject) => {
    const url = `/api/base.png?${query}`;
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

function buildStateFromSkeleton(skeleton) {
  const pts = skeleton.points || [];
  state.points = pts.map((p, i) => ({ id: i, x: p.x, y: p.y, arity: p.arity, conf: p.conf }));
  nextPointId = pts.length;

  const keyOf = (x, y) => `${Math.round(x * 100)}_${Math.round(y * 100)}`;
  const byKey = new Map();
  state.points.forEach((p) => byKey.set(keyOf(p.x, p.y), p.id));

  function findPointId(x, y) {
    const k = keyOf(x, y);
    if (byKey.has(k)) return byKey.get(k);
    const near = nearestPoint(x, y, null);
    return near && near.dist <= 1.0 ? near.id : null;
  }

  state.edges = [];
  nextEdgeId = 0;
  (skeleton.wall_segments || []).forEach(([[x1, y1], [x2, y2]]) => {
    const a = findPointId(x1, y1);
    const b = findPointId(x2, y2);
    if (a != null && b != null && a !== b) {
      state.edges.push({ id: nextEdgeId++, a, b });
    }
  });

  originalSnapshot = snapshot();
  undoStack = [];
  redoStack = [];
}

function downloadEditedJson() {
  const byId = new Map(state.points.map((p) => [p.id, p]));
  const out = {
    points: state.points.map((p) => ({ x: p.x, y: p.y, arity: p.arity, conf: p.conf })),
    wall_segments: state.edges.map((ed) => {
      const a = byId.get(ed.a);
      const b = byId.get(ed.b);
      return [
        [a.x, a.y],
        [b.x, b.y],
      ];
    }),
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "edited_skeleton.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

continueToGeoBtn.addEventListener("click", () => {
  // Hands the graph off via sessionStorage (same-tab, same-origin) rather than a
  // server round-trip -- consistent with editing itself being purely client-side
  // state; the doors step reads this key on load, adds its doors to it, and
  // passes the same object on to geo.js. The query string goes along in the URL
  // so the doors step can ask for the same frozen model/plan/criteria this page
  // was opened with. Points/edges are passed with their
  // stable ids intact (not the x/y-only wall_segments shape downloadEditedJson
  // produces for external consumers), so the export step doesn't have to
  // re-match coordinates back to ids the way loading skeleton.json originally did.
  // Points left dangling with no wall at all (never connected, or orphaned by a
  // deletion) carry nothing for georeferencing/export to use, so they're dropped
  // here rather than passed through as stray unconnected markers.
  const linkedPointIds = new Set();
  state.edges.forEach((e) => {
    linkedPointIds.add(e.a);
    linkedPointIds.add(e.b);
  });
  const points = state.points.filter((p) => linkedPointIds.has(p.id)).map((p) => ({ ...p }));
  try {
    sessionStorage.setItem(
      "cubi-geo-graph",
      JSON.stringify({ points, edges: state.edges.map((e) => ({ ...e })) })
    );
  } catch (e) {
    setStatus("Could not hand off the graph to the doors step: " + e.message, true);
    return;
  }
  window.location.href = `/doors${window.location.search}`;
});

undoBtn.addEventListener("click", undo);
redoBtn.addEventListener("click", redo);
deleteBtn.addEventListener("click", deleteSelected);
resetViewBtn.addEventListener("click", resetView);
dlEditedJsonBtn.addEventListener("click", downloadEditedJson);
resetGraphBtn.addEventListener("click", () => {
  if (!originalSnapshot) return;
  const before = snapshot();
  restore(originalSnapshot);
  selection.clear();
  commitIfChanged(before);
});

function updateMagnetismBtn() {
  magnetismBtn.classList.toggle("active", magnetismEnabled);
  magnetismBtn.setAttribute("aria-pressed", String(magnetismEnabled));
  magnetismBtn.textContent = `Magnetism: ${magnetismEnabled ? "on" : "off"}`;
}
function toggleMagnetism() {
  magnetismEnabled = !magnetismEnabled;
  try {
    localStorage.setItem("cubi-edit-magnetism", magnetismEnabled ? "1" : "0");
  } catch (e) {
    // ignore -- persistence is a convenience, not required for this toggle to work
  }
  updateMagnetismBtn();
}
magnetismBtn.addEventListener("click", toggleMagnetism);
updateMagnetismBtn();

async function init() {
  const query = window.location.search.replace(/^\?/, "");
  if (!query) {
    setStatus("Missing model/plan parameters -- open this page from the inference view's “Edit graph” button.", true);
    return;
  }
  setStatus("Loading frozen graph…");
  setBusy(true);
  try {
    const [skeleton, imgUrl] = await Promise.all([
      fetchJson(`/api/skeleton.json?${query}`),
      loadBackgroundImage(query),
    ]);
    buildStateFromSkeleton(skeleton);
    buildSvgSkeleton();
    bgImageEl.setAttributeNS("http://www.w3.org/1999/xlink", "href", imgUrl);
    bgImageEl.setAttribute("href", imgUrl);
    view = { x: 0, y: 0, w: imgW, h: imgH };
    applyView();
    render();
    setStatus(`Ready — ${state.points.length} points, ${state.edges.length} walls`);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    setBusy(false);
    updateToolbarState();
  }
}

init();
