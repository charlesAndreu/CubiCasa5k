const METERS_PER_DEG_LAT = 111320;
const COLLINEAR_TOLERANCE_DEG = 5;
const DEFAULT_ANCHOR = { lat: 48.8566, lon: 2.3522 }; // Paris -- arbitrary, just so something is visible before the user places it
const WALL_STROKE_WIDTH = 2;
// Real-world wall thickness (meters). The source graph is centerlines only,
// with no measured thickness anywhere in it, so these are stand-in constants
// sized from what a real wall measures: about 20cm for an inside wall, 35cm
// for one facing outside. Which walls are "outside" is worked out from the
// skeleton itself (see /api/footprint) -- an exterior wall is one of the same
// bands as every other wall, just wider. Unrelated to WALL_STROKE_WIDTH
// above, which only controls on-screen pixel rendering.
const WALL_WIDTH_M = 0.2;
const EXTERIOR_WALL_WIDTH_M = 0.35;
// Widest wall gap still treated as a doorway (so: sealed) rather than as a
// way out of the room, and the smallest void still worth exporting as a
// room. See /api/footprint.
const DOOR_BRIDGE_M = 0.9;
const MIN_ROOM_AREA_M2 = 1.0;

const statusEl = document.getElementById("status");
const statusSpinner = document.getElementById("statusSpinner");

const geocodeInput = document.getElementById("geocodeInput");
const geocodeResultsEl = document.getElementById("geocodeResults");
const latInput = document.getElementById("latInput");
const lonInput = document.getElementById("lonInput");
const useLatLonBtn = document.getElementById("useLatLonBtn");

const dlGeoJsonBtn = document.getElementById("dlGeoJsonBtn");
const levelNumberInput = document.getElementById("levelNumberInput");
const DEFAULT_SCALE_M_PER_PX = 0.03; // 3x the original 0.01 -- the whole generated venue footprint, not just the stroke width, should read as noticeably bigger/more real-world-sized on first placement

// User-editable level number (defaults to "0" in the input). Used both as the
// OSM-style tags.level (how Wemap's own bulk geofeature import scopes each
// feature to a level, per IndoorFieldSerializer) and as the flat
// properties.level the Pro dashboard's "Add level from file" upload gate
// checks for (see venueService.findLevelFeature) -- without a feature
// carrying that flat property, the dashboard rejects the whole file as "not a
// valid GeoJSON" before it ever reaches the backend's own (more lenient)
// tags-based schema.
function getLevelNumber() {
  const v = levelNumberInput.value.trim();
  return v === "" ? "0" : v;
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg || "";
  statusEl.classList.toggle("error", isError);
}
function setBusy(busy) {
  statusSpinner.classList.toggle("hidden", !busy);
}

// --- Graph loaded from the editor (sessionStorage handoff) -----------------

let graph = { points: [], edges: [] };
let pointById = new Map();
let chains = []; // consolidated wall runs: each an array of point ids
// Areal geometry from /api/footprint, in local pixel space like everything
// else here: one entry per polygon, each [exteriorRing, ...holeRings], each
// ring an array of [x, y]. All of it is rebuilt whenever the SCALE changes,
// since the thickness behind it is fixed in meters, not in pixels.
let wallPolygons = [];
let exteriorWallPolygons = [];
let roomPolygons = [];

function angleDiffDeg(a, b) {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

// Merges any chain of edges through a point where exactly 2 walls meet nearly
// in a straight line (not a real corner) into one longer run, so the map
// draws one stroke per actual wall instead of one per tiny edited segment. A
// point stays a hard break between two separate runs whenever it's a dead
// end, a real junction (3+ walls), or a corner (2 walls at an angle). These
// runs are the on-screen centerlines only -- what gets exported is the
// buffered bands from /api/footprint, which works off the raw edges.
function consolidate(points, edges) {
  pointById = new Map(points.map((p) => [p.id, p]));
  const adjacency = new Map(points.map((p) => [p.id, []]));
  edges.forEach((ed) => {
    adjacency.get(ed.a)?.push({ other: ed.b, edgeId: ed.id });
    adjacency.get(ed.b)?.push({ other: ed.a, edgeId: ed.id });
  });

  function isStraightThrough(prevId, pid, nextId) {
    const prev = pointById.get(prevId);
    const cur = pointById.get(pid);
    const next = pointById.get(nextId);
    const a1 = (Math.atan2(cur.y - prev.y, cur.x - prev.x) * 180) / Math.PI;
    const a2 = (Math.atan2(next.y - cur.y, next.x - cur.x) * 180) / Math.PI;
    return angleDiffDeg(a1, a2) <= COLLINEAR_TOLERANCE_DEG;
  }

  const visitedEdges = new Set();
  const result = [];

  edges.forEach((startEdge) => {
    if (visitedEdges.has(startEdge.id)) return;
    visitedEdges.add(startEdge.id);
    let chain = [startEdge.a, startEdge.b];

    // Extend forward from the tail end.
    for (;;) {
      const tail = chain[chain.length - 1];
      const prev = chain[chain.length - 2];
      const arms = (adjacency.get(tail) || []).filter((arm) => !visitedEdges.has(arm.edgeId));
      if (arms.length !== 1 || (adjacency.get(tail) || []).length !== 2) break;
      const arm = arms[0];
      if (!isStraightThrough(prev, tail, arm.other)) break;
      visitedEdges.add(arm.edgeId);
      chain.push(arm.other);
    }
    // Extend backward from the head end, symmetrically.
    for (;;) {
      const head = chain[0];
      const next = chain[1];
      const arms = (adjacency.get(head) || []).filter((arm) => !visitedEdges.has(arm.edgeId));
      if (arms.length !== 1 || (adjacency.get(head) || []).length !== 2) break;
      const arm = arms[0];
      if (!isStraightThrough(next, head, arm.other)) break;
      visitedEdges.add(arm.edgeId);
      chain.unshift(arm.other);
    }
    result.push(chain);
  });

  return result;
}

// --- Local-pixel -> lat/lon transform ---------------------------------------

let localOrigin = { x: 0, y: 0 };
let localBBox = null; // {minX, minY, maxX, maxY} in local pixel space, for the resize handles
let anchor = { ...DEFAULT_ANCHOR };
let rotationDeg = 0;
let scaleXmPerPx = DEFAULT_SCALE_M_PER_PX;
let scaleYmPerPx = DEFAULT_SCALE_M_PER_PX;

function localToLatLng(x, y) {
  const dx = (x - localOrigin.x) * scaleXmPerPx;
  const dy = (y - localOrigin.y) * scaleYmPerPx;
  const east = dx;
  const north = -dy; // plan-pixel y grows downward -- south
  const rad = (rotationDeg * Math.PI) / 180;
  const east2 = east * Math.cos(rad) + north * Math.sin(rad);
  const north2 = -east * Math.sin(rad) + north * Math.cos(rad);
  const dLat = north2 / METERS_PER_DEG_LAT;
  const dLon = east2 / (METERS_PER_DEG_LAT * Math.cos((anchor.lat * Math.PI) / 180));
  return L.latLng(anchor.lat + dLat, anchor.lon + dLon);
}

// Wall thickness and the door-gap tolerance are real-world meters, so the
// buffering that turns centerlines into wall bands has to happen in a metric
// plane -- not in plan pixels, whose meaning changes every time the shape is
// resized and whose x/y scales can even differ from each other. Plan-meter
// space is exactly localToLatLng's own intermediate step, before the
// rotation and the lat/lon conversion, both of which are rigid/conformal and
// so leave a band's width alone once it's placed on the map.
function localToPlanMeters(x, y) {
  return [(x - localOrigin.x) * scaleXmPerPx, (y - localOrigin.y) * scaleYmPerPx];
}

function planMetersToLocal(xm, ym) {
  return [xm / scaleXmPerPx + localOrigin.x, ym / scaleYmPerPx + localOrigin.y];
}

// Rotates a world point by deltaDeg (same clockwise-positive convention as
// localToLatLng's own rotation) around an arbitrary world pivot. Rotating the
// PLAN's content around an arbitrary point (not just around `anchor`) is done
// by rotating `anchor` itself around that same pivot by the same delta, in
// addition to the usual `rotationDeg += delta` -- see the "rotate" gesture.
function rotateLatLngAround(point, pivot, deltaDeg) {
  const metersPerDegLon = METERS_PER_DEG_LAT * Math.cos((pivot.lat * Math.PI) / 180);
  const east = (point.lon - pivot.lon) * metersPerDegLon;
  const north = (point.lat - pivot.lat) * METERS_PER_DEG_LAT;
  const rad = (deltaDeg * Math.PI) / 180;
  const east2 = east * Math.cos(rad) + north * Math.sin(rad);
  const north2 = -east * Math.sin(rad) + north * Math.cos(rad);
  return {
    lat: pivot.lat + north2 / METERS_PER_DEG_LAT,
    lon: pivot.lon + east2 / metersPerDegLon,
  };
}

// Solves for the `anchor` that makes local point (fx,fy) map (via
// localToLatLng, at the CURRENT rotationDeg/scale) to exactly `fixedWorld` --
// used by the resize-handle drag to keep the opposite corner/edge stationary
// while scaleXmPerPx/scaleYmPerPx change.
function anchorForFixedPoint(fx, fy, fixedWorld) {
  const dx = (fx - localOrigin.x) * scaleXmPerPx;
  const dy = (fy - localOrigin.y) * scaleYmPerPx;
  const east = dx;
  const north = -dy;
  const rad = (rotationDeg * Math.PI) / 180;
  const east2 = east * Math.cos(rad) + north * Math.sin(rad);
  const north2 = -east * Math.sin(rad) + north * Math.cos(rad);
  const metersPerDegLon = METERS_PER_DEG_LAT * Math.cos((fixedWorld.lat * Math.PI) / 180);
  return {
    lat: fixedWorld.lat - north2 / METERS_PER_DEG_LAT,
    lon: fixedWorld.lon - east2 / metersPerDegLon,
  };
}

// The 8 resize-handle definitions: 4 corners (both axes free) + 4 edge
// midpoints (one axis free). `fixedX`/`fixedY` is the reference point that
// must stay stationary while dragging this handle -- the opposite corner for
// a corner handle, the opposite edge (at this handle's own coordinate on the
// unaffected axis) for an edge handle.
function getHandles() {
  const { minX, minY, maxX, maxY } = localBBox;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  return [
    { id: "nw", x: minX, y: minY, fixedX: maxX, fixedY: maxY, axis: "both" },
    { id: "ne", x: maxX, y: minY, fixedX: minX, fixedY: maxY, axis: "both" },
    { id: "se", x: maxX, y: maxY, fixedX: minX, fixedY: minY, axis: "both" },
    { id: "sw", x: minX, y: maxY, fixedX: maxX, fixedY: minY, axis: "both" },
    { id: "n", x: midX, y: minY, fixedX: midX, fixedY: maxY, axis: "y" },
    { id: "s", x: midX, y: maxY, fixedX: midX, fixedY: minY, axis: "y" },
    { id: "e", x: maxX, y: midY, fixedX: minX, fixedY: midY, axis: "x" },
    { id: "w", x: minX, y: midY, fixedX: maxX, fixedY: midY, axis: "x" },
  ];
}

// --- Map + overlay layers ---------------------------------------------------

const map = L.map("map", { attributionControl: true });
map.setView([DEFAULT_ANCHOR.lat, DEFAULT_ANCHOR.lon], 17);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

let wallLayers = []; // parallel to `chains`
let wallPolygonLayers = []; // parallel to `wallPolygons`, then `exteriorWallPolygons`
let roomLayers = []; // parallel to `roomPolygons`
let handleMarkers = []; // parallel to getHandles(); shown only while selected
let rotateHandleMarker = null; // single rotate handle; shown only while selected
let rotateStalkLine = null; // thin line connecting the top-center handle to the rotate handle

// Selecting the shape (a single plain click, no modifier key) shows BOTH the
// resize handles and the rotate handle at once and keeps them up until
// something else is clicked -- there's no separate "resize mode" vs "rotate
// mode" to step in and out of, matching how every mainstream design tool
// (Figma/PowerPoint/Keynote/Google Slides) actually works: select once, then
// drag whichever handle you want.
let selected = false;

function setSelected(next) {
  if (selected === next) return;
  selected = next;
  if (selected) showHandles();
  else hideHandles();
}

// The rotate handle sits a bit above the top-center resize handle, in the
// same LOCAL pixel space as everything else -- so it (and the stalk line
// connecting it to the shape) rotates and scales along with the shape
// itself, exactly like the equivalent handle in Figma/PowerPoint/Keynote.
function rotateHandleLocalPoint() {
  const { minX, maxX, minY, maxY } = localBBox;
  const midX = (minX + maxX) / 2;
  const offset = Math.max((maxY - minY) * 0.15, 20);
  return { x: midX, y: minY - offset };
}

function showHandles() {
  if (!localBBox) return;
  if (!handleMarkers.length) {
    handleMarkers = getHandles().map((h) => {
      const ll = localToLatLng(h.x, h.y);
      return L.marker(ll, {
        icon: L.divIcon({
          className: `gw-handle gw-handle-${h.axis}`,
          iconSize: [12, 12],
          iconAnchor: [6, 6],
        }),
        keyboard: false,
      }).addTo(map);
    });
    // Tag each marker's DOM element with its handle id for the mousedown hit-test.
    handleMarkers.forEach((m, i) => {
      const el = m.getElement();
      if (el) el.dataset.handleId = getHandles()[i].id;
    });
  }
  if (!rotateHandleMarker) {
    const top = getHandles().find((h) => h.id === "n");
    const rp = rotateHandleLocalPoint();
    rotateHandleMarker = L.marker(localToLatLng(rp.x, rp.y), {
      icon: L.divIcon({ className: "gw-rotate-handle", html: "&#8635;", iconSize: [18, 18], iconAnchor: [9, 9] }),
      keyboard: false,
    }).addTo(map);
    rotateStalkLine = L.polyline([localToLatLng(top.x, top.y), localToLatLng(rp.x, rp.y)], {
      color: "#ffd83d",
      weight: 2,
      interactive: false,
    }).addTo(map);
  }
}

function hideHandles() {
  handleMarkers.forEach((m) => map.removeLayer(m));
  handleMarkers = [];
  if (rotateHandleMarker) {
    map.removeLayer(rotateHandleMarker);
    rotateHandleMarker = null;
  }
  if (rotateStalkLine) {
    map.removeLayer(rotateStalkLine);
    rotateStalkLine = null;
  }
}

function updateHandlePositions() {
  if (handleMarkers.length) {
    const handles = getHandles();
    handleMarkers.forEach((m, i) => {
      m.setLatLng(localToLatLng(handles[i].x, handles[i].y));
    });
  }
  if (rotateHandleMarker) {
    const top = getHandles().find((h) => h.id === "n");
    const rp = rotateHandleLocalPoint();
    rotateHandleMarker.setLatLng(localToLatLng(rp.x, rp.y));
    rotateStalkLine.setLatLngs([localToLatLng(top.x, top.y), localToLatLng(rp.x, rp.y)]);
  }
}

function bboxCenterWorld() {
  const cx = (localBBox.minX + localBBox.maxX) / 2;
  const cy = (localBBox.minY + localBBox.maxY) / 2;
  const ll = localToLatLng(cx, cy);
  return { lat: ll.lat, lon: ll.lng };
}

// The map view is just the wall structure, same color throughout, no
// separate point markers -- and the export is lines/polygons only, no Point
// features either.

// --- Undo (Ctrl+Z / Cmd+Z) for placement ------------------------------------

let history = []; // stack of pre-change snapshots; pushed once per gesture/placement, not per mousemove
const MAX_HISTORY = 50;

function snapshotTransform() {
  return { anchor: { ...anchor }, rotationDeg, scaleXmPerPx, scaleYmPerPx };
}

function pushHistory() {
  history.push(snapshotTransform());
  if (history.length > MAX_HISTORY) history.shift();
}

function undoTransform() {
  if (!history.length) return;
  const prev = history.pop();
  anchor = prev.anchor;
  rotationDeg = prev.rotationDeg;
  scaleXmPerPx = prev.scaleXmPerPx;
  scaleYmPerPx = prev.scaleYmPerPx;
  updateOverlay();
  updateHandlePositions();
  scheduleFootprintRefresh(); // undoing a resize puts the scale back, so the meter-sized bands need rebuilding too
  setStatus("Undid last placement change.");
}

window.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() !== "z" || !(e.ctrlKey || e.metaKey) || e.shiftKey) return;
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return; // let native text-field undo work
  e.preventDefault();
  undoTransform();
});

// One drag gesture on the overlay covers translate/rotate/resize -- no
// modifier key needed for any of it:
//   click (or click-and-drag) on the wall structure -> selects the shape
//     (showing its handles) and, if it turns into a drag, translates it in
//     the same motion -- exactly like clicking-and-dragging a shape in any
//     mainstream design tool selects AND moves it together
//   click on empty map while selected -> deselects (hides the handles)
//   while selected: drag a corner/edge handle to resize from the OPPOSITE
//     corner/edge -- a corner free-forms both axes unless Shift is held
//     (checked live, can be toggled mid-drag) to keep the current X:Y
//     proportion; an edge always affects only its own axis
//   while selected: drag the single rotate handle (above the top-center
//     resize handle) to rotate the whole plan around its own bounding-box
//     center
// Scroll wheel stays free for Leaflet's own map zoom throughout.
let dragGestureMode = null; // 'translate' | 'rotate' | 'resize' | null
let gestureStart = {};

function screenPointFor(latlng) {
  return map.latLngToContainerPoint(latlng);
}

function endDragGesture() {
  if (dragGestureMode === "resize") scheduleFootprintRefresh();
  dragGestureMode = null;
  map.dragging.enable();
  document.getElementById("map").classList.remove("dragging-overlay");
}

// A single listener, attached directly to the map container's native DOM
// element in the CAPTURE phase -- not map.on("mousedown", ...), which runs in
// the bubble phase, same as Leaflet's OWN internal Map.Drag mousedown handler
// (registered when the map was constructed, before this script's code runs
// at all). Since that handler always sees the event first, it always starts
// tracking a real map pan before we get a chance to react -- and even though
// we call map.dragging.disable() right after, Leaflet has already decided
// internally that a drag happened, and defensively swallows the very next
// click on the whole container once the gesture ends (this is what was
// silently eating the second gesture in testing: every drag "worked" once,
// but starting a second one immediately after always landed on the bare map
// container instead of the wall geometry underneath the cursor). Running in
// the capture phase and calling preventDefault/stopPropagation ourselves,
// before Leaflet's own bubble-phase handler ever runs, stops it from
// starting a pan in the first place -- so there's nothing for it to
// "remember" and suppress afterwards.
function attachOverlayDrag() {
  map.getContainer().addEventListener(
    "mousedown",
    (nativeEvent) => {
      const target = nativeEvent.target;
      const handleEl = target && target.closest && target.closest(".gw-handle");
      const rotateHandleEl = target && target.closest && target.closest(".gw-rotate-handle");
      const hitWall =
        target && target.closest && target.closest(".gw-wall-line, .gw-wall-poly, .gw-room-poly");
      const onShape = !!(handleEl || rotateHandleEl || hitWall);

      if (selected && handleEl) {
        const handle = getHandles().find((h) => h.id === handleEl.dataset.handleId);
        if (!handle) return;
        nativeEvent.preventDefault();
        nativeEvent.stopPropagation();
        const fixedWorldLL = localToLatLng(handle.fixedX, handle.fixedY);
        pushHistory();
        dragGestureMode = "resize";
        gestureStart = {
          handle,
          scaleX: scaleXmPerPx,
          scaleY: scaleYmPerPx,
          fixedWorld: { lat: fixedWorldLL.lat, lon: fixedWorldLL.lng },
        };
        map.dragging.disable();
        document.getElementById("map").classList.add("dragging-overlay");
        return;
      }

      if (selected && rotateHandleEl) {
        nativeEvent.preventDefault();
        nativeEvent.stopPropagation();
        const latlng = map.containerPointToLatLng(map.mouseEventToContainerPoint(nativeEvent));
        const pivot = bboxCenterWorld();
        const pivotPt = screenPointFor(L.latLng(pivot.lat, pivot.lon));
        const mousePt = screenPointFor(latlng);
        pushHistory();
        dragGestureMode = "rotate";
        gestureStart = {
          pivot,
          angle: (Math.atan2(mousePt.y - pivotPt.y, mousePt.x - pivotPt.x) * 180) / Math.PI,
          rotationDeg,
          anchor: { ...anchor },
        };
        map.dragging.disable();
        document.getElementById("map").classList.add("dragging-overlay");
        return;
      }

      if (!onShape) {
        setSelected(false);
        return; // empty map -- let Leaflet's own panning happen
      }

      if (!hitWall) return; // a stray handle from just before a deselect -- ignore

      nativeEvent.preventDefault();
      nativeEvent.stopPropagation();
      setSelected(true);
      const latlng = map.containerPointToLatLng(map.mouseEventToContainerPoint(nativeEvent));
      pushHistory();
      dragGestureMode = "translate";
      gestureStart = { latlng, anchor: { ...anchor } };
      map.dragging.disable();
      document.getElementById("map").classList.add("dragging-overlay");
    },
    true // capture phase
  );
}
attachOverlayDrag();

map.on("mousemove", (e) => {
  if (!dragGestureMode) return;
  if (dragGestureMode === "translate") {
    anchor = {
      lat: gestureStart.anchor.lat + (e.latlng.lat - gestureStart.latlng.lat),
      lon: gestureStart.anchor.lon + (e.latlng.lng - gestureStart.latlng.lng),
    };
  } else if (dragGestureMode === "rotate") {
    const pivot = gestureStart.pivot;
    const pivotPt = screenPointFor(L.latLng(pivot.lat, pivot.lon));
    const mousePt = screenPointFor(e.latlng);
    const curAngle = (Math.atan2(mousePt.y - pivotPt.y, mousePt.x - pivotPt.x) * 180) / Math.PI;
    const deltaAngle = curAngle - gestureStart.angle;
    let next = gestureStart.rotationDeg + deltaAngle;
    next = (((next + 180) % 360) + 360) % 360 - 180; // wrap to [-180, 180]
    rotationDeg = next;
    anchor = rotateLatLngAround(gestureStart.anchor, pivot, deltaAngle);
  } else if (dragGestureMode === "resize") {
    const h = gestureStart.handle;
    const fixedWorld = gestureStart.fixedWorld;
    const metersPerDegLon = METERS_PER_DEG_LAT * Math.cos((fixedWorld.lat * Math.PI) / 180);
    const eastWorld = (e.latlng.lng - fixedWorld.lon) * metersPerDegLon;
    const northWorld = (e.latlng.lat - fixedWorld.lat) * METERS_PER_DEG_LAT;
    const rad = (rotationDeg * Math.PI) / 180;
    // Inverse of the forward rotation (rotate the drag vector by -rotationDeg)
    // to get back into the plan's own local axes.
    const eastLocal = eastWorld * Math.cos(rad) - northWorld * Math.sin(rad);
    const northLocal = eastWorld * Math.sin(rad) + northWorld * Math.cos(rad);

    let newScaleX = gestureStart.scaleX;
    let newScaleY = gestureStart.scaleY;
    if ((h.axis === "both" || h.axis === "x") && h.x !== h.fixedX) {
      newScaleX = eastLocal / (h.x - h.fixedX);
    }
    if ((h.axis === "both" || h.axis === "y") && h.y !== h.fixedY) {
      newScaleY = -northLocal / (h.y - h.fixedY);
    }
    if (h.axis === "both" && e.originalEvent.shiftKey) {
      const factor = (newScaleX / gestureStart.scaleX + newScaleY / gestureStart.scaleY) / 2;
      newScaleX = gestureStart.scaleX * factor;
      newScaleY = gestureStart.scaleY * factor;
    }
    scaleXmPerPx = Math.max(1e-4, Math.min(10, newScaleX));
    scaleYmPerPx = Math.max(1e-4, Math.min(10, newScaleY));
    anchor = anchorForFixedPoint(h.fixedX, h.fixedY, fixedWorld);
  }
  updateOverlay();
  updateHandlePositions();
});
["mouseup", "mouseout"].forEach((evt) =>
  map.on(evt, () => {
    if (!dragGestureMode) return;
    endDragGesture();
  })
);

// A footprint polygon ([exteriorRing, ...holeRings] in local pixels) in the
// nested-array form Leaflet's L.polygon takes, which is the same nesting
// GeoJSON uses for a Polygon.
function polygonLatLngs(rings) {
  return rings.map((ring) => ring.map(([x, y]) => localToLatLng(x, y)));
}

function buildLayers() {
  wallLayers.forEach((l) => map.removeLayer(l));

  wallLayers = chains.map((chain) => {
    const latlngs = chain.map((id) => localToLatLng(pointById.get(id).x, pointById.get(id).y));
    return L.polyline(latlngs, { color: "#ff00ff", weight: WALL_STROKE_WIDTH, className: "gw-wall-line" }).addTo(map);
  });
  restackLayers();
}

// The areal layers -- what actually gets exported now -- drawn under the
// centerlines rather than instead of them: a 20cm band is well under a pixel
// wide at a zoomed-out view, so the fixed-width strokes stay on top as the
// thing you can always see and grab.
function buildFootprintLayers() {
  roomLayers.forEach((l) => map.removeLayer(l));
  wallPolygonLayers.forEach((l) => map.removeLayer(l));

  roomLayers = roomPolygons.map((rings) =>
    L.polygon(polygonLatLngs(rings), {
      color: "#4da3ff",
      weight: 1,
      fillColor: "#4da3ff",
      fillOpacity: 0.25,
      className: "gw-room-poly",
    }).addTo(map)
  );
  // Interior and exterior bands are drawn the same -- they differ only in
  // thickness, which is in the geometry itself.
  wallPolygonLayers = [...wallPolygons, ...exteriorWallPolygons].map((rings) =>
    L.polygon(polygonLatLngs(rings), {
      color: "#ff00ff",
      weight: 0,
      fillColor: "#ff00ff",
      fillOpacity: 0.85,
      className: "gw-wall-poly",
    }).addTo(map)
  );
  restackLayers();
}

// Back to front: rooms, the wall bands, then the centerlines. Leaflet stacks
// by insertion order, and the two layer groups are rebuilt independently of
// each other, so the order is re-asserted explicitly instead of being left to
// whichever was built last.
function restackLayers() {
  roomLayers.forEach((l) => l.bringToFront());
  wallPolygonLayers.forEach((l) => l.bringToFront());
  wallLayers.forEach((l) => l.bringToFront());
}

function updateOverlay() {
  chains.forEach((chain, i) => {
    wallLayers[i].setLatLngs(chain.map((id) => localToLatLng(pointById.get(id).x, pointById.get(id).y)));
  });
  // Translating/rotating just re-projects these; only a scale change makes
  // their (meter-based) thickness wrong, and that schedules a recompute.
  roomLayers.forEach((l, i) => l.setLatLngs(polygonLatLngs(roomPolygons[i])));
  const bands = [...wallPolygons, ...exteriorWallPolygons];
  wallPolygonLayers.forEach((l, i) => l.setLatLngs(polygonLatLngs(bands[i])));
}

// --- Loading the graph handed off by the editor -----------------------------

async function loadGraph() {
  let raw;
  try {
    raw = sessionStorage.getItem("cubi-geo-graph");
  } catch (e) {
    raw = null;
  }
  if (!raw) {
    setStatus('No graph found -- open this page via the editor\'s "Continue to export" button.', true);
    return;
  }
  try {
    graph = JSON.parse(raw);
  } catch (e) {
    setStatus("Could not read the handed-off graph: " + e.message, true);
    return;
  }
  if (!graph.points || !graph.points.length) {
    setStatus("The handed-off graph is empty.", true);
    return;
  }

  const xs = graph.points.map((p) => p.x);
  const ys = graph.points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  localOrigin = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
  localBBox = { minX, minY, maxX, maxY };

  chains = consolidate(graph.points, graph.edges);

  buildLayers();
  fitToOverlay();

  setBusy(true);
  try {
    await refreshFootprint();
    const bands = wallPolygons.length + exteriorWallPolygons.length;
    setStatus(
      `${bands} wall polygon${bands === 1 ? "" : "s"}` +
        ` (${Math.round(WALL_WIDTH_M * 100)}/${Math.round(EXTERIOR_WALL_WIDTH_M * 100)}cm inside/outside),` +
        ` ${roomPolygons.length} room${roomPolygons.length === 1 ? "" : "s"}.`
    );
  } catch (e) {
    setStatus("Could not build wall/room polygons: " + e.message, true);
  } finally {
    setBusy(false);
  }

  dlGeoJsonBtn.disabled = false;
}

// Wall bands + room polygons for the CURRENT scale. Sent (and returned) in
// plan-meter space so the server's buffer distances are real meters, then
// converted straight back to local pixels, which is what every other piece
// of geometry on this page is in.
async function fetchFootprint() {
  const segments = graph.edges.map((ed) => {
    const a = pointById.get(ed.a);
    const b = pointById.get(ed.b);
    return [localToPlanMeters(a.x, a.y), localToPlanMeters(b.x, b.y)];
  });
  if (!segments.length) return { walls: [], exteriorWalls: [], rooms: [] };
  const res = await fetch("/api/footprint", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      segments,
      wall_width: WALL_WIDTH_M,
      exterior_width: EXTERIOR_WALL_WIDTH_M,
      door_bridge: DOOR_BRIDGE_M,
      min_room_area: MIN_ROOM_AREA_M2,
    }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  const toLocal = (rings) => rings.map((ring) => ring.map(([xm, ym]) => planMetersToLocal(xm, ym)));
  return {
    walls: (data.walls || []).map(toLocal),
    exteriorWalls: (data.exterior_walls || []).map(toLocal),
    rooms: (data.rooms || []).map(toLocal),
  };
}

let footprintSeq = 0;
let footprintTimer = null;

// Resizing changes how many meters a pixel is worth, so the bands have to be
// rebuilt at the new scale -- debounced, because a resize drag fires this on
// every mousemove, and sequence-checked, because responses can land out of
// order and the last request started is the only one still describing the
// shape on screen.
function scheduleFootprintRefresh(delay = 250) {
  clearTimeout(footprintTimer);
  footprintTimer = setTimeout(() => {
    refreshFootprint().catch((e) => setStatus("Footprint: " + e.message, true));
  }, delay);
}

async function refreshFootprint() {
  const seq = ++footprintSeq;
  const fp = await fetchFootprint();
  if (seq !== footprintSeq) return;
  wallPolygons = fp.walls;
  exteriorWallPolygons = fp.exteriorWalls;
  roomPolygons = fp.rooms;
  buildFootprintLayers();
}

function fitToOverlay() {
  const allLatLngs = wallLayers.flatMap((l) => l.getLatLngs());
  if (allLatLngs.length) map.fitBounds(L.latLngBounds(allLatLngs), { padding: [40, 40] });
}

// --- Geocoding ---------------------------------------------------------------

async function geocode(q) {
  const res = await fetch(`/api/geocode?q=${encodeURIComponent(q)}`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function hideGeocodeResults() {
  geocodeResultsEl.innerHTML = "";
  geocodeResultsEl.classList.add("hidden");
}

function placeAt(lat, lon, zoom) {
  anchor = { lat, lon };
  map.setView([lat, lon], zoom || map.getZoom());
  updateOverlay();
}

// Search-as-you-type: debounced (standard ~400ms pause-in-typing threshold,
// same idea as most autocomplete UIs) so a normal typing burst fires one
// request, not one per keystroke -- Nominatim's usage policy caps at 1
// request/second anyway (enforced server-side in /api/geocode), so this also
// keeps us comfortably under that rather than relying on the server-side
// throttle to smooth over a request-per-keystroke flood. A minimum query
// length avoids firing on 1-2 noise characters. A sequence counter drops the
// response to any search that's since been superseded by a newer one, so a
// slow request that finally resolves after a faster/later one can't
// clobber the results with stale data.
const GEOCODE_DEBOUNCE_MS = 400;
const GEOCODE_MIN_LENGTH = 3;
let geocodeDebounceTimer = null;
let geocodeRequestSeq = 0;

async function performGeocodeSearch(q) {
  const seq = ++geocodeRequestSeq;
  if (q.length < GEOCODE_MIN_LENGTH) {
    hideGeocodeResults();
    setStatus("");
    return;
  }
  setBusy(true);
  setStatus("Searching…");
  try {
    const results = await geocode(q);
    if (seq !== geocodeRequestSeq) return; // superseded by a newer search
    if (!results.length) {
      hideGeocodeResults();
      setStatus("No results.", true);
      return;
    }
    geocodeResultsEl.innerHTML = "";
    results.forEach((r) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "list-group-item list-group-item-action";
      item.textContent = r.display_name;
      item.addEventListener("click", () => {
        placeAt(r.lat, r.lon, 18);
        hideGeocodeResults();
        setStatus(`Placed at: ${r.display_name}`);
      });
      geocodeResultsEl.appendChild(item);
    });
    geocodeResultsEl.classList.remove("hidden");
    setStatus(`${results.length} result${results.length > 1 ? "s" : ""} -- pick one.`);
  } catch (e) {
    if (seq !== geocodeRequestSeq) return;
    setStatus(e.message, true);
  } finally {
    if (seq === geocodeRequestSeq) setBusy(false);
  }
}

geocodeInput.addEventListener("input", () => {
  clearTimeout(geocodeDebounceTimer);
  const q = geocodeInput.value.trim();
  if (!q) {
    geocodeRequestSeq++; // invalidate any in-flight search
    hideGeocodeResults();
    setStatus("");
    return;
  }
  geocodeDebounceTimer = setTimeout(() => performGeocodeSearch(q), GEOCODE_DEBOUNCE_MS);
});
geocodeInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    clearTimeout(geocodeDebounceTimer);
    performGeocodeSearch(geocodeInput.value.trim());
  }
});

useLatLonBtn.addEventListener("click", () => {
  const lat = Number(latInput.value);
  const lon = Number(lonInput.value);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    setStatus("Enter valid numeric lat/lon.", true);
    return;
  }
  hideGeocodeResults();
  placeAt(lat, lon, 18);
  setStatus(`Placed at ${lat.toFixed(6)}, ${lon.toFixed(6)}`);
});


// --- Export ------------------------------------------------------------------

dlGeoJsonBtn.addEventListener("click", () => {
  // Matches the shape of real exported Wemap venue files (checked against
  // several actual examples): flat properties -- indoor/level set directly on
  // the feature, NOT nested under a tags/metadata wrapper.
  //
  // Walls go out as Polygons with their thickness in the geometry (bands
  // built by /api/footprint from the centerline graph), not as centerline
  // LineStrings -- a wall is a surface in the target venue model, and a
  // renderer given a bare centerline has to guess. The "width" property is
  // kept alongside as the thickness each band was built with. Walls facing
  // outside are wider, but they're the same kind of feature and come from
  // the same skeleton: nothing is exported that isn't a wall of the graph.
  //
  // Rooms are the voids those wall bands enclose, so each one sits flush
  // against the inner wall faces and no doorway splits it -- see
  // /api/footprint.
  //
  // Walls and rooms, and nothing else: no building envelope, no level
  // boundary, no centerlines. Every feature in the file is a Polygon.
  const levelNumber = getLevelNumber();
  const ringsToCoords = (rings) =>
    rings.map((ring) =>
      ring.map(([x, y]) => {
        const ll = localToLatLng(x, y);
        return [ll.lng, ll.lat];
      })
    );

  const bandFeature = (rings, id, width) => ({
    type: "Feature",
    properties: { external_id: id, indoor: "wall", level: levelNumber, width },
    geometry: { type: "Polygon", coordinates: ringsToCoords(rings) },
  });

  const wallFeatures = [
    ...wallPolygons.map((rings, i) => bandFeature(rings, `wall-${i}`, WALL_WIDTH_M)),
    ...exteriorWallPolygons.map((rings, i) => bandFeature(rings, `wall-ext-${i}`, EXTERIOR_WALL_WIDTH_M)),
  ];
  // The bands ARE the walls now -- there's no centerline geometry to fall
  // back on that wouldn't be a LineString -- so a failed footprint request
  // means there is nothing to export, and saying so beats writing a file
  // with no walls in it.
  if (!wallFeatures.length) {
    setStatus("No wall polygons to export -- the footprint request failed. Try again.", true);
    return;
  }

  const roomFeatures = roomPolygons.map((rings, i) => ({
    type: "Feature",
    properties: {
      external_id: `room-${i}`,
      indoor: "room",
      level: levelNumber,
    },
    geometry: { type: "Polygon", coordinates: ringsToCoords(rings) },
  }));

  const featureCollection = {
    type: "FeatureCollection",
    features: [...roomFeatures, ...wallFeatures],
  };
  const blob = new Blob([JSON.stringify(featureCollection, null, 2)], { type: "application/geo+json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "walls.geojson";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

loadGraph().catch((e) => setStatus(e.message, true));
