"""Step 7 - solar panel placement on the reconstructed roof faces.

All layout geometry is done in the *roof-plane* coordinate system of each face (the face unfolded
into 2D), so that setbacks, clearances and module sizes are true on-roof distances:

  map (x, y)  --rotate-->  (u, v)   u = along the eave, v = downslope (map-projected)
              --unfold-->  (u, w)   w = v / cos(slope)   (true distance along the roof surface)

For every roof face
  1. suitability: slope within limits, aspect within +-south_half_angle of south (or flat), enough area
  2. usable region (roof plane) = unfolded face polygon shrunk by the edge setback, minus the unfolded
     obstruction polygons grown by the clearance distance
  3. grid packing of true-size L x W rectangles in (u, w): rows are spaced by the panel height + row gap;
     for every candidate row phase the u-phase of each row is chosen to maximise the number of panels
     fully inside the usable region (ties -> aligned with the previous row for a compact layout); both
     orientations are tried; the best layout (max count, then smallest bounding box) wins
  4. fold back: (u, w) -> (u, v = w cos slope) -> map (x, y); corners lifted onto the face plane -> 3D
     rectangles with exactly the nominal module size
Also reported: a simple orientation yield factor (relative to the optimal tilt/azimuth at this latitude).
"""
from __future__ import annotations

import numpy as np
from shapely import prepared
from shapely.affinity import affine_transform, scale
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

from .config import PanelParams
from .segmentation import face_z
from .footprint import dominant_orientation


def orientation_yield(tilt_deg: float, aspect_deg: float, latitude_deg: float) -> float:
    """Crude relative annual yield (1.0 = optimum): cosine of the angle between the panel normal and the
    'mean sun' direction (south, tilt = latitude - 15), blended with an isotropic diffuse term."""
    opt_tilt = latitude_deg - 15.0
    def normal(t, a):
        t, a = np.radians(t), np.radians(a if a >= 0 else 180.0)
        return np.array([np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)])
    n_opt, n = normal(opt_tilt, 180.0), normal(tilt_deg, aspect_deg)
    direct = max(float(n @ n_opt), 0.0)
    diffuse = (1 + np.cos(np.radians(tilt_deg))) / 2
    ref = 0.75 + 0.25 * (1 + np.cos(np.radians(opt_tilt))) / 2
    return float(np.clip((0.75 * direct + 0.25 * diffuse) / ref, 0, 1))


def face_frame(face: dict, footprint: Polygon, p: PanelParams):
    """Unit vectors (e = along eave, d = downslope) in map coordinates."""
    if face["aspect"] < 0 or face["slope"] < p.flat_slope_deg:
        th = dominant_orientation(footprint)
        e = np.array([np.cos(th), np.sin(th)])
    else:
        az = np.radians(face["aspect"])
        d = np.array([np.sin(az), np.cos(az)])
        e = np.array([d[1], -d[0]])
    d = np.array([-e[1], e[0]])
    return e, d


def map_to_uv(e, d, origin):
    """shapely affine params for map -> (u, v): u = e.(p - o), v = d.(p - o)."""
    ox, oy = origin
    return [e[0], e[1], d[0], d[1], -(e[0] * ox + e[1] * oy), -(d[0] * ox + d[1] * oy)]


def uv_to_map(e, d, origin):
    """inverse: p = o + u*e + v*d."""
    return [e[0], d[0], e[1], d[1], origin[0], origin[1]]


def pack_rectangles(region, pw: float, ph: float, gap: float, row_gap: float, n_off: int):
    """Axis-aligned packing of pw x ph rectangles inside `region`. Returns list of (u0, w0)."""
    if region.is_empty:
        return []
    prep = prepared.prep(region)
    minu, minw, maxu, maxw = region.bounds
    pitch_u, pitch_w = pw + gap, ph + row_gap
    best, best_score = [], (-1, 0.0)
    for dw in np.linspace(0, pitch_w, n_off, endpoint=False):
        placed, prev_du = [], None
        for w0 in np.arange(minw + dw - pitch_w, maxw, pitch_w):
            best_row, best_key, best_du = [], None, None
            for du in np.linspace(0, pitch_u, n_off, endpoint=False):
                us = np.arange(minu + du - pitch_u, maxu, pitch_u)
                row = [(u0, w0) for u0 in us if prep.contains(box(u0, w0, u0 + pw, w0 + ph))]
                if not row:
                    continue
                align = -abs(du - prev_du) if prev_du is not None else 0.0
                key = (len(row), align)
                if best_key is None or key > best_key:
                    best_row, best_key, best_du = row, key, du
            if best_row:
                placed += best_row
                prev_du = best_du
        if placed:
            arr = np.array(placed)
            score = (len(placed), -((np.ptp(arr[:, 0]) + pw) * (np.ptp(arr[:, 1]) + ph)))
            if score > best_score:
                best, best_score = placed, score
    return best


def place_panels(faces: list, obstructions: list, footprint: Polygon, p: PanelParams, clearance: float) -> dict:
    obst = [o["polygon"] for o in obstructions]
    panels, face_reports = [], []
    pid = 0
    for f in faces:
        rep = {"face_id": f["id"], "slope": f["slope"], "aspect": f["aspect"], "area_3d": f["area_3d"],
               "yield_factor": orientation_yield(f["slope"], f["aspect"], p.latitude_deg)}
        flat = f["aspect"] < 0 or f["slope"] < p.flat_slope_deg
        south_dev = abs(((f["aspect"] - 180 + 180) % 360) - 180) if not flat else 0.0
        if f["slope"] > p.max_slope_deg or f["slope"] < p.min_slope_deg:
            rep.update(status="rejected: slope", usable_area=0.0, n_panels=0); face_reports.append(rep); continue
        if not flat and south_dev > p.south_half_angle:
            rep.update(status="rejected: aspect (north-facing)", usable_area=0.0, n_panels=0); face_reports.append(rep); continue

        # ---- unfold the face and the obstructions into the roof plane
        e, d = face_frame(f, footprint, p)
        origin = (f["polygon"].centroid.x, f["polygon"].centroid.y)
        cos_s = float(np.cos(np.radians(f["slope"])))
        unfold = lambda g: scale(affine_transform(g, map_to_uv(e, d, origin)), xfact=1.0, yfact=1.0 / cos_s, origin=(0, 0))
        fold = lambda g: affine_transform(scale(g, xfact=1.0, yfact=cos_s, origin=(0, 0)), uv_to_map(e, d, origin))
        face_r = unfold(f["polygon"])
        usable = face_r.buffer(-p.edge_setback, join_style=2)
        near = [g for g in obst if g.distance(f["polygon"]) < clearance + 0.5]
        if near:
            usable = usable.difference(unary_union([unfold(g).buffer(clearance, join_style=2) for g in near]))
        rep["usable_area"] = float(usable.area)                      # true roof-surface area
        if usable.is_empty or usable.area < p.min_face_area:
            rep.update(status="rejected: not enough usable area", n_panels=0); face_reports.append(rep); continue

        # ---- pack true-size modules, both orientations
        best, best_orient = [], None
        for orient, (pw, ph) in {"landscape": (p.length, p.width), "portrait": (p.width, p.length)}.items():
            placed = pack_rectangles(usable, pw, ph, p.gap, p.row_gap, p.offset_search)
            if len(placed) > len(best):
                best, best_orient = placed, (orient, pw, ph)
        if not best:
            rep.update(status="no panel fits", n_panels=0); face_reports.append(rep); continue
        orient, pw, ph = best_orient
        face_ring = face_r.exterior
        for (u0, w0) in best:
            rect_r = box(u0, w0, u0 + pw, w0 + ph)
            rect = fold(rect_r)
            xy = np.array(rect.exterior.coords[:-1])
            z = face_z(f, xy[:, 0], xy[:, 1])
            pid += 1
            panels.append({"id": pid, "face_id": f["id"], "orientation": orient, "polygon": rect,
                           "corners3d": np.column_stack([xy, z]).tolist(),
                           "tilt_deg": f["slope"], "azimuth_deg": f["aspect"] if not flat else 180.0,
                           "width_m": pw, "height_m": ph, "area_m2": pw * ph, "power_wp": p.power_wp,
                           "yield_factor": rep["yield_factor"],
                           "edge_distance_m": float(face_ring.distance(rect_r))})
        rep.update(status="ok", n_panels=len(best), orientation=orient,
                   panel_area=len(best) * p.length * p.width,
                   fill_ratio=float(len(best) * p.length * p.width / rep["usable_area"]))
        face_reports.append(rep)
    total_kwp = sum(x["power_wp"] for x in panels) / 1000
    summary = {"n_panels": len(panels), "capacity_kwp": total_kwp,
               "panel_area_m2": sum(x["area_m2"] for x in panels),
               "usable_area_m2": sum(r.get("usable_area", 0) for r in face_reports if r.get("status") == "ok"),
               "weighted_yield_factor": (sum(x["yield_factor"] for x in panels) / len(panels)) if panels else 0.0,
               "faces": face_reports}
    return {"panels": panels, "summary": summary}


def validate_layout(panels: list, faces: list, obstructions: list, p: PanelParams, clearance: float) -> dict:
    """Independent geometric checks of the final layout (map space + 3D)."""
    by_face = {f["id"]: f for f in faces}
    polys = [x["polygon"] for x in panels]
    overlaps = sum(1 for i in range(len(polys)) for j in range(i + 1, len(polys))
                   if polys[i].intersection(polys[j]).area > 1e-6)
    obst = unary_union([o["polygon"] for o in obstructions]) if obstructions else None
    obst_hits = sum(1 for g in polys if obst is not None and g.distance(obst) < clearance - 1e-3)
    outside = sum(1 for x in panels if not by_face[x["face_id"]]["polygon"].buffer(1e-6).contains(x["polygon"]))
    setback_viol = sum(1 for x in panels if x["edge_distance_m"] < p.edge_setback - 1e-3)
    size_err = []
    for x in panels:
        c = np.array(x["corners3d"])
        l1, l2 = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1])
        size_err.append(abs(min(l1, l2) - p.width) + abs(max(l1, l2) - p.length))
    coplanar = []
    for x in panels:
        c = np.array(x["corners3d"])
        n = by_face[x["face_id"]]["n"]
        coplanar.append(float(np.ptp((c - c[0]) @ n)))
    return {"n_panels": len(panels), "overlapping_pairs": overlaps, "panels_closer_than_clearance_to_obstruction": obst_hits,
            "panels_outside_face": outside, "setback_violations": setback_viol,
            "max_3d_size_error_m": float(max(size_err)) if size_err else 0.0,
            "max_out_of_plane_m": float(max(coplanar)) if coplanar else 0.0}
