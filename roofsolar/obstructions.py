"""Step 5 - rooftop obstructions (chimneys, skylights, roof windows, dormers, vents).

Three complementary cues, all evaluated inside the building footprint:
  geometric  : measured DSM sits more than `protrusion_threshold` above the fitted roof plane
  topological: pixels that no roof plane could explain (left unassigned by the segmentation)
  radiometric: bright / bluish glazing in the RGB image (skylights and roof windows are often flush
               with the roof and therefore invisible in the DSM)
Blobs are cleaned morphologically, vectorised and classified by height and brightness, then buffered by
a clearance distance to give the keep-out zones used by the panel layout.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPolygon

from .config import ObstructionParams
from .footprint import disk
from .io_utils import mask_to_polygons, polygons_to_mask
from .segmentation import face_z


def glazing_index(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = [c.astype(np.float32) for c in rgb[:3]]
    brightness = (r + g + b) / 3
    blueness = b - (r + g) / 2
    return brightness, blueness


def nearest_face_labels(labels: np.ndarray) -> np.ndarray:
    idx = ndimage.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
    return labels[idx[0], idx[1]]


def nearest_face_plane_height(near: np.ndarray, faces: list, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Height of the plane of the *nearest* labelled face, for every pixel (used under obstructions)."""
    z = np.full(near.shape, np.nan, np.float32)
    for f in faces:
        m = near == f["id"]
        z[m] = face_z(f, X[m], Y[m])
    return z


def detect_obstructions(dsm_meas: np.ndarray, labels: np.ndarray, faces: list, footprint: np.ndarray,
                        rgb: np.ndarray, X: np.ndarray, Y: np.ndarray, res: float, transform,
                        p: ObstructionParams, tree_mask: np.ndarray | None = None) -> dict:
    near = nearest_face_labels(labels)
    base = nearest_face_plane_height(near, faces, X, Y)          # 'roof plane' under every pixel
    protrusion = np.where(footprint, dsm_meas - base, np.nan)
    geom = footprint & (protrusion > p.protrusion_threshold)
    topo = footprint & (labels == 0)
    # radiometric cue: brightness relative to the median of the face the pixel belongs to
    # (absolute thresholds fail: sunlit slate is as bright as glass)
    brightness, blueness = glazing_index(rgb)
    contrast = np.zeros_like(brightness)
    for f in faces:
        m = labels == f["id"]
        if m.any():
            contrast[near == f["id"]] = brightness[near == f["id"]] - np.median(brightness[m])
    bright = footprint & (contrast > p.brightness_contrast)
    bright = _drop_small(ndimage.binary_opening(bright, structure=disk(1)), p.bright_min_area, res)

    raw = geom | topo | bright
    raw = _drop_small(raw, p.min_area, res)
    raw = ndimage.binary_closing(raw, structure=disk(2))
    raw = ndimage.binary_fill_holes(raw)
    raw = ndimage.binary_dilation(raw, structure=disk(p.dilate / res)) & footprint

    lab, n = ndimage.label(raw, structure=np.ones((3, 3)))
    objs = []
    for k in range(1, n + 1):
        m = lab == k
        polys = mask_to_polygons(m, transform)
        if not polys:
            continue
        poly = max(polys, key=lambda g: g.area).simplify(0.1, preserve_topology=True)
        area = float(m.sum() * res * res)
        h = protrusion[m]
        hmax = float(np.nanpercentile(h, 95)) if np.isfinite(h).any() else 0.0
        bright_frac = float(bright[m].mean())
        veg_frac = float(tree_mask[m].mean()) if tree_mask is not None else 0.0
        if veg_frac > p.vegetation_overlap:
            kind = "vegetation overhang"
        elif bright_frac > 0.35:
            kind = "skylight / glazing"
        elif hmax >= p.chimney_height and area < 6.0:
            kind = "chimney"
        elif hmax >= 0.5 and area >= 4.0:
            kind = "dormer"
        elif hmax >= p.protrusion_threshold:
            kind = "vent / roof window"
        else:
            kind = "unmodelled structure"
        objs.append({"id": k, "kind": kind, "polygon": poly, "area_m2": area, "height_m": max(hmax, 0.0),
                     "bright_fraction": bright_frac, "vegetation_fraction": veg_frac,
                     "keepout": poly.buffer(p.clearance, join_style=2)})
    keepout_mask = polygons_to_mask([o["keepout"] for o in objs], footprint.shape, transform) & footprint
    return {"objects": objs, "mask": raw, "keepout_mask": keepout_mask, "protrusion": protrusion,
            "brightness": brightness, "cues": {"geometric": geom, "topological": topo, "bright": bright}}


def _drop_small(mask: np.ndarray, min_area: float, res: float) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel()) * res * res
    keep = sizes >= min_area
    keep[0] = False
    return keep[lab]
