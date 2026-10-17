"""Step 1 - building footprint from the DSM (height above ground) + RGB (vegetation cue).

Idea: the parcel-clipped DSM tells us where the property is. Inside it, anything 2.5 m or more above
ground is either the house or a tree. Trees are separated by (a) being much taller than a 2-storey
roof, (b) having a very rough surface (DSM minus local median has a large std) and (c) a green
excess index. The largest remaining connected component is the house. Its outline is then
regularised: simplified and snapped to the two dominant edge orientations (rectilinear building).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon

from .config import FootprintParams
from .io_utils import Raster, mask_to_polygons


def disk(radius_px: float) -> np.ndarray:
    r = int(max(1, round(radius_px)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def local_roughness(dsm: np.ndarray, window: int) -> np.ndarray:
    """Std of (DSM - local median) in a moving window: ~0 on planes, large on canopy."""
    med = ndimage.median_filter(dsm, size=window)
    resid = dsm - med
    mean = ndimage.uniform_filter(resid, window)
    sq = ndimage.uniform_filter(resid * resid, window)
    return np.sqrt(np.maximum(sq - mean * mean, 0))


def estimate_ground(dsm: np.ndarray, parcel: np.ndarray, roughness: np.ndarray) -> float:
    """Ground = the lowest smooth pixels within the parcel (lawn / driveway)."""
    cand = parcel & (roughness < 0.1)
    z = dsm[cand]
    low = z[z <= np.percentile(z, 25)]
    return float(np.median(low))


def excess_green(rgb: np.ndarray) -> np.ndarray:
    r, g, b = [c.astype(np.float32) for c in rgb[:3]]
    return (2 * g - r - b) / (r + g + b + 1e-6)


def extract_footprint(dsm: Raster, parcel_mask: np.ndarray, rgb: Raster, p: FootprintParams) -> dict:
    z = np.asarray(dsm.data, dtype=np.float32)
    res = dsm.res
    rough = local_roughness(z, p.roughness_window)
    ground = estimate_ground(z, parcel_mask, rough)
    ndsm = z - ground

    exg = excess_green(np.asarray(rgb.data))
    tree = (ndsm > p.tree_height) | (rough > p.roughness_tree) | (exg > p.green_index_tree)
    tree = ndimage.binary_dilation(tree, structure=disk(p.tree_dilate / res))

    elevated = (ndsm > p.min_height_above_ground) & parcel_mask & ~tree
    elevated = ndimage.binary_opening(elevated, structure=disk(p.open_radius / res))
    labels, n = ndimage.label(elevated)
    if n == 0:
        raise RuntimeError("no elevated structure found inside the parcel")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    house = labels == sizes.argmax()
    house = ndimage.binary_closing(house, structure=disk(p.close_radius / res))
    house = ndimage.binary_fill_holes(house)

    raw_polys = mask_to_polygons(house, dsm.transform)
    raw_poly = max(raw_polys, key=lambda g: g.area)
    poly = raw_poly.simplify(p.simplify_tol, preserve_topology=True)
    if p.rectilinear_snap:
        snapped = rectilinearize(poly)
        # accept the regularised outline only if it stays close to the evidence
        iou = snapped.intersection(raw_poly).area / snapped.union(raw_poly).area
        poly = snapped if (snapped.is_valid and iou > 0.85) else poly

    return {
        "ground_z": ground,
        "ndsm": ndsm,
        "roughness": rough,
        "exg": exg,
        "tree_mask": tree,
        "house_mask_raw": house,
        "footprint_raw": raw_poly,
        "footprint": poly,
    }


# ---------------------------------------------------------------- polygon regularisation
def dominant_orientation(poly: Polygon) -> float:
    """Angle (radians) of the longest edge of the minimum rotated rectangle."""
    rect = poly.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    edges = [(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(4)]
    dx, dy = max(edges, key=lambda e: e[0] ** 2 + e[1] ** 2)
    return float(np.arctan2(dy, dx))


def rotate_xy(coords: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return coords @ np.array([[c, -s], [s, c]]).T


def rectilinearize(poly: Polygon) -> Polygon:
    """Snap every edge of a simplified polygon to one of two perpendicular dominant directions.

    In the frame rotated by the dominant orientation each edge is classified as horizontal or
    vertical, consecutive edges of the same class are merged (length-weighted mean coordinate), and
    the new vertices are the intersections of consecutive axis-aligned lines.
    """
    theta = dominant_orientation(poly)
    pts = np.array(poly.exterior.coords[:-1])
    centre = pts.mean(0)
    q = rotate_xy(pts - centre, -theta)
    n = len(q)
    edges = []
    for i in range(n):
        a, b = q[i], q[(i + 1) % n]
        d = b - a
        horiz = abs(d[0]) >= abs(d[1])
        length = float(np.hypot(*d))
        coord = (a[1] + b[1]) / 2 if horiz else (a[0] + b[0]) / 2
        edges.append([horiz, coord, length])
    merged: list[list] = []
    for e in edges:
        if merged and merged[-1][0] == e[0]:
            h, c0, l0 = merged[-1]
            merged[-1] = [h, (c0 * l0 + e[1] * e[2]) / (l0 + e[2] + 1e-9), l0 + e[2]]
        else:
            merged.append(list(e))
    if len(merged) > 1 and merged[0][0] == merged[-1][0]:
        h, c0, l0 = merged[-1]
        merged[0] = [h, (c0 * l0 + merged[0][1] * merged[0][2]) / (l0 + merged[0][2] + 1e-9), l0 + merged[0][2]]
        merged.pop()
    if len(merged) < 4:
        return poly
    verts = []
    for i in range(len(merged)):
        h1, c1, _ = merged[i]
        h2, c2, _ = merged[(i + 1) % len(merged)]
        x = c2 if h1 else c1
        y = c1 if h1 else c2
        verts.append((x, y))
    out = Polygon(rotate_xy(np.array(verts), theta) + centre)
    if not out.is_valid:
        out = out.buffer(0)
    if isinstance(out, MultiPolygon):
        out = max(out.geoms, key=lambda g: g.area)
    return out
