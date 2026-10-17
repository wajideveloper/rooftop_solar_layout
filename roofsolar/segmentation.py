"""Step 4 - roof face segmentation and planar reconstruction.

Pipeline
  1. local plane-fit normals + planarity (residual std) in a k x k window
  2. region growing (seeds = most planar pixels): a neighbour joins a region when its normal is within
     `angle_threshold` of the region's running mean normal AND it lies within `dist_threshold` of the
     region's running plane (centroid + mean normal)
  3. RANSAC plane fit per region (robust to chimney/edge pixels), then least-squares refinement on inliers
  4. merge adjacent regions whose planes coincide (angle + height agreement along the shared border)
  5. competitive re-assignment: every footprint pixel goes to the adjacent face whose plane explains the
     measured height best (if the residual is small enough). This moves face borders onto the true
     plane-plane intersection lines (ridges, hips, valleys) and absorbs small leftover regions.
  6. faces are vectorised to polygons; small or near-vertical regions are dropped (-> obstruction candidates)

A face is a dict with: id, plane (z = a*x' + b*y' + c, x' = x - x0, y' = y - y0), normal, slope, aspect,
polygon, area_2d, area_3d, rmse, n_pixels.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPolygon

from .config import SegmentationParams
from .io_utils import labels_to_polygons
from .terrain import plane_normal_to_slope_aspect

# ------------------------------------------------------------------ local plane fits
def local_plane_normals(z: np.ndarray, res: float, window: int):
    """Least-squares plane in a window around every pixel via box filters.
    Returns unit normals (h, w, 3) in map axes (x east, y north, z up) and the residual std."""
    w = window
    h_, w_ = z.shape
    yy, xx = np.mgrid[0:h_, 0:w_].astype(np.float64)
    x = xx * res
    y = -yy * res                      # rows increase southwards
    U = lambda a: ndimage.uniform_filter(a, w, mode="nearest")
    mx, my, mz = U(x), U(y), U(z.astype(np.float64))
    sxx, syy, sxy = U(x * x) - mx * mx, U(y * y) - my * my, U(x * y) - mx * my
    sxz, syz = U(x * z) - mx * mz, U(y * z) - my * mz
    det = sxx * syy - sxy * sxy + 1e-12
    a = (sxz * syy - syz * sxy) / det          # dz/dx
    b = (syz * sxx - sxz * sxy) / det          # dz/dy
    n = np.stack([-a, -b, np.ones_like(a)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    # residual std of the local fit
    szz = U(z * z) - mz * mz
    resid_var = szz - (a * sxz + b * syz)
    planarity = np.sqrt(np.maximum(resid_var, 0))
    return n.astype(np.float32), planarity.astype(np.float32)


def fit_plane_lsq(P: np.ndarray):
    """Total least squares plane through points P (n,3) -> (unit normal with nz>0, centroid)."""
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    return n, c


def fit_plane_ransac(P: np.ndarray, iters: int, tol: float, rng: np.random.Generator):
    """RANSAC plane: returns (normal, centroid, inlier mask)."""
    m = len(P)
    if m < 10:
        n, c = fit_plane_lsq(P)
        return n, c, np.ones(m, bool)
    idx = rng.integers(0, m, size=(iters, 3))
    A, B, C = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    N = np.cross(B - A, C - A)
    norm = np.linalg.norm(N, axis=1)
    ok = norm > 1e-9
    N, A = N[ok] / norm[ok, None], A[ok]
    # distances of all points to every candidate plane: (iters, m)
    sub = P if m <= 4000 else P[rng.choice(m, 4000, replace=False)]
    d = np.abs(np.einsum("ij,kj->ik", N, sub) - np.einsum("ij,ij->i", N, A)[:, None])
    best = (d < tol).sum(1).argmax()
    n0, a0 = N[best], A[best]
    inl = np.abs(P @ n0 - a0 @ n0) < tol
    if inl.sum() < 3:
        inl[:] = True
    n, c = fit_plane_lsq(P[inl])
    inl = np.abs((P - c) @ n) < tol * 1.5
    if inl.sum() >= 3:
        n, c = fit_plane_lsq(P[inl])
    return n, c, inl


def plane_z(n, c, x, y):
    """Height of plane (normal n through point c) at (x, y)."""
    return c[2] - (n[0] * (x - c[0]) + n[1] * (y - c[1])) / n[2]


# ------------------------------------------------------------------ region growing
def region_grow(normals, planarity, X, Y, Z, mask, p: SegmentationParams, res):
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    cos_thr = np.cos(np.radians(p.angle_threshold_deg))
    order = np.argsort(planarity[mask])
    seeds = np.column_stack(np.nonzero(mask))[order]
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cur = 0
    min_px = int(p.min_region_area / (res * res))
    for r0, c0 in seeds:
        if labels[r0, c0] != 0:
            continue
        cur += 1
        nsum = normals[r0, c0].astype(np.float64).copy()
        csum = np.array([X[r0, c0], Y[r0, c0], Z[r0, c0]], float)
        cnt = 1
        labels[r0, c0] = cur
        members = [(r0, c0)]
        q = deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            nmean = nsum / np.linalg.norm(nsum)
            cen = csum / cnt
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if rr < 0 or cc < 0 or rr >= h or cc >= w or labels[rr, cc] != 0 or not mask[rr, cc]:
                    continue
                nq = normals[rr, cc]
                if float(nq @ nmean) < cos_thr:
                    continue
                pq = np.array([X[rr, cc], Y[rr, cc], Z[rr, cc]])
                if abs(float((pq - cen) @ nmean)) > p.dist_threshold:
                    continue
                labels[rr, cc] = cur
                members.append((rr, cc))
                nsum += nq
                csum += pq
                cnt += 1
                q.append((rr, cc))
        if cnt < min_px:            # too small: release the pixels (they get re-assigned later)
            for rr, cc in members:
                labels[rr, cc] = 0
            cur -= 1
            # mark visited-but-rejected so that we don't re-seed inside the same blob forever
            # (they can still be absorbed by later regions)
    return labels


# ------------------------------------------------------------------ planes per label
def fit_all_planes(labels, X, Y, Z, p: SegmentationParams, rng):
    planes = {}
    for lab in np.unique(labels):
        if lab == 0:
            continue
        m = labels == lab
        P = np.column_stack([X[m], Y[m], Z[m]])
        n, c, inl = fit_plane_ransac(P, p.ransac_iters, p.ransac_inlier, rng)
        resid = (P - c) @ n
        planes[int(lab)] = {"n": n, "c": c, "rmse": float(np.sqrt(np.mean(resid ** 2))), "n_px": int(m.sum())}
    return planes


def adjacency(labels):
    """Set of (a, b) label pairs that touch (4-connectivity), a < b, ignoring 0."""
    pairs = set()
    for a, b in [(labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])]:
        d = (a != b) & (a > 0) & (b > 0)
        for u, v in zip(a[d].ravel(), b[d].ravel()):
            pairs.add((min(int(u), int(v)), max(int(u), int(v))))
    return pairs


def merge_coplanar(labels, planes, X, Y, Z, p: SegmentationParams, rng):
    cos_thr = np.cos(np.radians(p.merge_angle_deg))
    changed = True
    while changed:
        changed = False
        for a, b in sorted(adjacency(labels)):
            if a not in planes or b not in planes:
                continue
            pa, pb = planes[a], planes[b]
            similar = float(pa["n"] @ pb["n"]) >= cos_thr
            if similar:
                # height agreement: evaluate both planes at the centroid of the *other* region
                dz1 = abs(plane_z(pa["n"], pa["c"], pb["c"][0], pb["c"][1]) - pb["c"][2])
                dz2 = abs(plane_z(pb["n"], pb["c"], pa["c"][0], pa["c"][1]) - pa["c"][2])
                similar = max(dz1, dz2) <= p.merge_dist
            m = (labels == a) | (labels == b)
            P = np.column_stack([X[m], Y[m], Z[m]])
            n, c = fit_plane_lsq(P)
            rmse_union = float(np.sqrt(np.mean(((P - c) @ n) ** 2)))
            if not similar and rmse_union > p.merge_union_rmse:
                continue
            labels[labels == b] = a
            n, c, _ = fit_plane_ransac(P, p.ransac_iters, p.ransac_inlier, rng)
            planes[a] = {"n": n, "c": c, "rmse": float(np.sqrt(np.mean(((P - c) @ n) ** 2))), "n_px": int(m.sum())}
            del planes[b]
            changed = True
            break
    return labels, planes


def competitive_reassign(labels, planes, X, Y, Z, mask, tol, iters=6):
    """Each footprint pixel adopts the label of the 8-neighbour whose plane explains Z best (if < tol).
    Repeated, this pushes borders to the plane intersections and fills unlabeled holes."""
    h, w = labels.shape
    ids = sorted(planes)
    if not ids:
        return labels
    # per-label height prediction (stack) - memory ok: faces x pixels
    pred = np.stack([plane_z(planes[i]["n"], planes[i]["c"], X, Y) for i in ids])
    lut = {lab: k for k, lab in enumerate(ids)}
    for _ in range(iters):
        cur_res = np.full((h, w), np.inf)
        for lab, k in lut.items():
            m = labels == lab
            cur_res[m] = np.abs(pred[k][m] - Z[m])
        best_res = cur_res.copy()
        best_lab = labels.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = np.zeros_like(labels)
                src = labels[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)]
                nb[max(0, dr):h - max(0, -dr), max(0, dc):w - max(0, -dc)] = src
                valid = (nb > 0) & mask & (nb != labels)
                if not valid.any():
                    continue
                rr, cc = np.nonzero(valid)
                k = np.array([lut[int(v)] for v in nb[rr, cc]])
                r = np.abs(pred[k, rr, cc] - Z[rr, cc])
                better = (r < best_res[rr, cc] - 0.01) & (r < tol)
                best_res[rr[better], cc[better]] = r[better]
                best_lab[rr[better], cc[better]] = nb[rr[better], cc[better]]
        if np.array_equal(best_lab, labels):
            break
        labels = best_lab
    return labels


def enforce_connectivity(labels, min_px):
    """Every label must be one connected blob; extra blobs become new labels or 0 if tiny."""
    out = np.zeros_like(labels)
    nxt = 1
    for lab in np.unique(labels):
        if lab == 0:
            continue
        comp, n = ndimage.label(labels == lab)
        sizes = np.bincount(comp.ravel())
        for k in range(1, n + 1):
            if sizes[k] >= min_px:
                out[comp == k] = nxt
                nxt += 1
    return out


# ------------------------------------------------------------------ main entry
def segment_roof(z_smooth, z_meas, mask, res, transform, p: SegmentationParams, seed=0):
    rng = np.random.default_rng(seed)
    h, w = mask.shape
    normals, planarity = local_plane_normals(z_smooth, res, p.normal_window)
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    X = transform.c + cols * transform.a
    Y = transform.f + rows * transform.e
    x0, y0 = float(X[mask].mean()), float(Y[mask].mean())
    Xl, Yl = X - x0, Y - y0                     # local coords for numerical stability
    Z = z_smooth.astype(np.float64)
    min_px = int(p.min_region_area / (res * res))

    labels = region_grow(normals, planarity, Xl, Yl, Z, mask, p, res)
    n_grow = len(np.unique(labels)) - 1
    planes = fit_all_planes(labels, Xl, Yl, Z, p, rng)
    labels, planes = merge_coplanar(labels, planes, Xl, Yl, Z, p, rng)
    n_merge = len(planes)

    def cleanup(labels, planes, min_area_px):
        """Dissolve too-small or too-steep faces, let neighbours absorb what their planes explain,
        re-split disconnected labels and refit. Repeated until stable."""
        for _ in range(4):
            before = len(planes)
            for lab in list(planes):
                slope, _ = plane_normal_to_slope_aspect(planes[lab]["n"])
                if planes[lab]["n_px"] < min_area_px or slope > p.max_face_slope_deg:
                    labels[labels == lab] = 0
                    del planes[lab]
            labels = competitive_reassign(labels, planes, Xl, Yl, Z, mask, p.reassign_tol)
            labels = enforce_connectivity(labels, min_px)
            planes = fit_all_planes(labels, Xl, Yl, Z, p, rng)
            labels, planes = merge_coplanar(labels, planes, Xl, Yl, Z, p, rng)
            if len(planes) == before and all(pl["n_px"] >= min_area_px for pl in planes.values()):
                break
        return labels, planes

    labels, planes = cleanup(labels, planes, int(p.min_face_area / (res * res)))
    labels = competitive_reassign(labels, planes, Xl, Yl, Z, mask, p.reassign_tol, iters=3)
    labels = enforce_connectivity(labels, min_px)
    planes = fit_all_planes(labels, Xl, Yl, Z, p, rng)

    # ---- build face records
    polys = labels_to_polygons(labels, transform)
    faces = []
    for lab in sorted(planes):
        pl = planes[lab]
        slope, aspect = plane_normal_to_slope_aspect(pl["n"])
        poly = polys[lab].simplify(p.polygon_simplify, preserve_topology=True)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        m = labels == lab
        resid_meas = z_meas[m] - plane_z(pl["n"], pl["c"], Xl[m], Yl[m])
        faces.append({
            "id": int(lab), "n": pl["n"], "c": pl["c"], "origin": (x0, y0),
            "slope": slope, "aspect": aspect, "polygon": poly,
            "area_2d": float(m.sum() * res * res), "area_3d": float(m.sum() * res * res / np.cos(np.radians(slope))),
            "rmse": pl["rmse"], "rmse_meas": float(np.sqrt(np.mean(resid_meas ** 2))), "n_px": int(m.sum()),
        })
    # reconstructed planar height field + residual of the measured DSM to the planar model
    recon = np.full((h, w), np.nan, np.float32)
    for f in faces:
        m = labels == f["id"]
        recon[m] = plane_z(f["n"], f["c"], Xl[m], Yl[m])
    residual = np.where(np.isfinite(recon), z_meas - recon, np.nan).astype(np.float32)
    unassigned = mask & (labels == 0)
    stats = {"regions_after_growing": n_grow, "after_merge": n_merge, "faces": len(faces),
             "unassigned_px": int(unassigned.sum()),
             "coverage": float((labels > 0).sum() / mask.sum())}
    return {"labels": labels, "faces": faces, "recon": recon, "residual": residual, "unassigned": unassigned,
            "normals": normals, "planarity": planarity, "origin": (x0, y0), "stats": stats}


def face_z(face, x, y):
    """Plane height of a face at absolute map coords."""
    x0, y0 = face["origin"]
    return plane_z(face["n"], face["c"], np.asarray(x) - x0, np.asarray(y) - y0)
