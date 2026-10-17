"""Step 6 - 3D representation of the reconstructed roof.

Outputs
  * planar DSM            : every face pixel replaced by its fitted plane (GeoTIFF, from step 4)
  * roof edges            : ridge / hip / valley lines = intersections of adjacent face planes, clipped
                            to the shared border; eave / gable edges = face borders on the footprint
  * consistency metrics   : height mismatch between adjacent planes along their shared border
  * mesh (OBJ + PLY)      : each face polygon triangulated with z from its plane, plus vertical walls
                            from the roof edge down to ground level and a floor slab
  * interactive 3D (HTML) : plotly Mesh3d with per-face colouring, obstructions and panels (later)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point
from shapely.ops import unary_union

from .segmentation import face_z


# ------------------------------------------------------------------ plane-plane intersections
def plane_intersection_line(f1: dict, f2: dict):
    """3D line (point, direction) where two face planes intersect, or None if (near) parallel."""
    n1, n2 = f1["n"], f2["n"]
    d = np.cross(n1, n2)
    if np.linalg.norm(d) < 1e-3:
        return None
    d = d / np.linalg.norm(d)
    x0, y0 = f1["origin"]
    c1 = np.array([f1["c"][0] + x0, f1["c"][1] + y0, f1["c"][2]])
    c2 = np.array([f2["c"][0] + x0, f2["c"][1] + y0, f2["c"][2]])
    # solve for a point: n1.p = n1.c1, n2.p = n2.c2, d.p = d.c1
    A = np.vstack([n1, n2, d])
    b = np.array([n1 @ c1, n2 @ c2, d @ c1])
    p = np.linalg.solve(A, b)
    return p, d


def shared_border(labels: np.ndarray, a: int, b: int, transform) -> np.ndarray:
    """Map coordinates (x, y) of pixel centres of face a that touch face b."""
    ma, mb = labels == a, labels == b
    touch = ma & ndimage.binary_dilation(mb, structure=np.ones((3, 3)))
    rr, cc = np.nonzero(touch)
    x = transform.c + (cc + 0.5) * transform.a
    y = transform.f + (rr + 0.5) * transform.e
    return np.column_stack([x, y])


def roof_edges(faces: list, labels: np.ndarray, transform) -> tuple[list, dict]:
    """Ridge/hip/valley segments between adjacent faces + consistency statistics."""
    by_id = {f["id"]: f for f in faces}
    ids = sorted(by_id)
    edges, mism = [], []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pts = shared_border(labels, a, b, transform)
            if len(pts) < 5:
                continue
            fa, fb = by_id[a], by_id[b]
            za, zb = face_z(fa, pts[:, 0], pts[:, 1]), face_z(fb, pts[:, 0], pts[:, 1])
            dz = float(np.mean(np.abs(za - zb)))
            mism.append({"faces": (a, b), "border_px": int(len(pts)), "mean_abs_dz": dz})
            line = plane_intersection_line(fa, fb)
            if line is None:
                continue
            p, d = line
            # project border pixels on the line, take the extent
            t = (pts - p[:2]) @ d[:2] / (d[:2] @ d[:2] + 1e-12)
            t0, t1 = np.percentile(t, [2, 98])
            p0, p1 = p + t0 * d, p + t1 * d
            # classify: convex (ridge/hip) if the roof is below both planes beyond the edge, else valley
            mid = (p0 + p1) / 2
            cen_a = np.array([fa["c"][0] + fa["origin"][0], fa["c"][1] + fa["origin"][1]])
            cen_b = np.array([fb["c"][0] + fb["origin"][0], fb["c"][1] + fb["origin"][1]])
            za_at_b = face_z(fa, cen_b[0], cen_b[1]); zb_true = face_z(fb, cen_b[0], cen_b[1])
            kind = "ridge/hip" if za_at_b > zb_true else "valley"
            if abs(d[2]) > 0.05:
                kind = "hip" if kind == "ridge/hip" else "valley"
            elif kind == "ridge/hip":
                kind = "ridge"
            edges.append({"faces": (a, b), "kind": kind, "p0": p0, "p1": p1, "mean_abs_dz": dz,
                          "geometry": LineString([p0[:2], p1[:2]])})
    stats = {"n_adjacent_pairs": len(mism),
             "mean_border_dz": float(np.mean([m["mean_abs_dz"] for m in mism])) if mism else 0.0,
             "max_border_dz": float(np.max([m["mean_abs_dz"] for m in mism])) if mism else 0.0,
             "pairs": mism}
    return edges, stats


# ------------------------------------------------------------------ meshing
def triangulate_polygon(poly: Polygon, spacing: float = 1.0):
    """Constrained-ish triangulation: Delaunay of boundary + interior grid points, keep triangles whose
    centroid is inside the polygon. Returns (pts2d, tris)."""
    ring = np.array(poly.exterior.coords[:-1])
    pts = [ring]
    for hole in poly.interiors:
        pts.append(np.array(hole.coords[:-1]))
    # densify boundary so that long edges are honoured by Delaunay
    dense = []
    for arr in pts:
        for i in range(len(arr)):
            a, b = arr[i], arr[(i + 1) % len(arr)]
            n = max(1, int(np.hypot(*(b - a)) / spacing))
            for k in range(n):
                dense.append(a + (b - a) * k / n)
    dense = np.array(dense)
    minx, miny, maxx, maxy = poly.bounds
    gx, gy = np.meshgrid(np.arange(minx, maxx, spacing), np.arange(miny, maxy, spacing))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    inner = poly.buffer(-spacing * 0.3)
    if not inner.is_empty:
        from shapely import contains_xy
        grid = grid[contains_xy(inner, grid[:, 0], grid[:, 1])]
        pts2d = np.vstack([dense, grid])
    else:
        pts2d = dense
    if len(pts2d) < 3:
        return pts2d, np.zeros((0, 3), int)
    tri = Delaunay(pts2d)
    cen = pts2d[tri.simplices].mean(1)
    from shapely import contains_xy
    keep = contains_xy(poly, cen[:, 0], cen[:, 1])
    return pts2d, tri.simplices[keep]


def build_mesh(faces: list, footprint: Polygon, ground_z: float, obstructions: list | None = None):
    """Returns dict with vertices (N,3), triangles (M,3), tri_face_id (M,), tri_kind (M,) strings."""
    V, T, FID, KIND = [], [], [], []

    def add(pts3d, tris, fid, kind):
        off = len(V)
        V.extend(pts3d.tolist())
        T.extend((tris + off).tolist())
        FID.extend([fid] * len(tris))
        KIND.extend([kind] * len(tris))

    for f in faces:
        poly = f["polygon"]
        polys = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
        for pg in polys:
            p2, tr = triangulate_polygon(pg)
            if len(tr) == 0:
                continue
            z = face_z(f, p2[:, 0], p2[:, 1])
            add(np.column_stack([p2, z]), tr, f["id"], "roof")
    # walls: footprint ring from roof height (nearest face plane) down to ground
    by_id = {f["id"]: f for f in faces}
    ring = np.array(footprint.exterior.coords)
    roof_union = unary_union([f["polygon"] for f in faces])
    for i in range(len(ring) - 1):
        a, b = ring[i], ring[i + 1]
        n = max(1, int(np.hypot(*(b - a)) / 0.5))
        seg = [a + (b - a) * k / n for k in range(n + 1)]
        zs = []
        for q in seg:
            pt = Point(q)
            best = min(faces, key=lambda f: f["polygon"].distance(pt))
            zs.append(face_z(best, q[0], q[1]))
        top = np.column_stack([seg, zs])
        bot = np.column_stack([seg, np.full(len(seg), ground_z)])
        pts = np.vstack([top, bot])
        tris = []
        m = len(seg)
        for k in range(m - 1):
            tris.append([k, k + 1, m + k])
            tris.append([k + 1, m + k + 1, m + k])
        add(pts, np.array(tris), -1, "wall")
    # floor slab
    p2, tr = triangulate_polygon(footprint, spacing=2.0)
    add(np.column_stack([p2, np.full(len(p2), ground_z)]), tr[:, ::-1], -2, "floor")
    # obstructions as extruded prisms sitting on the roof
    for o in obstructions or []:
        poly = o["polygon"]
        polys = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
        for pg in polys:
            p2, tr = triangulate_polygon(pg, spacing=0.5)
            if len(tr) == 0:
                continue
            base = _nearest_face_height(faces, p2)
            top = base + max(o.get("height_m", 0.3), 0.3)
            add(np.column_stack([p2, top]), tr, 1000 + o["id"], "obstruction")
            ring = np.array(pg.exterior.coords)
            zb = _nearest_face_height(faces, ring)
            pts = np.vstack([np.column_stack([ring, zb + max(o.get("height_m", 0.3), 0.3)]),
                             np.column_stack([ring, zb])])
            m = len(ring)
            tris = []
            for k in range(m - 1):
                tris.append([k, k + 1, m + k]); tris.append([k + 1, m + k + 1, m + k])
            add(pts, np.array(tris), 1000 + o["id"], "obstruction")
    return {"vertices": np.array(V), "triangles": np.array(T, int), "face_id": np.array(FID), "kind": np.array(KIND)}


def _nearest_face_height(faces, pts2d):
    z = np.zeros(len(pts2d))
    for i, q in enumerate(pts2d):
        pt = Point(q)
        best = min(faces, key=lambda f: f["polygon"].distance(pt))
        z[i] = face_z(best, q[0], q[1])
    return z


def write_obj(path: Path, mesh: dict, offset=(0.0, 0.0, 0.0)):
    V, T = mesh["vertices"] - np.array(offset), mesh["triangles"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# roofsolar mesh, coordinates offset by {offset} (EPSG:32630)\n")
        for v in V:
            fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        cur = None
        for t, fid, kind in zip(T, mesh["face_id"], mesh["kind"]):
            key = f"{kind}_{fid}"
            if key != cur:
                fh.write(f"g {key}\n"); cur = key
            fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def write_ply(path: Path, mesh: dict, colors: np.ndarray, offset=(0.0, 0.0, 0.0)):
    V, T = mesh["vertices"] - np.array(offset), mesh["triangles"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n")
        fh.write(f"element face {len(T)}\nproperty list uchar int vertex_indices\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for v in V:
            fh.write(f"{v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for t, c in zip(T, colors):
            fh.write(f"3 {t[0]} {t[1]} {t[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def face_colors(mesh: dict, faces: list) -> np.ndarray:
    import matplotlib
    cmap = matplotlib.colormaps["tab20"]
    ids = {f["id"]: i for i, f in enumerate(faces)}
    cols = np.zeros((len(mesh["triangles"]), 3))
    for i, (fid, kind) in enumerate(zip(mesh["face_id"], mesh["kind"])):
        if kind == "roof":
            cols[i] = np.array(cmap(ids[fid] % 20)[:3]) * 255
        elif kind == "wall":
            cols[i] = (225, 220, 205)
        elif kind == "floor":
            cols[i] = (120, 120, 120)
        else:
            cols[i] = (230, 60, 60)
    return cols


# ------------------------------------------------------------------ interactive 3D
def write_html(path: Path, mesh: dict, faces: list, edges: list, panels: list | None, obstructions: list | None,
               offset, dsm_pts: np.ndarray | None = None):
    import plotly.graph_objects as go
    V = mesh["vertices"] - np.array(offset)
    T = mesh["triangles"]
    cols = face_colors(mesh, faces)
    fig = go.Figure()
    for kind, opacity in [("roof", 1.0), ("wall", 0.5), ("floor", 0.4), ("obstruction", 1.0)]:
        sel = mesh["kind"] == kind
        if not sel.any():
            continue
        tri = T[sel]
        c = ["rgb(%d,%d,%d)" % tuple(x) for x in cols[sel]]
        fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                                facecolor=c, opacity=opacity, name=kind, flatshading=True,
                                lighting=dict(ambient=0.5, diffuse=0.8), showlegend=True))
    if edges:
        xs, ys, zs = [], [], []
        for e in edges:
            p0, p1 = e["p0"] - offset, e["p1"] - offset
            xs += [p0[0], p1[0], None]; ys += [p0[1], p1[1], None]; zs += [p0[2] + 0.03, p1[2] + 0.03, None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="black", width=4), name="ridges / hips / valleys"))
    if panels:
        xs, ys, zs = [], [], []
        for p in panels:
            c3 = np.array(p["corners3d"]) - offset
            c3 = np.vstack([c3, c3[:1]])
            xs += c3[:, 0].tolist() + [None]; ys += c3[:, 1].tolist() + [None]; zs += (c3[:, 2] + 0.08).tolist() + [None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#0a2a8a", width=3), name=f"{len(panels)} panels"))
        # filled panels
        PV, PT = [], []
        for p in panels:
            c3 = np.array(p["corners3d"]) - offset
            off = len(PV)
            PV.extend((c3 + [0, 0, 0.06]).tolist())
            PT += [[off, off + 1, off + 2], [off, off + 2, off + 3]]
        PV, PT = np.array(PV), np.array(PT)
        fig.add_trace(go.Mesh3d(x=PV[:, 0], y=PV[:, 1], z=PV[:, 2], i=PT[:, 0], j=PT[:, 1], k=PT[:, 2],
                                color="#1f4fd8", opacity=0.95, name="panels", showlegend=True))
    if dsm_pts is not None:
        fig.add_trace(go.Scatter3d(x=dsm_pts[:, 0] - offset[0], y=dsm_pts[:, 1] - offset[1], z=dsm_pts[:, 2] - offset[2],
                                   mode="markers", marker=dict(size=1.2, color="rgba(60,60,60,0.5)"), name="measured DSM (sample)",
                                   visible="legendonly"))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title=f"x - {offset[0]:.0f} (m)", yaxis_title=f"y - {offset[1]:.0f} (m)",
                                 zaxis_title=f"z - {offset[2]:.0f} (m)"),
                      title="Reconstructed roof - planar faces, edges, obstructions and panel layout",
                      legend=dict(x=0, y=1), margin=dict(l=0, r=0, t=40, b=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
