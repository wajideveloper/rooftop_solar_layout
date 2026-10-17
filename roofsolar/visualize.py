"""Matplotlib figures for every step (all georeferenced in map units)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from .io_utils import Raster


def extent(r: Raster):
    h, w = r.shape
    t = r.transform
    return [t.c, t.c + w * t.a, t.f + h * t.e, t.f]


def rgb_image(rgb: Raster) -> np.ndarray:
    return np.moveaxis(np.asarray(rgb.data), 0, -1)


def draw_poly(ax, geom, **kw):
    geoms = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for g in geoms:
        if g.is_empty:
            continue
        x, y = g.exterior.xy
        ax.plot(x, y, **kw)
        for ring in g.interiors:
            x, y = ring.xy
            ax.plot(x, y, **kw)


def fill_poly(ax, geom, **kw):
    geoms = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for g in geoms:
        if g.is_empty:
            continue
        ax.fill(*g.exterior.xy, **kw)


def zoom_to(ax, geom, pad=3.0):
    minx, miny, maxx, maxy = geom.bounds
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")


def plot_footprint(dsm: Raster, rgb: Raster, fp: dict, path: Path):
    ex = extent(dsm)
    fig, ax = plt.subplots(2, 2, figsize=(16, 15))
    ax[0, 0].imshow(rgb_image(rgb), extent=ex)
    draw_poly(ax[0, 0], fp["footprint_raw"], color="yellow", lw=1, label="raw")
    draw_poly(ax[0, 0], fp["footprint"], color="red", lw=2, label="regularised")
    ax[0, 0].legend(); ax[0, 0].set_title("RGB + footprint")
    im = ax[0, 1].imshow(fp["ndsm"], extent=ex, cmap="viridis", vmin=0, vmax=12)
    plt.colorbar(im, ax=ax[0, 1], label="m"); ax[0, 1].set_title(f"nDSM (ground = {fp['ground_z']:.2f} m)")
    ax[1, 0].imshow(fp["tree_mask"], extent=ex, cmap="Greens"); ax[1, 0].set_title("vegetation / canopy mask")
    ax[1, 1].imshow(fp["house_mask_raw"], extent=ex, cmap="gray"); ax[1, 1].set_title("building mask")
    draw_poly(ax[1, 1], fp["footprint"], color="red", lw=2)
    for a in ax.ravel():
        zoom_to(a, fp["footprint"], pad=8)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def plot_preprocess(dsm: Raster, fp: dict, pre: dict, path: Path):
    ex = extent(dsm)
    m = fp["footprint_mask"]
    raw = np.where(m, np.asarray(dsm.data, float), np.nan)
    sm = np.where(m, pre["dsm_smooth"], np.nan)
    vmin, vmax = np.nanpercentile(raw, [1, 99])
    fig, ax = plt.subplots(2, 2, figsize=(16, 15))
    im = ax[0, 0].imshow(raw, extent=ex, cmap="terrain", vmin=vmin, vmax=vmax); plt.colorbar(im, ax=ax[0, 0], label="m")
    ax[0, 0].set_title("raw DSM (footprint)")
    im = ax[0, 1].imshow(sm, extent=ex, cmap="terrain", vmin=vmin, vmax=vmax); plt.colorbar(im, ax=ax[0, 1], label="m")
    ax[0, 1].set_title("filled + edge-preserving smoothed DSM")
    d = np.where(m, pre["dsm_filled"] - pre["dsm_smooth"], np.nan)
    im = ax[1, 0].imshow(d, extent=ex, cmap="RdBu_r", vmin=-0.3, vmax=0.3); plt.colorbar(im, ax=ax[1, 0], label="m")
    ax[1, 0].set_title(f"removed noise (raw - smooth), std = {pre['noise_std']:.3f} m")
    ax[1, 1].imshow(np.where(m, 0.3, np.nan), extent=ex, cmap="gray", vmin=0, vmax=1)
    ax[1, 1].imshow(np.ma.masked_where(~pre["gap_mask"], pre["gap_mask"]), extent=ex, cmap="autumn")
    ax[1, 1].imshow(np.ma.masked_where(~pre["spike_mask"], pre["spike_mask"]), extent=ex, cmap="cool")
    ax[1, 1].set_title(f"gaps filled: {pre['n_gaps']} px (orange), spikes removed: {pre['n_spikes']} px (cyan)")
    for a in ax.ravel():
        zoom_to(a, fp["footprint"], pad=2)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def plot_terrain(dsm: Raster, rgb: Raster, fp: dict, pre: dict, terr: dict, path: Path):
    ex = extent(dsm)
    m = fp["footprint_mask"]
    fig, ax = plt.subplots(2, 2, figsize=(16, 15))
    im = ax[0, 0].imshow(np.where(m, pre["dsm_smooth"], np.nan), extent=ex, cmap="terrain")
    plt.colorbar(im, ax=ax[0, 0], label="m"); ax[0, 0].set_title("elevation")
    im = ax[0, 1].imshow(np.where(m, terr["slope"], np.nan), extent=ex, cmap="magma", vmin=0, vmax=60)
    plt.colorbar(im, ax=ax[0, 1], label="deg"); ax[0, 1].set_title("slope")
    asp = np.where(m & (terr["slope"] > 3), terr["aspect"], np.nan)
    im = ax[1, 0].imshow(asp, extent=ex, cmap="hsv", vmin=0, vmax=360)
    plt.colorbar(im, ax=ax[1, 0], label="deg from N (clockwise)"); ax[1, 0].set_title("aspect (downslope azimuth)")
    # hillshade for a visual sanity check
    n = terr["normals"]
    az, el = np.radians(315), np.radians(45)
    light = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    shade = np.clip((n @ light), 0, 1)
    ax[1, 1].imshow(rgb_image(rgb), extent=ex)
    ax[1, 1].imshow(np.where(m, shade, np.nan), extent=ex, cmap="gray", alpha=0.85)
    ax[1, 1].set_title("hillshade of smoothed DSM over RGB")
    for a in ax.ravel():
        zoom_to(a, fp["footprint"], pad=2)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def label_cmap(n):
    rng = np.random.default_rng(3)
    cols = rng.uniform(0.25, 0.95, size=(max(n, 1) + 1, 3))
    cols[0] = (1, 1, 1)
    return matplotlib.colors.ListedColormap(cols)


def plot_segmentation(dsm: Raster, rgb: Raster, fp: dict, seg: dict, path: Path):
    from .terrain import aspect_label
    ex = extent(dsm)
    m = fp["footprint_mask"]
    labels = seg["labels"]
    nl = int(labels.max())
    fig, ax = plt.subplots(2, 2, figsize=(16, 15))
    ax[0, 0].imshow(rgb_image(rgb), extent=ex)
    ax[0, 0].imshow(np.ma.masked_where(labels == 0, labels), extent=ex, cmap=label_cmap(nl), vmin=0, vmax=nl, alpha=0.75,
                    interpolation="nearest")
    for f in seg["faces"]:
        c = f["polygon"].representative_point()
        ax[0, 0].text(c.x, c.y, f"{f['id']}\n{aspect_label(f['aspect'])} {f['slope']:.0f}°", ha="center", va="center",
                      fontsize=7, color="k", bbox=dict(fc="w", alpha=0.6, lw=0, pad=1))
    ax[0, 0].set_title(f"{len(seg['faces'])} roof faces (region growing + RANSAC planes)")
    ax[0, 1].imshow(rgb_image(rgb), extent=ex)
    for f in seg["faces"]:
        draw_poly(ax[0, 1], f["polygon"], color="yellow", lw=1.2)
    draw_poly(ax[0, 1], fp["footprint"], color="red", lw=1.5)
    ax[0, 1].set_title("face polygons over RGB")
    im = ax[1, 0].imshow(np.where(m, seg["planarity"], np.nan), extent=ex, cmap="magma", vmin=0, vmax=0.15)
    plt.colorbar(im, ax=ax[1, 0], label="m"); ax[1, 0].set_title("local planarity (plane-fit residual std)")
    im = ax[1, 1].imshow(seg["residual"], extent=ex, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax[1, 1], label="m")
    ax[1, 1].imshow(np.ma.masked_where(~seg["unassigned"], seg["unassigned"]), extent=ex, cmap="autumn", alpha=0.9)
    ax[1, 1].set_title("measured DSM - fitted planes (unassigned px in yellow)")
    for a in ax.ravel():
        zoom_to(a, fp["footprint"], pad=2)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


OBS_COLORS = {"chimney": "red", "skylight / glazing": "cyan", "dormer": "orange", "vent / roof window": "magenta",
              "unmodelled structure": "yellow", "vegetation overhang": "lime"}


def plot_obstructions(dsm: Raster, rgb: Raster, fp: dict, seg: dict, obs: dict, path: Path):
    ex = extent(dsm)
    fig, ax = plt.subplots(2, 2, figsize=(16, 15))
    im = ax[0, 0].imshow(obs["protrusion"], extent=ex, cmap="inferno", vmin=0, vmax=1.5)
    plt.colorbar(im, ax=ax[0, 0], label="m"); ax[0, 0].set_title("protrusion above fitted roof plane")
    ax[0, 1].imshow(rgb_image(rgb), extent=ex)
    for name, cue, col in [("geometric", obs["cues"]["geometric"], "red"), ("unexplained by planes", obs["cues"]["topological"], "yellow"),
                           ("bright glazing", obs["cues"]["bright"], "cyan")]:
        ax[0, 1].contour(cue.astype(float), levels=[0.5], extent=ex, colors=col, linewidths=1, origin="upper")
        ax[0, 1].plot([], [], color=col, label=name)
    ax[0, 1].legend(loc="lower left"); ax[0, 1].set_title("obstruction cues")
    ax[1, 0].imshow(rgb_image(rgb), extent=ex)
    for o in obs["objects"]:
        col = OBS_COLORS.get(o["kind"], "white")
        fill_poly(ax[1, 0], o["polygon"], color=col, alpha=0.55)
        draw_poly(ax[1, 0], o["keepout"], color=col, lw=0.8, ls="--")
        c = o["polygon"].representative_point()
        ax[1, 0].text(c.x, c.y, f"{o['id']}", fontsize=7, ha="center", va="center")
    for k, col in OBS_COLORS.items():
        ax[1, 0].plot([], [], color=col, lw=6, label=k)
    ax[1, 0].legend(loc="lower left", fontsize=8); ax[1, 0].set_title("classified obstructions + keep-out (dashed)")
    ax[1, 1].imshow(rgb_image(rgb), extent=ex)
    for f in seg["faces"]:
        draw_poly(ax[1, 1], f["polygon"], color="yellow", lw=1)
    ax[1, 1].imshow(np.ma.masked_where(~obs["keepout_mask"], obs["keepout_mask"]), extent=ex, cmap="autumn", alpha=0.6)
    ax[1, 1].set_title("roof faces minus keep-out zones = candidate usable area")
    for a in ax.ravel():
        zoom_to(a, fp["footprint"], pad=2)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def plot_reconstruction(dsm: Raster, rgb: Raster, fp: dict, pre: dict, seg: dict, edges: list, mesh: dict, offset, path: Path):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from .reconstruct import face_colors
    ex = extent(dsm)
    m = fp["footprint_mask"]
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(2, 2, 1)
    im = ax.imshow(np.where(m, seg["recon"], np.nan), extent=ex, cmap="terrain")
    plt.colorbar(im, ax=ax, label="m")
    for e in edges:
        col = {"ridge": "red", "hip": "orange", "valley": "blue"}[e["kind"]]
        ax.plot([e["p0"][0], e["p1"][0]], [e["p0"][1], e["p1"][1]], color=col, lw=2)
    for k, col in {"ridge": "red", "hip": "orange", "valley": "blue"}.items():
        ax.plot([], [], color=col, lw=2, label=k)
    ax.legend(loc="lower left"); ax.set_title("reconstructed planar DSM + plane-intersection edges"); zoom_to(ax, fp["footprint"], 2)
    ax = fig.add_subplot(2, 2, 2)
    im = ax.imshow(np.where(m, seg["residual"], np.nan), extent=ex, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    plt.colorbar(im, ax=ax, label="m"); ax.set_title("measured - reconstructed (white = unmodelled)"); zoom_to(ax, fp["footprint"], 2)
    for view, pos in [((35, -50), 3), ((60, 20), 4)]:
        ax = fig.add_subplot(2, 2, pos, projection="3d")
        V = mesh["vertices"] - np.array(offset)
        cols = face_colors(mesh, seg["faces"]) / 255
        for kind, alpha in [("floor", 0.3), ("wall", 0.35), ("roof", 1.0), ("obstruction", 1.0)]:
            sel = mesh["kind"] == kind
            if not sel.any():
                continue
            tris = V[mesh["triangles"][sel]]
            pc = Poly3DCollection(tris, facecolors=np.c_[cols[sel], np.full(sel.sum(), alpha)], edgecolors="none")
            ax.add_collection3d(pc)
        for e in edges:
            p0, p1 = e["p0"] - offset, e["p1"] - offset
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2] + 0.05, p1[2] + 0.05], color="k", lw=1.2)
        ax.set_xlim(V[:, 0].min(), V[:, 0].max()); ax.set_ylim(V[:, 1].min(), V[:, 1].max()); ax.set_zlim(0, V[:, 2].max())
        ax.set_box_aspect((np.ptp(V[:, 0]), np.ptp(V[:, 1]), np.ptp(V[:, 2])))
        ax.view_init(*view); ax.set_title(f"3D model (elev {view[0]}, azim {view[1]})")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def plot_panels(dsm: Raster, rgb: Raster, fp: dict, seg: dict, obs: dict, res: dict, pp, path: Path):
    from .terrain import aspect_label
    ex = extent(dsm)
    status = {r["face_id"]: r for r in res["summary"]["faces"]}
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(rgb_image(rgb), extent=ex)
    for f in seg["faces"]:
        r = status[f["id"]]
        col = "lime" if r["status"] == "ok" else ("orange" if "usable" in r["status"] or "fits" in r["status"] else "red")
        draw_poly(ax[0], f["polygon"], color=col, lw=1.5)
        c = f["polygon"].representative_point()
        ax[0].text(c.x, c.y, f"{f['id']} {aspect_label(f['aspect'])}\n{r['status'].split(':')[0]}", fontsize=7, ha="center",
                   va="center", bbox=dict(fc="w", alpha=0.7, lw=0, pad=1))
    for o in obs["objects"]:
        fill_poly(ax[0], o["keepout"], color="red", alpha=0.25)
    ax[0].set_title("face suitability (green = used, orange = no room, red = rejected) and keep-out zones (red)")
    ax[1].imshow(rgb_image(rgb), extent=ex)
    for f in seg["faces"]:
        draw_poly(ax[1], f["polygon"], color="yellow", lw=0.8)
    for o in obs["objects"]:
        fill_poly(ax[1], o["polygon"], color="red", alpha=0.5)
    for x in res["panels"]:
        fill_poly(ax[1], x["polygon"], color="#1f4fd8", alpha=0.9)
        draw_poly(ax[1], x["polygon"], color="white", lw=0.4)
    s = res["summary"]
    ax[1].set_title(f"{s['n_panels']} panels ({pp.length} x {pp.width} m, {pp.power_wp:.0f} Wp) = {s['capacity_kwp']:.1f} kWp")
    for a in ax:
        zoom_to(a, fp["footprint"], pad=2)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
