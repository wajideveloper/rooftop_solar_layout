"""Step 8 - summary figure + machine-readable and human-readable report."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import Raster, save_json
from .terrain import aspect_label
from .visualize import extent, rgb_image, draw_poly, fill_poly, zoom_to, label_cmap


def summary_figure(dsm: Raster, rgb: Raster, fp: dict, pre: dict, seg: dict, obs: dict, pan: dict, rec: dict, path: Path):
    ex = extent(dsm)
    m = fp["footprint_mask"]
    labels = seg["labels"]
    fig, ax = plt.subplots(2, 3, figsize=(24, 15))
    ax[0, 0].imshow(rgb_image(rgb), extent=ex); draw_poly(ax[0, 0], fp["footprint"], color="red", lw=2)
    ax[0, 0].set_title("1. input RGB + extracted building footprint")
    im = ax[0, 1].imshow(np.where(m, pre["dsm_smooth"], np.nan), extent=ex, cmap="terrain"); plt.colorbar(im, ax=ax[0, 1], label="m")
    ax[0, 1].set_title("2-3. regularised DSM (elevation)")
    ax[0, 2].imshow(rgb_image(rgb), extent=ex)
    ax[0, 2].imshow(np.ma.masked_where(labels == 0, labels), extent=ex, cmap=label_cmap(int(labels.max())), vmin=0,
                    vmax=int(labels.max()), alpha=0.75, interpolation="nearest")
    for e in rec["edges"]:
        ax[0, 2].plot([e["p0"][0], e["p1"][0]], [e["p0"][1], e["p1"][1]], color="k", lw=1.2)
    ax[0, 2].set_title(f"4. {len(seg['faces'])} planar roof faces + ridge/hip/valley lines")
    ax[1, 0].imshow(rgb_image(rgb), extent=ex)
    for o in obs["objects"]:
        fill_poly(ax[1, 0], o["polygon"], color="red", alpha=0.6)
        draw_poly(ax[1, 0], o["keepout"], color="red", lw=0.8, ls="--")
    ax[1, 0].set_title(f"5. {len(obs['objects'])} obstructions + keep-out zones")
    im = ax[1, 1].imshow(np.where(m, seg["residual"], np.nan), extent=ex, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    plt.colorbar(im, ax=ax[1, 1], label="m"); ax[1, 1].set_title("6. measured DSM - reconstructed planar roof")
    ax[1, 2].imshow(rgb_image(rgb), extent=ex)
    for f in seg["faces"]:
        draw_poly(ax[1, 2], f["polygon"], color="yellow", lw=0.7)
    for o in obs["objects"]:
        fill_poly(ax[1, 2], o["polygon"], color="red", alpha=0.5)
    for x in pan["panels"]:
        fill_poly(ax[1, 2], x["polygon"], color="#1f4fd8", alpha=0.9); draw_poly(ax[1, 2], x["polygon"], color="w", lw=0.4)
    s = pan["summary"]
    ax[1, 2].set_title(f"7. {s['n_panels']} panels = {s['capacity_kwp']:.1f} kWp")
    for a in ax.ravel():
        zoom_to(a, fp["footprint"], 2)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def write_report(out: Path, cfg, fp, pre, seg, obs, pan, rec, val, timings: dict | None = None):
    faces = seg["faces"]
    s = pan["summary"]
    m = fp["footprint_mask"]
    rmse = float(np.sqrt(np.nanmean(seg["residual"][m] ** 2)))
    data = {
        "footprint": {"area_m2": fp["footprint"].area, "vertices": len(fp["footprint"].exterior.coords) - 1,
                      "ground_elevation_m": fp["ground_z"]},
        "preprocessing": {"gaps_filled_px": pre["n_gaps"], "spikes_removed_px": pre["n_spikes"], "noise_std_m": pre["noise_std"]},
        "segmentation": seg["stats"] | {"rmse_planar_model_m": rmse},
        "faces": [{k: f[k] for k in ("id", "slope", "aspect", "area_2d", "area_3d", "rmse", "rmse_meas")} | {"orientation": aspect_label(f["aspect"])}
                  for f in faces],
        "obstructions": [{k: o[k] for k in ("id", "kind", "area_m2", "height_m")} for o in obs["objects"]],
        "reconstruction": {k: rec["consistency"][k] for k in ("n_adjacent_pairs", "mean_border_dz", "max_border_dz")} |
                          {"edges": {k: sum(e["kind"] == k for e in rec["edges"]) for k in ("ridge", "hip", "valley")}},
        "panels": {k: s[k] for k in ("n_panels", "capacity_kwp", "panel_area_m2", "usable_area_m2", "weighted_yield_factor")},
        "panel_faces": s["faces"],
        "validation": val,
        "parameters": {k: vars(getattr(cfg, k)) for k in ("footprint", "preprocess", "segmentation", "obstruction", "panels")},
    }
    save_json(out / "report.json", data)

    L = []
    L.append("# Roof reconstruction & solar layout - results\n")
    L.append(f"Building footprint: **{fp['footprint'].area:.1f} m²** ({len(fp['footprint'].exterior.coords) - 1} vertices, rectilinear), "
             f"ground at {fp['ground_z']:.2f} m.\n")
    L.append(f"DSM preprocessing: {pre['n_gaps']} nodata px filled, {pre['n_spikes']} spike px removed, residual noise std {pre['noise_std']:.3f} m.\n")
    L.append(f"Roof model: **{len(faces)} planar faces** covering {100 * seg['stats']['coverage']:.1f}% of the footprint "
             f"(the rest is obstructions), RMSE planar model vs measured DSM **{rmse:.3f} m**; "
             f"{data['reconstruction']['edges']['ridge']} ridges, {data['reconstruction']['edges']['hip']} hips, "
             f"{data['reconstruction']['edges']['valley']} valleys; mean height mismatch between adjacent planes along their "
             f"shared border {rec['consistency']['mean_border_dz']:.3f} m.\n")
    L.append("\n## Roof faces\n\n| id | orientation | slope ° | aspect ° | area 2D m² | area 3D m² | RMSE m | panels |\n|---|---|---|---|---|---|---|---|")
    pf = {r["face_id"]: r for r in s["faces"]}
    for f in faces:
        r = pf[f["id"]]
        L.append(f"| {f['id']} | {aspect_label(f['aspect'])} | {f['slope']:.1f} | {f['aspect']:.0f} | {f['area_2d']:.1f} | {f['area_3d']:.1f} | "
                 f"{f['rmse']:.3f} | {r['n_panels']} ({r['status']}) |")
    L.append("\n## Obstructions\n\n| id | kind | area m² | height above roof m |\n|---|---|---|---|")
    for o in obs["objects"]:
        L.append(f"| {o['id']} | {o['kind']} | {o['area_m2']:.1f} | {o['height_m']:.2f} |")
    L.append(f"\n## Panel layout\n\n* modules: {cfg.panels.length} x {cfg.panels.width} m, {cfg.panels.power_wp:.0f} Wp, "
             f"edge setback {cfg.panels.edge_setback} m, obstruction clearance {cfg.obstruction.clearance} m")
    L.append(f"* **{s['n_panels']} panels, {s['capacity_kwp']:.1f} kWp**, {s['panel_area_m2']:.1f} m² of modules on {s['usable_area_m2']:.1f} m² usable roof surface")
    L.append(f"* mean orientation yield factor of the placed panels: {s['weighted_yield_factor']:.2f} (1.0 = optimal tilt/azimuth)")
    L.append("\n### Geometric validation\n")
    for k, v in val.items():
        L.append(f"* {k}: {v}")
    L.append("\n## Figures\n")
    for i, cap in enumerate(["footprint", "preprocess", "terrain", "faces", "obstructions", "reconstruction", "panels"], 1):
        L.append(f"![{cap}](fig{i}_{cap}.png)\n")
    L.append("![summary](fig8_summary.png)\n")
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    return data
