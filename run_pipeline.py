"""Command-line entry point: run the whole workflow or a single step.

    python run_pipeline.py            # all steps
    python run_pipeline.py --step 3   # one step (earlier steps are loaded from outputs/cache)
    python run_pipeline.py --steps 4 5 6

Steps
  1 footprint      building outline from DSM + RGB
  2 preprocess     spike removal, gap filling, edge-preserving smoothing
  3 terrain        elevation / slope / aspect rasters
  4 segmentation   roof faces: region growing on normals + RANSAC plane fit + merging
  5 obstructions   chimneys, skylights, roof windows from plane residuals + RGB
  6 reconstruct    3D roof: planar DSM, ridge lines, OBJ/PLY mesh, interactive 3D HTML
  7 panels         solar panel layout per face (grid packing with setbacks/keep-outs)
  8 report         summary figures + JSON/Markdown report
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from roofsolar.config import Config
from roofsolar import io_utils as io


def cache_path(cfg: Config, name: str) -> Path:
    return cfg.paths.out_dir / "cache" / f"{name}.pkl"


def save(cfg, name, obj):
    p = cache_path(cfg, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def load(cfg, name):
    with open(cache_path(cfg, name), "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------- steps
def step1_footprint(cfg: Config):
    from roofsolar.footprint import extract_footprint
    from roofsolar.visualize import plot_footprint

    dsm = io.read_raster(cfg.paths.dsm_full, masked=False)
    clipped = io.read_raster(cfg.paths.dsm_clipped, masked=True)
    rgb = io.read_raster(cfg.paths.rgb, masked=False)
    # the clipped DSM has the same grid origin but a smaller window -> rebuild the parcel mask on the full grid
    parcel = io.polygons_to_mask(io.mask_to_polygons(~np.ma.getmaskarray(clipped.data), clipped.transform),
                                 dsm.shape, dsm.transform)
    fp = extract_footprint(dsm, parcel, rgb, cfg.footprint)
    fp["footprint_mask"] = io.polygons_to_mask([fp["footprint"]], dsm.shape, dsm.transform)
    out = cfg.paths.out_dir
    io.write_geojson(out / "footprint.geojson",
                     [{"geometry": fp["footprint"], "properties": {"kind": "regularised", "area_m2": fp["footprint"].area}},
                      {"geometry": fp["footprint_raw"], "properties": {"kind": "raw", "area_m2": fp["footprint_raw"].area}}],
                     crs_epsg=32630)
    io.write_raster(out / "ndsm.tif", dsm, fp["ndsm"])
    plot_footprint(dsm, rgb, fp, out / "fig1_footprint.png")
    print(f"  ground elevation      : {fp['ground_z']:.2f} m")
    print(f"  raw footprint area    : {fp['footprint_raw'].area:.1f} m2")
    print(f"  regularised footprint : {fp['footprint'].area:.1f} m2, {len(fp['footprint'].exterior.coords) - 1} vertices")
    save(cfg, "step1", {"dsm": dsm, "rgb": rgb, "parcel": parcel, "fp": fp})


def step2_preprocess(cfg: Config):
    from roofsolar.preprocess import preprocess_dsm
    from roofsolar.visualize import plot_preprocess

    s1 = load(cfg, "step1")
    dsm, fp = s1["dsm"], s1["fp"]
    pre = preprocess_dsm(dsm.data, fp["footprint_mask"], dsm.res, cfg.preprocess, nodata=dsm.nodata)
    out = cfg.paths.out_dir
    io.write_raster(out / "dsm_filled.tif", dsm, np.where(fp["footprint_mask"], pre["dsm_filled"], -9999))
    io.write_raster(out / "dsm_smooth.tif", dsm, np.where(fp["footprint_mask"], pre["dsm_smooth"], -9999))
    plot_preprocess(dsm, fp, pre, out / "fig2_preprocess.png")
    print(f"  nodata gaps in footprint : {pre['n_gaps']} px")
    print(f"  spikes removed           : {pre['n_spikes']} px")
    print(f"  residual noise (raw-smooth) std : {pre['noise_std']:.3f} m")
    save(cfg, "step2", pre)


def step3_terrain(cfg: Config):
    from roofsolar.terrain import slope_aspect, normals_from_gradient
    from roofsolar.visualize import plot_terrain

    s1, pre = load(cfg, "step1"), load(cfg, "step2")
    dsm, fp = s1["dsm"], s1["fp"]
    m = fp["footprint_mask"]
    slope, aspect, gx, gy = slope_aspect(pre["dsm_smooth"], dsm.res)
    normals = normals_from_gradient(gx, gy)
    out = cfg.paths.out_dir
    io.write_raster(out / "elevation.tif", dsm, np.where(m, pre["dsm_smooth"], -9999))
    io.write_raster(out / "slope_deg.tif", dsm, np.where(m, slope, -9999))
    io.write_raster(out / "aspect_deg.tif", dsm, np.where(m, aspect, -9999))
    terr = {"slope": slope, "aspect": aspect, "normals": normals.astype(np.float32)}
    plot_terrain(dsm, s1["rgb"], fp, pre, terr, out / "fig3_terrain.png")
    print(f"  elevation range on roof : {pre['dsm_smooth'][m].min():.2f} .. {pre['dsm_smooth'][m].max():.2f} m")
    print(f"  median slope on roof    : {np.median(slope[m]):.1f} deg")
    hist, edges = np.histogram(aspect[m & (slope > 5)], bins=[0, 45, 90, 135, 180, 225, 270, 315, 360])
    print("  aspect histogram (sloped px): " + ", ".join(f"{int(e)}-{int(e2)}:{h}" for e, e2, h in zip(edges[:-1], edges[1:], hist)))
    save(cfg, "step3", terr)


def step4_segmentation(cfg: Config):
    from roofsolar.segmentation import segment_roof
    from roofsolar.terrain import aspect_label
    from roofsolar.visualize import plot_segmentation

    s1, pre = load(cfg, "step1"), load(cfg, "step2")
    dsm, fp = s1["dsm"], s1["fp"]
    seg = segment_roof(pre["dsm_smooth"], pre["dsm_filled"], fp["footprint_mask"], dsm.res, dsm.transform,
                       cfg.segmentation)
    out = cfg.paths.out_dir
    io.write_raster(out / "roof_faces_labels.tif", dsm, seg["labels"], nodata=0, dtype="int32")
    io.write_raster(out / "dsm_planar.tif", dsm, np.where(np.isfinite(seg["recon"]), seg["recon"], -9999))
    io.write_geojson(out / "roof_faces.geojson", [
        {"geometry": f["polygon"], "properties": {k: f[k] for k in ("id", "slope", "aspect", "area_2d", "area_3d", "rmse", "rmse_meas", "n_px")}
         | {"orientation": aspect_label(f["aspect"])}} for f in seg["faces"]], crs_epsg=32630)
    plot_segmentation(dsm, s1["rgb"], fp, seg, out / "fig4_faces.png")
    st = seg["stats"]
    print(f"  regions after growing: {st['regions_after_growing']}, after merge: {st['after_merge']}, final faces: {st['faces']}")
    print(f"  footprint coverage by planar faces: {100 * st['coverage']:.1f}% (unassigned {st['unassigned_px']} px)")
    print(f"  {'id':>3} {'orient':>6} {'slope':>6} {'aspect':>7} {'area2d':>7} {'area3d':>7} {'rmse':>6}")
    for f in seg["faces"]:
        print(f"  {f['id']:>3} {aspect_label(f['aspect']):>6} {f['slope']:6.1f} {f['aspect']:7.1f} {f['area_2d']:7.1f} {f['area_3d']:7.1f} {f['rmse']:6.3f}")
    save(cfg, "step4", seg)


def step5_obstructions(cfg: Config):
    from roofsolar.obstructions import detect_obstructions
    from roofsolar.visualize import plot_obstructions

    s1, pre, seg = load(cfg, "step1"), load(cfg, "step2"), load(cfg, "step4")
    dsm, rgb, fp = s1["dsm"], s1["rgb"], s1["fp"]
    X, Y = dsm.xy_grid()
    obs = detect_obstructions(pre["dsm_filled"], seg["labels"], seg["faces"], fp["footprint_mask"],
                              np.asarray(rgb.data), X, Y, dsm.res, dsm.transform, cfg.obstruction,
                              tree_mask=fp["tree_mask"])
    out = cfg.paths.out_dir
    io.write_geojson(out / "obstructions.geojson", [
        {"geometry": o["polygon"], "properties": {k: o[k] for k in ("id", "kind", "area_m2", "height_m", "bright_fraction")}}
        for o in obs["objects"]], crs_epsg=32630)
    io.write_geojson(out / "obstruction_keepout.geojson", [
        {"geometry": o["keepout"], "properties": {"id": o["id"], "kind": o["kind"]}} for o in obs["objects"]], crs_epsg=32630)
    io.write_raster(out / "obstruction_mask.tif", dsm, obs["mask"].astype(np.uint8), nodata=255, dtype="uint8")
    plot_obstructions(dsm, rgb, fp, seg, obs, out / "fig5_obstructions.png")
    print(f"  {len(obs['objects'])} obstructions, total {sum(o['area_m2'] for o in obs['objects']):.1f} m2, "
          f"keep-out {obs['keepout_mask'].sum() * dsm.res ** 2:.1f} m2")
    for o in obs["objects"]:
        print(f"   #{o['id']:<3} {o['kind']:<22} area {o['area_m2']:5.1f} m2  height {o['height_m']:4.2f} m  bright {o['bright_fraction']:.2f}")
    save(cfg, "step5", obs)


def step6_reconstruct(cfg: Config, panels=None):
    from roofsolar.reconstruct import roof_edges, build_mesh, write_obj, write_ply, face_colors, write_html
    from roofsolar.visualize import plot_reconstruction

    s1, pre, seg, obs = load(cfg, "step1"), load(cfg, "step2"), load(cfg, "step4"), load(cfg, "step5")
    dsm, rgb, fp = s1["dsm"], s1["rgb"], s1["fp"]
    edges, cons = roof_edges(seg["faces"], seg["labels"], dsm.transform)
    mesh = build_mesh(seg["faces"], fp["footprint"], fp["ground_z"], obs["objects"])
    out = cfg.paths.out_dir
    offset = (float(np.floor(fp["footprint"].centroid.x)), float(np.floor(fp["footprint"].centroid.y)), float(np.floor(fp["ground_z"])))
    write_obj(out / "roof_mesh.obj", mesh, offset)
    write_ply(out / "roof_mesh.ply", mesh, face_colors(mesh, seg["faces"]), offset)
    io.write_geojson(out / "roof_edges.geojson", [
        {"geometry": e["geometry"], "properties": {"kind": e["kind"], "face_a": e["faces"][0], "face_b": e["faces"][1],
                                                   "z0": e["p0"][2], "z1": e["p1"][2], "mean_abs_dz": e["mean_abs_dz"]}}
        for e in edges], crs_epsg=32630)
    # sample of measured DSM points for the HTML viewer
    m = fp["footprint_mask"]
    X, Y = dsm.xy_grid()
    rr, cc = np.nonzero(m)
    sel = np.random.default_rng(0).choice(len(rr), min(6000, len(rr)), replace=False)
    pts = np.column_stack([X[rr[sel], cc[sel]], Y[rr[sel], cc[sel]], pre["dsm_filled"][rr[sel], cc[sel]]])
    write_html(out / "roof_3d.html", mesh, seg["faces"], edges, panels, obs["objects"], offset, pts)
    plot_reconstruction(dsm, rgb, fp, pre, seg, edges, mesh, offset, out / "fig6_reconstruction.png")
    io.save_json(out / "reconstruction_consistency.json", cons)
    print(f"  mesh: {len(mesh['vertices'])} vertices, {len(mesh['triangles'])} triangles "
          f"({(mesh['kind'] == 'roof').sum()} roof, {(mesh['kind'] == 'wall').sum()} wall, {(mesh['kind'] == 'obstruction').sum()} obstruction)")
    print(f"  roof edges: {sum(e['kind'] == 'ridge' for e in edges)} ridges, {sum(e['kind'] == 'hip' for e in edges)} hips, "
          f"{sum(e['kind'] == 'valley' for e in edges)} valleys")
    print(f"  plane consistency along shared borders: mean |dz| = {cons['mean_border_dz']:.3f} m, max = {cons['max_border_dz']:.3f} m")
    rm = np.sqrt(np.nanmean(seg["residual"][m] ** 2))
    print(f"  planar model vs measured DSM (assigned px): RMSE = {rm:.3f} m")
    save(cfg, "step6", {"edges": edges, "mesh": mesh, "offset": offset, "consistency": cons})


def step7_panels(cfg: Config):
    from roofsolar.panels import place_panels, validate_layout
    from roofsolar.reconstruct import write_html
    from roofsolar.visualize import plot_panels

    s1, pre, seg, obs, rec = load(cfg, "step1"), load(cfg, "step2"), load(cfg, "step4"), load(cfg, "step5"), load(cfg, "step6")
    dsm, rgb, fp = s1["dsm"], s1["rgb"], s1["fp"]
    res = place_panels(seg["faces"], obs["objects"], fp["footprint"], cfg.panels, cfg.obstruction.clearance)
    val = validate_layout(res["panels"], seg["faces"], obs["objects"], cfg.panels, cfg.obstruction.clearance)
    out = cfg.paths.out_dir
    io.write_geojson(out / "panels.geojson", [
        {"geometry": x["polygon"], "properties": {k: x[k] for k in ("id", "face_id", "orientation", "tilt_deg", "azimuth_deg",
                                                                     "width_m", "height_m", "area_m2", "power_wp", "yield_factor")}
         | {"corners3d": x["corners3d"]}} for x in res["panels"]], crs_epsg=32630)
    io.save_json(out / "panel_summary.json", {"summary": res["summary"], "validation": val})
    # regenerate the interactive 3D with the panels
    m = fp["footprint_mask"]; X, Y = dsm.xy_grid(); rr, cc = np.nonzero(m)
    sel = np.random.default_rng(0).choice(len(rr), min(6000, len(rr)), replace=False)
    pts = np.column_stack([X[rr[sel], cc[sel]], Y[rr[sel], cc[sel]], pre["dsm_filled"][rr[sel], cc[sel]]])
    write_html(out / "roof_3d.html", rec["mesh"], seg["faces"], rec["edges"], res["panels"], obs["objects"], rec["offset"], pts)
    plot_panels(dsm, rgb, fp, seg, obs, res, cfg.panels, out / "fig7_panels.png")
    s = res["summary"]
    print(f"  panels placed: {s['n_panels']}  ({s['capacity_kwp']:.1f} kWp, {s['panel_area_m2']:.1f} m2 of modules on "
          f"{s['usable_area_m2']:.1f} m2 usable roof)")
    print(f"  {'face':>4} {'slope':>5} {'aspect':>6} {'yield':>5} {'usable':>6} {'n':>3} {'orient':>9}  status")
    for r in s["faces"]:
        print(f"  {r['face_id']:>4} {r['slope']:5.1f} {r['aspect']:6.1f} {r['yield_factor']:5.2f} {r.get('usable_area', 0):6.1f} "
              f"{r['n_panels']:>3} {r.get('orientation', '-'):>9}  {r['status']}")
    print("  validation: " + ", ".join(f"{k}={v}" for k, v in val.items()))
    save(cfg, "step7", {"result": res, "validation": val})


def step8_report(cfg: Config):
    from roofsolar.report import summary_figure, write_report

    s1, pre, seg, obs, rec, p7 = (load(cfg, k) for k in ("step1", "step2", "step4", "step5", "step6", "step7"))
    dsm, rgb, fp = s1["dsm"], s1["rgb"], s1["fp"]
    out = cfg.paths.out_dir
    summary_figure(dsm, rgb, fp, pre, seg, obs, p7["result"], rec, out / "fig8_summary.png")
    data = write_report(out, cfg, fp, pre, seg, obs, p7["result"], rec, p7["validation"])
    print(f"  wrote {out / 'REPORT.md'}, {out / 'report.json'}, {out / 'fig8_summary.png'}")
    print(f"  faces {len(seg['faces'])}, obstructions {len(obs['objects'])}, panels {data['panels']['n_panels']} "
          f"({data['panels']['capacity_kwp']:.1f} kWp)")


STEPS = {1: step1_footprint, 2: step2_preprocess, 3: step3_terrain, 4: step4_segmentation, 5: step5_obstructions,
         6: step6_reconstruct, 7: step7_panels, 8: step8_report}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int)
    ap.add_argument("--steps", type=int, nargs="+")
    ap.add_argument("--data", type=Path, default=Path("."), help="folder with dsm_full.tif, dsm_clipped.tif, rgb.tif")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    cfg = Config()
    cfg.paths.dsm_full = args.data / "dsm_full.tif"
    cfg.paths.dsm_clipped = args.data / "dsm_clipped.tif"
    cfg.paths.rgb = args.data / "rgb.tif"
    cfg.paths.out_dir = args.out
    steps = args.steps or ([args.step] if args.step else sorted(STEPS))
    for s in steps:
        t0 = time.time()
        print(f"\n=== Step {s}: {STEPS[s].__name__} ===")
        STEPS[s](cfg)
        print(f"  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
