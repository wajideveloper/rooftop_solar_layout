# 3D Roof Reconstruction & Solar Panel Placement

Python pipeline that takes a photogrammetric **DSM** (0.1 m) and an **RGB orthoimage** of a property,
reconstructs the rooftop as a set of geometrically consistent planar faces, detects rooftop obstructions
(chimneys, dormers, skylights, roof windows, overhanging vegetation) and lays out standard-size solar
modules on the suitable faces while respecting setbacks and keep-out zones.

Everything is plain Python (numpy / scipy / rasterio / shapely / OpenCV / matplotlib / plotly); no GDAL
command-line tools, no deep learning, no external services.

![summary](outputs/fig8_summary.png)

## Results on the provided data

| quantity | value |
|---|---|
| building footprint | 468.7 m², 8-vertex rectilinear polygon |
| planar roof faces | 16 (RMSE planar model vs measured DSM 0.078 m, 87.6 % of the footprint; the rest is obstructions) |
| roof edges recovered | 6 ridges, 10 hips, 15 valleys (plane–plane intersections) |
| obstructions | 8 (4 dormer groups, central glass lantern, roof window, tree overhang, 1 small unmodelled patch) |
| panels | **45 × 400 Wp = 18.0 kWp** (1.722 × 1.134 m modules) on 164 m² of usable roof surface |
| validation | 0 overlaps, 0 keep-out violations, 0 setback violations, 3D module size exact to 1e-9 m |

Interactive 3D model: [`outputs/roof_3d.html`](outputs/roof_3d.html) (faces, ridge lines, obstructions,
panels and a sample of the measured DSM points). Mesh: `outputs/roof_mesh.obj` / `.ply`.
Full numeric report: [`outputs/REPORT.md`](outputs/REPORT.md) and `outputs/report.json`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Tested with Python 3.13, numpy 2.3, scipy 1.16, rasterio 1.5, shapely 2.1, opencv 4.13, matplotlib 3.10, plotly 6.5.

## Running

Put `dsm_full.tif`, `dsm_clipped.tif` and `rgb.tif` next to `run_pipeline.py` (or pass `--data <folder>`), then:

```bash
python run_pipeline.py             # all 8 steps (~1 min)
python run_pipeline.py --step 4    # re-run one step; earlier results are read from outputs/cache
python run_pipeline.py --steps 7 8
```

Every step prints its key numbers, writes GeoTIFF / GeoJSON products and a figure `outputs/figN_*.png`.
All tunable parameters live in [`roofsolar/config.py`](roofsolar/config.py) with a comment each.

| step | module | what it does | main outputs |
|---|---|---|---|
| 1 | `footprint.py` | building outline from nDSM + vegetation cues, rectilinear regularisation | `footprint.geojson`, `ndsm.tif`, `fig1` |
| 2 | `preprocess.py` | nodata & spike detection, interpolation, edge-preserving smoothing | `dsm_filled.tif`, `dsm_smooth.tif`, `fig2` |
| 3 | `terrain.py` | elevation, slope, aspect (Horn), normals | `elevation.tif`, `slope_deg.tif`, `aspect_deg.tif`, `fig3` |
| 4 | `segmentation.py` | roof faces: region growing on normals → RANSAC planes → merging → competitive re-assignment | `roof_faces.geojson`, `roof_faces_labels.tif`, `dsm_planar.tif`, `fig4` |
| 5 | `obstructions.py` | chimneys / dormers / skylights from plane residuals, unexplained pixels and RGB contrast | `obstructions.geojson`, `obstruction_keepout.geojson`, `fig5` |
| 6 | `reconstruct.py` | ridge/hip/valley lines, consistency metrics, OBJ/PLY mesh, interactive 3D HTML | `roof_edges.geojson`, `roof_mesh.obj/.ply`, `roof_3d.html`, `fig6` |
| 7 | `panels.py` | panel layout per face in the unfolded roof plane + independent geometric validation | `panels.geojson`, `panel_summary.json`, `fig7` |
| 8 | `report.py` | summary figure + Markdown/JSON report | `REPORT.md`, `report.json`, `fig8` |

## Approach

### 1. Building footprint
The parcel-clipped DSM defines the property. Inside it, pixels ≥ 2.5 m above ground are either the
house or trees. Ground is the median of the lowest smooth pixels (lawn/driveway). Trees are rejected by
three cues: height > 9.5 m, high local roughness (std of DSM minus its 7×7 median, which is ~0 on planes
and large on canopy) and an excess-green index. The largest remaining component is the house; its outline
is simplified (Douglas–Peucker) and **rectilinearised**: in the frame of the dominant orientation every
edge is snapped to horizontal/vertical, consecutive collinear edges are merged with length-weighted
coordinates and the vertices are recomputed as line intersections. The regularised polygon is accepted
only if its IoU with the raw mask is > 0.85.

### 2. Regularising noisy elevations / missing data
Nodata and NaN pixels are gaps. Isolated pixels deviating > 1.5 m from the local median whose blob is
< 0.4 m² are spikes (chimneys are larger and survive). Gaps are filled by linear interpolation over valid
pixels (nearest-neighbour fallback at the boundary). Noise is reduced with a 3×3 median followed by a
**bilateral filter** (σ_colour = 0.25 m) so that ridges and step edges stay sharp. On this dataset there
are 0 gaps and 0 spikes and the residual noise is 2 cm, but the code path is exercised and generic.
The *real* regularisation happens in step 4: every face pixel is replaced by its fitted plane.

### 3. Elevation, slope, aspect
Horn's 3×3 finite differences on the smoothed DSM. Aspect is the compass azimuth of the downslope
direction (0 = N, 90 = E …), −1 for flat pixels. Written as GeoTIFFs in EPSG:32630.

### 4. Roof faces (the core)
1. **Local plane normals** by least squares in a 7×7 window (box-filter implementation, O(N)), together
   with the residual std = planarity.
2. **Region growing** seeded from the most planar pixels: a 4-neighbour joins when its normal is within
   9° of the region's running mean normal *and* it lies within 0.15 m of the region's running plane.
3. **RANSAC plane fit** per region (vectorised candidate evaluation), refined by total-least-squares on
   the inliers.
4. **Merging** of adjacent regions whose planes agree (angle < 6° and height agreement < 0.10 m at each
   other's centroid) or whose union is still planar (RMSE < 0.05 m).
5. **Competitive re-assignment**: every footprint pixel adopts the label of the 8-neighbour whose plane
   predicts the measured height best (if the residual is < 0.35 m). Iterated, this moves face borders onto
   the true plane–plane intersection lines (ridges, hips, valleys) and absorbs small leftovers.
6. **Clean-up loop**: faces < 5 m² or steeper than 75° (walls) are dissolved; neighbours absorb what their
   planes can explain; the rest stays *unassigned* and becomes an obstruction candidate. Labels are
   forced to be single connected components; planes are refitted.
7. Faces are vectorised (rasterio shapes → simplify 0.15 m) and described by plane, slope, aspect, 2D and
   true 3D area, RMSE.

Geometric consistency is measured explicitly: along every shared border the two adjacent planes are
evaluated and the mean |Δz| reported (0.094 m mean here; the maximum occurs at the flat lantern base).

### 5. Obstructions
Three cues, all inside the footprint: (a) **geometric** — measured DSM > 0.20 m above the plane of the
nearest face; (b) **topological** — pixels no plane could explain in step 4; (c) **radiometric** —
brightness more than 40 units above the median of the surrounding face (absolute thresholds fail because
sunlit slate is as bright as glass; the relative contrast isolates glazing). Blobs are cleaned, vectorised
and classified by height / area / brightness / overlap with the canopy mask into chimney, dormer,
skylight/glazing, vent/roof window, vegetation overhang or unmodelled structure. Keep-out zones are the
obstruction polygons buffered by 0.40 m.

### 6. 3D representation
* planar DSM (`dsm_planar.tif`) and residual raster;
* ridge / hip / valley segments as the analytic intersection of adjacent face planes, clipped to the
  extent of the shared border, classified by the sign of the dihedral (convex = ridge/hip, concave =
  valley) and the inclination of the line;
* a closed-ish mesh: each face polygon is triangulated (Delaunay on densified boundary + interior grid,
  triangles outside the polygon discarded) with z from its plane; vertical walls from the footprint ring
  down to ground; floor slab; obstructions as prisms. Exported as OBJ (grouped by face) and coloured PLY,
  plus an interactive plotly HTML with the measured DSM point sample for visual verification.

### 7. Panel placement
All layout geometry is done in the **unfolded roof plane** of each face (map → rotate so that u = along
the eave, v = downslope → stretch v by 1/cos slope). There, setbacks, clearances and module sizes are true
on-roof distances:

* suitability: slope ≤ 60°, aspect within ±112.5° of south (E–S–W; N/NE/NW faces are rejected for a UK
  site), usable area ≥ 4 m²;
* usable region = unfolded face shrunk by the 0.40 m edge setback minus unfolded obstructions grown by
  the 0.40 m clearance;
* **grid packing**: rows every (module height + gap); for each of 12 row phases, every row picks the
  u-phase (12 candidates) that fits the most modules fully inside the usable region, ties broken by
  alignment with the previous row (compact, column-aligned layout); landscape and portrait are both tried;
  the layout with the most modules (then the smallest bounding box) wins;
* corners are folded back to the map and lifted onto the face plane, giving exact 1.722 × 1.134 m 3D
  rectangles (checked: size error 1e-9 m, out-of-plane 1e-14 m);
* an independent validator re-checks overlaps, containment in the face, setback, clearance to
  obstructions and 3D size;
* a simple orientation yield factor (cosine to the optimal tilt/azimuth blended with a diffuse term) is
  reported per face to make the suitability decision transparent.

## Assumptions
* The DSM and the orthoimage are co-registered and in the same metric CRS (EPSG:32630, 0.1 m).
* The parcel-clipped DSM is used only to restrict the search for the building; the house is the largest
  elevated, non-vegetation object on the parcel.
* Roofs are piecewise planar. Curved surfaces would be approximated by several planes.
* Northern-hemisphere site: north-facing faces are unsuitable. The threshold (±112.5° of south) is a parameter.
* Standard module 1.722 × 1.134 m / 400 Wp, 0.40 m edge setback, 0.40 m clearance around obstructions,
  2 cm module gap. Flat faces would receive flush-mounted modules (no tilt frames) along the building axis.

## Limitations / incomplete items
* **Tree overhang**: the west corner of the roof is partly hidden by a tree; that part is excluded as an
  obstruction rather than reconstructed. The footprint there is inferred by the rectilinear regularisation.
* **Dormers** are treated as obstructions (their small roofs are < 5 m² and the DSM blurs their walls into
  ramps). Their flanks are not modelled as faces; the keep-out zone around them is therefore generous.
  Adjacent dormers and chimneys on the NE wing merge into one obstruction object (#3) after morphological
  closing; this does not affect the layout but the per-object classification is coarse there.
* The flat area around the central lantern is modelled as a face but is fully inside the lantern keep-out.
* The yield factor is a geometric proxy, not an irradiance simulation (no shading analysis by the
  chimneys, dormers or the neighbouring trees).
* Face polygons are simplified independently, so after simplification adjacent polygons can differ by
  a few centimetres along shared edges; the pixel-level labelling and the plane-intersection lines are
  exactly consistent, and the mismatch metric is reported.
* Panel packing is a grid search (12 × 12 phases × 2 orientations per face), not a global optimiser;
  mixed orientations within one face are not tried.
* No video is included in this repository; see the demonstration recorded separately.

## Repository layout

```
run_pipeline.py        CLI: run all steps or single steps
roofsolar/
  config.py            all parameters (dataclasses)
  io_utils.py          raster/vector I/O (rasterio, shapely)
  footprint.py         step 1
  preprocess.py        step 2
  terrain.py           step 3
  segmentation.py      step 4
  obstructions.py      step 5
  reconstruct.py       step 6
  panels.py            step 7
  report.py            step 8
  visualize.py         figures
outputs/               all products (created by the pipeline)
```
