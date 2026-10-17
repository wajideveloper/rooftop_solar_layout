# Technical explanation

3D roof reconstruction and solar panel placement from a 0.1 m photogrammetric DSM and an RGB orthoimage
(EPSG:32630). This document describes the approach, the key algorithms, the assumptions, the
limitations and the incomplete items. The numbers quoted are from `outputs/report.json`.

## 1. Problem framing

The input DSM is a 2.5D height field: one height per ground pixel. A roof, in contrast, is a set of
planar faces meeting along ridges, hips and valleys, with objects (chimneys, dormers, skylights) sitting
on top. The pipeline therefore does not try to "smooth" the DSM into a roof; it **explains** the DSM
with a small number of planes and treats whatever the planes cannot explain as obstructions. This gives
a model that is geometrically consistent by construction (every face is an exact plane, every edge is the
intersection of two planes) and a residual map that doubles as the obstruction detector.

Pipeline: footprint → DSM regularisation → terrain derivatives → face segmentation → obstructions →
3D reconstruction → panel layout → report. Each stage is a module in `roofsolar/` with its parameters in
`config.py`; `run_pipeline.py` runs them in sequence or one at a time with cached intermediates.

## 2. Key algorithms

### 2.1 Building footprint (`footprint.py`)
* **Ground level**: median of the lowest quartile of *smooth* pixels on the parcel (roughness < 0.1 m),
  i.e. lawn and driveway. Result 106.86 m.
* **Vegetation**: three cues OR-ed together, then dilated by 1 m: height above ground > 9.5 m, local
  roughness (std of DSM minus its 7×7 median) > 0.9 m, excess-green index > 0.10. Roughness is the
  decisive cue: canopy is rough at the decimetre scale, roofs are not.
* **Building**: pixels > 2.5 m above ground inside the parcel and not vegetation; opening with a 0.5 m
  disk; largest connected component; closing and hole filling.
* **Regularisation**: Douglas–Peucker simplification (0.25 m), then *rectilinearisation* in the frame of
  the dominant orientation (longest side of the minimum rotated rectangle): every edge is classified as
  horizontal or vertical, consecutive edges of the same class are merged with a length-weighted
  coordinate, and vertices are the intersections of consecutive lines. Accepted when IoU with the raw
  mask > 0.85 (here 8 vertices, 468.7 m²).

### 2.2 DSM regularisation (`preprocess.py`)
* Gaps: nodata / NaN pixels. Spikes: |z − 5×5 median| > 1.5 m in blobs smaller than 0.4 m² (real
  objects are larger). Both are filled by linear interpolation on the valid pixels with a
  nearest-neighbour fallback.
* Noise: 3×3 median followed by a bilateral filter (σ_colour = 0.25 m, σ_space = 3 px). The bilateral
  filter smooths within faces while keeping ridges and the step edges of chimneys sharp, which is what
  the normal estimation downstream needs.
* On the provided data: 0 gaps, 0 spikes, residual noise std 0.019 m.

### 2.3 Elevation, slope, aspect (`terrain.py`)
Horn's 3×3 kernels on the smoothed DSM. Slope in degrees from horizontal; aspect as the azimuth of the
downslope direction, clockwise from north; −1 for slope < 1°.

### 2.4 Roof face segmentation (`segmentation.py`)
1. **Local normals**: least-squares plane in a 7×7 window around every pixel, implemented with box
   filters so the whole image is O(N). The fit residual std is used as a *planarity* score.
2. **Region growing**: seeds are taken in order of increasing residual (most planar first). A
   4-neighbour joins a region if the angle between its normal and the region's running mean normal is
   < 9° **and** its distance to the region's running plane (through the running centroid) is < 0.15 m.
   The second test prevents two parallel but offset planes (e.g. a dormer roof parallel to the main
   roof) from merging. Regions smaller than 2 m² are released.
3. **RANSAC plane per region**: 200 random 3-point planes evaluated in one vectorised pass over (a
   sample of) the region's points; the plane with most inliers (|d| < 0.08 m) is refined by total least
   squares (SVD) on the inliers.
4. **Merging**: two adjacent regions are merged if their normals are within 6° and each plane predicts
   the other's centroid height within 0.10 m, or if a single plane fitted to their union has RMSE
   < 0.05 m.
5. **Competitive re-assignment**: for every pixel the heights predicted by the planes of all adjacent
   labels are compared with the measured height; the pixel takes the best label if the residual is
   < 0.35 m and improves by > 1 cm. Iterated to convergence (≤ 6 passes). Because the residual of two
   planes flips sign exactly on their intersection line, face borders converge to the true ridges, hips
   and valleys without ever computing them explicitly.
6. **Clean-up loop**: faces < 5 m² or steeper than 75° are dissolved; their pixels are re-assigned if a
   neighbour plane explains them, otherwise they stay unassigned (→ obstruction candidates). Labels are
   split into connected components and planes are refitted. Repeated until stable.

Result: 36 grown regions → 28 after merging → 16 final faces covering 87.6 % of the footprint; the
planar model reproduces the measured DSM with RMSE 0.078 m on the assigned pixels. Faces have slopes of
30–39° (slate pitch), aspects clustered at 43°, 133°, 223°, 313° (the building is rotated ~43° from
north), one flat face (the lantern base) and one 21° west-facing face at the tree-occluded corner.

### 2.5 Obstructions (`obstructions.py`)
Three cues inside the footprint:
* **geometric**: measured DSM more than 0.20 m above the plane of the nearest face (nearest by
  Euclidean distance transform, so pixels under an obstruction are compared with the roof they sit on);
* **topological**: pixels no plane could explain in 2.4;
* **radiometric**: brightness more than 40 grey levels above the median of the surrounding face. An
  absolute threshold was tried first and failed (sunlit slate on the SE face reaches 170–195, the same
  as glass); the per-face contrast isolates the lantern (+50) and the roof windows (+45–60).
Blobs < 0.15 m² are discarded, the mask is closed, hole-filled and dilated by 0.15 m, and each component
is vectorised and classified from its 95th-percentile protrusion, area, bright fraction and overlap with
the canopy mask: *chimney* (≥ 0.8 m, < 6 m²), *dormer* (≥ 0.5 m, ≥ 4 m²), *skylight/glazing*
(bright fraction > 0.35), *vent/roof window*, *vegetation overhang*, *unmodelled structure*. Keep-out
zones are the polygons buffered by 0.40 m.

### 2.6 3D reconstruction (`reconstruct.py`)
* **Planar DSM**: every face pixel replaced by its plane height; residual raster = measured − planar.
* **Edges**: for every pair of faces that share a border, the 3D intersection line of the two planes is
  computed analytically (direction = n₁ × n₂, a point from a 3×3 linear solve), clipped to the extent of
  the shared border, and classified: convex dihedral → ridge (horizontal) or hip (inclined), concave →
  valley. 6 ridges, 10 hips, 15 valleys. The mean height mismatch between adjacent planes along their
  shared border is 0.094 m (max 0.44 m at the flat lantern base, where the measured surface is a blurred
  step rather than a true intersection).
* **Mesh**: each face polygon is triangulated (Delaunay on the densified boundary plus an interior grid,
  triangles whose centroid falls outside the polygon are dropped) with z from its plane; walls are
  extruded from the footprint ring down to ground; a floor slab closes the volume; obstructions are added
  as prisms. Exported as OBJ (one group per face), colour PLY, and an interactive plotly HTML that also
  shows a sample of the measured DSM points for visual verification.

### 2.7 Panel placement (`panels.py`)
All geometry is done in the **unfolded roof plane** of each face: map → rotation so that u is along the
eave and v is the map-projected downslope direction → v is stretched by 1/cos(slope). In that frame
setbacks, clearances and module sizes are true on-roof distances (a 0.40 m setback is 0.40 m on the
roof, not 0.40 m on the map, which would over-penalise the across-slope direction by 1/cos 36° ≈ 24 %).
* Suitability: slope ≤ 60°, aspect within ±112.5° of south or flat, usable area ≥ 4 m².
* Usable region: unfolded face shrunk by 0.40 m minus unfolded obstructions grown by 0.40 m.
* Packing: rows every (module height + 2 cm); for each of 12 row phases, each row picks the u-phase
  (12 candidates) maximising the number of modules fully inside the usable region, ties resolved by
  alignment with the previous row; landscape and portrait are both evaluated; the layout with the most
  modules, then the smallest bounding box, wins. Containment tests use shapely prepared geometries.
* Fold-back: rectangle corners → map → lifted onto the face plane. The 3D rectangles have exactly the
  nominal 1.722 × 1.134 m size (max error 1e-9 m) and lie in the face plane (1e-14 m).
* Independent validation re-checks pairwise overlap, containment in the face, setback, clearance to
  obstructions and 3D size.
Result: 45 modules (18.0 kWp) on the SE (18 + 4), SW (8 + 11 + 2 + 2) faces; the four NE and four NW
faces are rejected as north-facing, the flat face is fully inside the lantern keep-out, and the
west-facing corner face has too little usable area.

## 3. Assumptions
* DSM and orthoimage are co-registered, same CRS and grid (verified: identical transforms).
* The parcel-clipped DSM is used only to delimit the property; the house is the largest elevated
  non-vegetation object on it.
* Roofs are piecewise planar; a curved roof would be approximated by several planes.
* Northern-hemisphere site (lat ≈ 52.3° N): faces facing N/NE/NW are not suitable for PV. The angular
  threshold is a parameter.
* Standard module 1.722 × 1.134 m, 400 Wp; 0.40 m edge setback; 0.40 m clearance to obstructions; 2 cm
  gaps; modules are mounted flush (same tilt and azimuth as the face).
* Obstruction height is measured relative to the nearest reconstructed face plane.

## 4. Limitations
* **Tree overhang**: the west corner of the roof is partly under a tree. That area is excluded as a
  "vegetation overhang" obstruction; the footprint there is inferred by the rectilinear regularisation,
  and the small W-facing face next to it (face 16, 21°) is probably biased by the canopy.
* **Dormers are not modelled as roof geometry**. Their own roofs are < 5 m² and the photogrammetric DSM
  blurs their walls into ~0.5 m ramps, so they are handled as obstructions with a keep-out, which is
  conservative. On the NE wing two dormers and a chimney merge into one obstruction object after
  morphological closing (#3, 38 m²); the keep-out is correct, the per-object classification is coarse.
* The flat face around the lantern is reconstructed but fully inside the lantern keep-out.
* **Yield**: the per-face "yield factor" is a geometric proxy (cosine to the optimal orientation plus an
  isotropic diffuse term). There is no irradiance or shading simulation (chimneys, dormers, trees).
* Face polygons are simplified independently; after simplification adjacent polygons can differ by a few
  centimetres along a shared edge. The pixel labelling and the analytic edge lines are exactly
  consistent, and the mismatch is reported.
* Panel packing is a phase-grid search (12 × 12 × 2 per face), not a global optimiser; mixed
  orientations inside one face are not tried.
* Parameters were tuned on this single building. They are physically motivated (metres, degrees) and
  documented, but other roof types (very low pitch, metal roofs, flat roofs with parapets) would need
  checking.
* Runtime ≈ 50 s for 47 000 roof pixels; the region growing is a Python BFS and would be the first
  thing to vectorise for larger areas.

## 5. Incomplete tasks
* No roof **topology graph** (explicit vertices where three or more planes meet). Edges are computed
  pairwise and clipped to the shared border; they meet within the border-mismatch tolerance but are not
  snapped to common vertices.
* No shading analysis and no electrical design (strings, inverter) for the panel layout.
* Flat roofs: modules would be laid flush; tilted racking with row spacing is not implemented.
* The demonstration video is produced separately (see `VIDEO_SCRIPT.md` and `make_demo_video.py`).
