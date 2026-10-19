# Roof reconstruction & solar layout - results

Building footprint: **468.7 m²** (8 vertices, rectilinear), ground at 106.86 m.

DSM preprocessing: 0 nodata px filled, 0 spike px removed, residual noise std 0.023 m.

Roof model: **16 planar faces** covering 85.9% of the footprint (the rest is obstructions), RMSE planar model vs measured DSM **0.077 m**; 7 ridges, 12 hips, 12 valleys; mean height mismatch between adjacent planes along their shared border 0.117 m.


## Roof faces

| id | orientation | slope ° | aspect ° | area 2D m² | area 3D m² | RMSE m | panels |
|---|---|---|---|---|---|---|---|
| 1 | SW | 35.7 | 224 | 12.8 | 15.7 | 0.033 | 2 (ok) |
| 2 | SE | 36.0 | 133 | 56.9 | 70.3 | 0.028 | 18 (ok) |
| 3 | NW | 38.4 | 314 | 21.3 | 27.2 | 0.057 | 0 (rejected: aspect (north-facing)) |
| 4 | NE | 37.2 | 44 | 23.4 | 29.4 | 0.064 | 0 (rejected: aspect (north-facing)) |
| 5 | SW | 29.7 | 222 | 14.7 | 16.9 | 0.059 | 2 (ok) |
| 6 | SW | 35.0 | 223 | 75.3 | 91.9 | 0.066 | 12 (ok) |
| 7 | SW | 36.8 | 223 | 31.8 | 39.6 | 0.038 | 8 (ok) |
| 8 | NE | 37.4 | 43 | 28.4 | 35.7 | 0.033 | 0 (rejected: aspect (north-facing)) |
| 9 | NE | 31.3 | 43 | 21.6 | 25.2 | 0.103 | 0 (rejected: aspect (north-facing)) |
| 10 | SW | 13.2 | 218 | 12.5 | 12.8 | 0.114 | 0 (rejected: not enough usable area) |
| 11 | NW | 34.2 | 308 | 11.4 | 13.8 | 0.088 | 0 (rejected: aspect (north-facing)) |
| 12 | NW | 37.9 | 317 | 13.1 | 16.6 | 0.079 | 0 (rejected: aspect (north-facing)) |
| 13 | W | 20.9 | 284 | 12.7 | 13.6 | 0.089 | 0 (rejected: not enough usable area) |
| 14 | NW | 14.4 | 313 | 10.1 | 10.5 | 0.092 | 0 (rejected: aspect (north-facing)) |
| 15 | SE | 35.8 | 132 | 19.8 | 24.4 | 0.066 | 5 (ok) |
| 16 | NE | 35.5 | 43 | 36.9 | 45.4 | 0.061 | 0 (rejected: aspect (north-facing)) |

## Obstructions

| id | kind | area m² | height above roof m |
|---|---|---|---|
| 1 | vegetation overhang | 3.4 | 1.25 |
| 2 | unmodelled structure | 0.3 | 0.00 |
| 3 | dormer | 10.3 | 1.93 |
| 4 | skylight / glazing | 1.0 | 0.04 |
| 5 | dormer | 7.2 | 0.57 |
| 6 | dormer | 26.5 | 1.67 |
| 7 | dormer | 22.8 | 0.88 |
| 8 | dormer | 11.6 | 2.04 |
| 9 | skylight / glazing | 20.5 | 0.86 |

## Panel layout

* modules: 1.722 x 1.134 m, 400 Wp, edge setback 0.4 m, obstruction clearance 0.4 m
* **47 panels, 18.8 kWp**, 91.8 m² of modules on 169.6 m² usable roof surface
* mean orientation yield factor of the placed panels: 0.92 (1.0 = optimal tilt/azimuth)

### Geometric validation

* n_panels: 47
* overlapping_pairs: 0
* panels_closer_than_clearance_to_obstruction: 3
* panels_outside_face: 0
* setback_violations: 0
* max_3d_size_error_m: 9.194189853900525e-10
* max_out_of_plane_m: 1.0778602536871335e-14

## Figures

![footprint](fig1_footprint.png)

![preprocess](fig2_preprocess.png)

![terrain](fig3_terrain.png)

![faces](fig4_faces.png)

![obstructions](fig5_obstructions.png)

![reconstruction](fig6_reconstruction.png)

![panels](fig7_panels.png)

![summary](fig8_summary.png)
