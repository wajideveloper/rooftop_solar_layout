# Visual results

All figures are produced by `python run_pipeline.py` into `outputs/`. Coordinates are EPSG:32630 metres.

## Overview

![summary](outputs/fig8_summary.png)

## 1. Building footprint
Raw mask (yellow) and the rectilinear regularised outline (red) over the orthoimage; nDSM; vegetation mask; building mask.

![footprint](outputs/fig1_footprint.png)

## 2. DSM regularisation
Raw vs filled/smoothed DSM, the removed noise, and the gap/spike map (none on this dataset).

![preprocess](outputs/fig2_preprocess.png)

## 3. Elevation, slope, aspect
Horn slope and downslope azimuth, plus a hillshade of the smoothed DSM for a visual sanity check.

![terrain](outputs/fig3_terrain.png)

## 4. Reconstructed roof faces
16 planar faces (region growing + RANSAC), their polygons, the local planarity score and the residual of the measured DSM to the fitted planes. Yellow pixels are explained by no plane and become obstruction candidates.

![faces](outputs/fig4_faces.png)

## 5. Identified obstructions
Protrusion above the fitted roof plane, the three detection cues, the classified objects with their keep-out buffers, and the candidate usable area.

![obstructions](outputs/fig5_obstructions.png)

## 6. 3D reconstruction
Planar DSM with ridge (red), hip (orange) and valley (blue) lines computed as plane intersections; residual map; two views of the mesh. The interactive version is `outputs/roof_3d.html` (open in a browser; toggle the measured DSM point sample in the legend).

![reconstruction](outputs/fig6_reconstruction.png)

## 7. Final panel placement
Face suitability and keep-out zones (left); 45 modules of 1.722 × 1.134 m = 18.0 kWp (right).

![panels](outputs/fig7_panels.png)
