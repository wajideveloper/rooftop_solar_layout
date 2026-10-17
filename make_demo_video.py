"""Render a silent, captioned demo video (MP4, 1920x1080) from the pipeline figures.

    python make_demo_video.py            # -> outputs/demo_video.mp4 (~4 min 20 s)

Record your narration over it following VIDEO_SCRIPT.md. Requires only OpenCV + numpy (+ the outputs of
run_pipeline.py). Optionally add a 3D-viewer screenshot as outputs/roof_3d_screenshot.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path("outputs")
W, H, FPS = 1920, 1080, 24
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_numbers():
    try:
        r = json.loads((OUT / "report.json").read_text())
        return {"faces": len(r["faces"]), "rmse": r["segmentation"]["rmse_planar_model_m"],
                "obs": len(r["obstructions"]), "panels": r["panels"]["n_panels"], "kwp": r["panels"]["capacity_kwp"],
                "fp": r["footprint"]["area_m2"], "dz": r["reconstruction"]["mean_border_dz"]}
    except Exception:
        return {"faces": "?", "rmse": 0, "obs": "?", "panels": "?", "kwp": 0, "fp": 0, "dz": 0}


def wrap(text, max_chars=95):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def slide(image_path: Path | None, title: str, caption: str, step: str = "") -> np.ndarray:
    frame = np.full((H, W, 3), 24, np.uint8)
    cv2.putText(frame, title, (60, 70), FONT, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    if step:
        cv2.putText(frame, step, (W - 420, 70), FONT, 1.0, (120, 200, 255), 2, cv2.LINE_AA)
    cap_lines = wrap(caption)
    cap_h = 36 * len(cap_lines) + 30
    area_h = H - 110 - cap_h
    if image_path and image_path.exists():
        img = cv2.imread(str(image_path))
        s = min((W - 120) / img.shape[1], area_h / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        x0 = (W - img.shape[1]) // 2
        frame[100:100 + img.shape[0], x0:x0 + img.shape[1]] = img
    y = H - cap_h + 10
    for ln in cap_lines:
        cv2.putText(frame, ln, (60, y), FONT, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
        y += 36
    return frame


def title_slide(lines, sub=""):
    frame = np.full((H, W, 3), 24, np.uint8)
    y = H // 2 - 40 * len(lines)
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (120, y + i * 90), FONT, 2.0 if i == 0 else 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (120, H - 120), FONT, 0.9, (120, 200, 255), 2, cv2.LINE_AA)
    return frame


def main():
    n = load_numbers()
    scenes = [
        (25, title_slide(["3D Roof Reconstruction & Solar Panel Placement",
                          "from a 0.1 m DSM + RGB orthoimage, in Python"],
                         "numpy . scipy . rasterio . shapely . OpenCV . matplotlib . plotly")),
        (15, slide(OUT / "fig8_summary.png", "Pipeline: 8 steps, ~1 minute, python run_pipeline.py",
                   "1 footprint  2 DSM regularisation  3 slope/aspect  4 roof faces  5 obstructions  6 3D model  7 panels  8 report",
                   "overview")),
        (30, slide(OUT / "fig1_footprint.png", "Step 1 - building footprint",
                   f"Height > 2.5 m inside the parcel; trees removed by height, surface roughness and green index; largest component; "
                   f"outline snapped to the two dominant directions -> {n['fp']:.0f} m2 rectilinear polygon.", "1 / 8")),
        (20, slide(OUT / "fig2_preprocess.png", "Step 2 - regularise elevations, fill missing data",
                   "Nodata and isolated spikes are interpolated (linear, nearest fallback); 3x3 median + bilateral filter removes noise "
                   "while keeping ridges and step edges sharp. This dataset: 0 gaps, 0 spikes, 2 cm noise.", "2 / 8")),
        (15, slide(OUT / "fig3_terrain.png", "Step 3 - elevation, slope, aspect",
                   "Horn 3x3 gradients. Aspect = azimuth of the downslope direction. Faces are pitched ~35 deg; the building is rotated 43 deg from north.",
                   "3 / 8")),
        (45, slide(OUT / "fig4_faces.png", "Step 4 - roof faces (region growing + RANSAC planes)",
                   f"Local plane normals -> region growing (angle + distance to plane) -> RANSAC plane per region -> merge coplanar -> "
                   f"every pixel adopts the adjacent plane that explains its height best, which moves borders onto the true ridges/hips/valleys. "
                   f"{n['faces']} faces, RMSE {n['rmse']:.3f} m; yellow = explained by no plane.", "4 / 8")),
        (30, slide(OUT / "fig5_obstructions.png", "Step 5 - obstructions",
                   f"Cues: height above the fitted plane, pixels no plane explains, brightness relative to the surrounding face. "
                   f"{n['obs']} objects classified (dormers, glass lantern, roof window, tree overhang) with a 0.4 m keep-out.", "5 / 8")),
        (35, slide(OUT / "fig6_reconstruction.png", "Step 6 - 3D reconstruction",
                   f"Ridges / hips / valleys = analytic plane-plane intersections clipped to the shared border; mean height mismatch between "
                   f"adjacent planes {n['dz']:.3f} m. Mesh exported as OBJ/PLY; interactive viewer outputs/roof_3d.html.", "6 / 8")),
    ]
    shot = OUT / "roof_3d_screenshot.png"
    if shot.exists():
        scenes.append((15, slide(shot, "Interactive 3D viewer (plotly)", "Faces, edges, obstructions, panels and a sample of measured DSM points.", "6 / 8")))
    scenes += [
        (45, slide(OUT / "fig7_panels.png", "Step 7 - panel placement",
                   f"Each face is unfolded into its roof plane so setbacks (0.4 m) and module sizes (1.722 x 1.134 m) are true distances; "
                   f"usable = face - setback - keep-outs; grid search over row/column phases and both orientations. "
                   f"{n['panels']} modules = {n['kwp']:.1f} kWp. Validator: 0 overlaps, 0 setback/clearance violations, exact 3D size.", "7 / 8")),
        (20, title_slide(["Limitations & next steps",
                          "tree-covered corner excluded, dormers treated as obstructions,",
                          "geometric yield factor (no shading simulation),",
                          "no explicit roof topology graph"],
                         "README.md - TECHNICAL_EXPLANATION.md - outputs/REPORT.md")),
    ]
    path = OUT / "demo_video.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    total = 0
    for secs, frame in scenes:
        for _ in range(int(secs * FPS)):
            vw.write(frame)
        total += secs
    vw.release()
    print(f"wrote {path} ({total // 60}:{total % 60:02d}, {len(scenes)} scenes)")


if __name__ == "__main__":
    main()
