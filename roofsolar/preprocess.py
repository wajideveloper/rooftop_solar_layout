"""Step 2 - regularise noisy elevations and fill missing data inside the footprint.

1. nodata / NaN pixels are recorded as 'gaps'.
2. spikes: isolated pixels far from the local median (|z - med| > threshold) whose blob is too small
   to be a real object are treated as gaps too (chimneys are big and stay).
3. gaps are filled by natural-neighbour-like interpolation: scipy LinearNDInterpolator over the valid
   pixels, with a nearest-neighbour fallback at the footprint boundary.
4. salt-and-pepper noise: 3x3 median.
5. edge-preserving smoothing: bilateral filter (sigma_color in metres) so ridges/hips and step edges
   stay sharp while planar faces become smooth. Final 'regularisation' happens in step 4/6 where every
   face is replaced by its fitted plane.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from .config import PreprocessParams


def fill_gaps(z: np.ndarray, gaps: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """Interpolate z at `gaps` from valid pixels inside `domain` (linear, nearest fallback)."""
    out = z.copy()
    if not gaps.any():
        return out
    valid = domain & ~gaps
    rr, cc = np.nonzero(valid)
    gr, gc = np.nonzero(gaps)
    pts = np.column_stack([rr, cc]).astype(float)
    vals = z[rr, cc]
    lin = LinearNDInterpolator(pts, vals)
    filled = lin(np.column_stack([gr, gc]).astype(float))
    bad = ~np.isfinite(filled)
    if bad.any():
        nn = NearestNDInterpolator(pts, vals)
        filled[bad] = nn(np.column_stack([gr[bad], gc[bad]]).astype(float))
    out[gr, gc] = filled
    return out


def preprocess_dsm(dsm_raw: np.ndarray, footprint: np.ndarray, res: float, p: PreprocessParams,
                   nodata: float | None = None) -> dict:
    z = np.array(dsm_raw, dtype=np.float32)
    gaps = ~np.isfinite(z)
    if nodata is not None:
        gaps |= z == nodata
    if np.ma.isMaskedArray(dsm_raw):
        gaps |= np.ma.getmaskarray(dsm_raw)
    z[gaps] = np.nan

    # ---- spike detection (isolated outliers), evaluated inside the footprint only
    zf = np.where(gaps, np.nanmedian(z[footprint]), z)
    med = ndimage.median_filter(zf, size=p.spike_window)
    dev = np.abs(zf - med)
    cand = (dev > p.spike_threshold) & footprint & ~gaps
    lab, n = ndimage.label(cand)
    sizes = np.bincount(lab.ravel()) * res * res
    spikes = cand & (sizes[lab] < p.spike_max_area)

    all_gaps = (gaps | spikes) & footprint
    filled = fill_gaps(np.where(gaps, med, z), all_gaps, footprint)
    filled = np.where(np.isfinite(filled), filled, med).astype(np.float32)

    # ---- noise suppression
    med3 = ndimage.median_filter(filled, size=p.median_size)
    smooth = cv2.bilateralFilter(med3, d=p.bilateral_d, sigmaColor=p.bilateral_sigma_color,
                                 sigmaSpace=p.bilateral_sigma_space)
    noise = float(np.std((filled - smooth)[footprint]))
    return {
        "dsm_filled": filled,       # gaps + spikes replaced, otherwise raw (used for obstruction residuals)
        "dsm_smooth": smooth,       # edge-preserving smoothed (used for normals / segmentation)
        "gap_mask": gaps & footprint,
        "spike_mask": spikes,
        "n_gaps": int((gaps & footprint).sum()),
        "n_spikes": int(spikes.sum()),
        "noise_std": noise,
    }
