"""Step 3 - elevation, slope and aspect (Horn 1981 finite differences on the smoothed DSM).

Conventions
  slope  : degrees from horizontal (0 = flat)
  aspect : compass azimuth of the *downslope* direction, degrees clockwise from north (0 = N, 90 = E,
           180 = S, 270 = W); -1 for flat pixels (slope < 1 deg)
  normal : unit vector (nx, ny, nz) in map axes (x east, y north, z up)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def horn_gradient(z: np.ndarray, res: float):
    """dz/dx (east) and dz/dy (north) using Horn's 3x3 kernels. Rows run north->south."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / (8 * res)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float) / (8 * res)   # +north = up in the array
    gx = ndimage.convolve(z, kx, mode="nearest")
    gy = ndimage.convolve(z, ky, mode="nearest")
    return gx, gy


def slope_aspect(z: np.ndarray, res: float, flat_deg: float = 1.0):
    gx, gy = horn_gradient(z, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (np.degrees(np.arctan2(-gx, -gy)) + 360.0) % 360.0   # downslope = -(gx, gy)
    aspect = np.where(slope < flat_deg, -1.0, aspect)
    return slope.astype(np.float32), aspect.astype(np.float32), gx, gy


def normals_from_gradient(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def plane_normal_to_slope_aspect(n: np.ndarray):
    """n = (a, b, c) unit normal, c > 0 -> (slope_deg, aspect_deg)."""
    a, b, c = n
    slope = np.degrees(np.arccos(np.clip(c, -1, 1)))
    aspect = (np.degrees(np.arctan2(a, b)) + 360.0) % 360.0   # downslope direction is +(a, b) horizontally
    return float(slope), (float(aspect) if slope >= 1.0 else -1.0)


def aspect_label(aspect: float) -> str:
    if aspect < 0:
        return "flat"
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int(((aspect + 22.5) % 360) // 45)]
