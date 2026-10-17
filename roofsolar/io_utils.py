"""Raster/vector I/O helpers built on rasterio + shapely (no GDAL CLI needed)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes, rasterize
from rasterio.transform import Affine
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union


@dataclass
class Raster:
    data: np.ndarray          # 2D float32 (or 3D for RGB)
    transform: Affine
    crs: object
    nodata: float | None = None

    @property
    def res(self) -> float:
        return float(abs(self.transform.a))

    @property
    def shape(self):
        return self.data.shape[-2:]

    def xy_grid(self):
        """Map coordinates (pixel centres) for every pixel."""
        h, w = self.shape
        cols = np.arange(w) + 0.5
        rows = np.arange(h) + 0.5
        x = self.transform.c + cols * self.transform.a
        y = self.transform.f + rows * self.transform.e
        return np.meshgrid(x, y)

    def like(self, data: np.ndarray, nodata=None) -> "Raster":
        return Raster(data, self.transform, self.crs, nodata)


def read_raster(path: Path, masked=True) -> Raster:
    with rasterio.open(path) as src:
        data = src.read(masked=masked)
        data = data[0] if data.shape[0] == 1 else data
        return Raster(data, src.transform, src.crs, src.nodata)


def write_raster(path: Path, r: Raster, data: np.ndarray | None = None, nodata=-9999.0, dtype="float32"):
    data = r.data if data is None else data
    if np.ma.isMaskedArray(data):
        data = data.filled(nodata)
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype=dtype, crs=r.crs, transform=r.transform,
                       nodata=nodata, compress="deflate") as dst:
        dst.write(data.astype(dtype))


def mask_to_polygons(mask: np.ndarray, transform: Affine, min_area=0.0) -> list[Polygon]:
    polys = []
    for geom, val in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform, connectivity=4):
        if val != 1:
            continue
        p = shape(geom)
        if p.area >= min_area:
            polys.append(p)
    return polys


def labels_to_polygons(labels: np.ndarray, transform: Affine) -> dict[int, Polygon | MultiPolygon]:
    """Vectorise an integer label raster -> {label: polygon}, 0 = background."""
    out: dict[int, list] = {}
    for geom, val in shapes(labels.astype(np.int32), mask=labels > 0, transform=transform, connectivity=4):
        out.setdefault(int(val), []).append(shape(geom))
    return {k: unary_union(v) for k, v in out.items()}


def polygons_to_mask(geoms, shape_hw, transform: Affine) -> np.ndarray:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape_hw, bool)
    return rasterize([(g, 1) for g in geoms], out_shape=shape_hw, transform=transform,
                     fill=0, dtype=np.uint8).astype(bool)


def write_geojson(path: Path, features: list[dict], crs_epsg: int | None = None):
    """features: list of {"geometry": shapely geom, "properties": {...}}"""
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(f["geometry"]), "properties": _jsonable(f.get("properties", {}))}
        for f in features]}
    if crs_epsg:
        fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs_epsg}"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, indent=1))


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2,
                               default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
