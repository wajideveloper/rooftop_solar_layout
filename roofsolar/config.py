"""All tunable parameters of the pipeline in one place (units: metres unless noted)."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paths:
    dsm_full: Path = Path("dsm_full.tif")
    dsm_clipped: Path = Path("dsm_clipped.tif")   # DSM clipped to the parcel; its nodata = outside parcel
    rgb: Path = Path("rgb.tif")
    out_dir: Path = Path("outputs")


@dataclass
class FootprintParams:
    min_height_above_ground: float = 2.5   # nDSM threshold for "elevated structure"
    tree_height: float = 9.5               # anything taller than this is treated as vegetation
    tree_dilate: float = 1.0               # grow the tree mask by this distance
    roughness_window: int = 7              # px; window for local roughness (std of DSM minus median)
    roughness_tree: float = 0.9            # std [m] above which a region is "rough" = canopy
    green_index_tree: float = 0.10         # ExG above this = vegetation
    open_radius: float = 0.5               # morphological opening radius (removes thin links)
    close_radius: float = 0.6              # morphological closing radius (fills notches)
    simplify_tol: float = 0.25             # Douglas-Peucker tolerance for the footprint polygon
    rectilinear_snap: bool = True          # snap footprint edges to the two dominant orientations


@dataclass
class PreprocessParams:
    spike_window: int = 5                  # px; median window for spike detection
    spike_threshold: float = 1.5           # |z - median| > this AND blob < spike_max_area => spike
    spike_max_area: float = 0.4            # m^2; larger protrusions are real objects (chimneys)
    median_size: int = 3                   # px; salt/pepper noise removal
    bilateral_sigma_color: float = 0.25    # m; edge-preserving smoothing strength (keeps ridges sharp)
    bilateral_sigma_space: float = 3.0     # px
    bilateral_d: int = 7                   # px


@dataclass
class SegmentationParams:
    normal_window: int = 7                 # px; window for local plane-fit normals
    angle_threshold_deg: float = 9.0       # region growing: max angle between pixel normal and region normal
    dist_threshold: float = 0.15           # region growing: max orthogonal distance to region plane
    min_region_area: float = 2.0           # m^2; smaller grown regions are released immediately
    min_face_area: float = 5.0             # m^2; smaller final faces are dissolved (dormer flanks, chimney ramps)
    merge_angle_deg: float = 6.0           # merge adjacent regions whose planes are nearly identical
    merge_dist: float = 0.10               # ... and whose planes agree in height within this
    merge_union_rmse: float = 0.05         # ... or whose union is still planar (RMSE below this)
    reassign_tol: float = 0.35             # unlabeled pixels are assigned to the neighbouring plane if residual < this
    max_face_slope_deg: float = 75.0       # steeper = wall, not roof
    ransac_iters: int = 200
    ransac_inlier: float = 0.08
    polygon_simplify: float = 0.15


@dataclass
class ObstructionParams:
    protrusion_threshold: float = 0.20     # m; DSM above fitted plane by more than this = obstruction
    min_area: float = 0.15                 # m^2; ignore tinier residual blobs (noise)
    chimney_height: float = 0.8            # protrusion above this => chimney, below => skylight/roof window/vent
    brightness_contrast: float = 40.0      # RGB brightness above the median of the surrounding face => glazing
    bright_min_area: float = 0.3           # m^2
    vegetation_overlap: float = 0.5        # fraction of an object inside the canopy mask => 'vegetation overhang'
    clearance: float = 0.40                # m; keep-out buffer around obstructions
    dilate: float = 0.15                   # m; grow raw obstruction masks slightly (edge pixels are blurred)


@dataclass
class PanelParams:
    length: float = 1.722                  # m; standard 60/120-cell module
    width: float = 1.134
    power_wp: float = 400.0
    edge_setback: float = 0.40             # m; distance from face edges (fire/wind codes)
    gap: float = 0.02                      # m; spacing between adjacent modules
    row_gap: float = 0.02
    min_slope_deg: float = 0.0
    max_slope_deg: float = 60.0
    south_half_angle: float = 112.5        # usable aspects: within +-this of south (180): E .. S .. W; N/NE/NW excluded
    latitude_deg: float = 52.3             # site latitude (UTM 30N, ~5794 km N) for the orientation-yield estimate
    flat_slope_deg: float = 5.0            # faces flatter than this are "flat": orientation = footprint axis
    offset_search: int = 12                # grid offsets tried per axis when packing
    min_face_area: float = 4.0             # m^2 of usable area needed to even try


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    footprint: FootprintParams = field(default_factory=FootprintParams)
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    segmentation: SegmentationParams = field(default_factory=SegmentationParams)
    obstruction: ObstructionParams = field(default_factory=ObstructionParams)
    panels: PanelParams = field(default_factory=PanelParams)
