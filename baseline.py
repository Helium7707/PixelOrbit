import os
import re
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from kornia.feature import LoFTR


# =============================================================================
# SIH 2026 - PS 26166
# HYBRID LUNAR IMAGE CORRESPONDENCE
#
# Architecture:
#
#   OHRC + TMC
#       |
#       +----------------------+
#       |                      |
#       v                      v
#   appearance branch      crater/CNSFM branch
#       |                      |
#     LoFTR              crater neighbourhoods
#       |                      |
#     RANSAC              graph matching + RANSAC
#       |                      |
#       +----------+-----------+
#                  |
#           aggregate confidence
#
# Important:
# - CNSFM is NOT replacing LoFTR.
# - LoFTR and CNSFM are independent evidence sources.
# - The final confidence is a heuristic engineering score, NOT a calibrated
#   probability.
# - The current products are OHRC = 0.20 m/px and TMC-2 = 6.13 m/px.
# - OHRC is resampled to approximately TMC physical scale before both branches.
# - The old arbitrary 12,000-line central OHRC crop is NOT used.
# =============================================================================


ROOT = os.path.dirname(os.path.abspath(__file__))

OHRC_XML = f"{ROOT}/ch2_ohr_ncp_20231004T0406038822_d_img_d18.xml"
TMC_XML = f"{ROOT}/ch2_tmc_ncn_20250707T1853051045_d_img_d18.xml"

R_MOON = 1737400.0

# CURRENT DATASET - do not use the old 0.25 / 5.21 values.
CURRENT_GSD = {
    "OHRC": 0.20,
    "TMC-2": 6.13,
}

DTYPE_MAP = {
    "UnsignedByte": np.uint8,
    "SignedByte": np.int8,
    "UnsignedMSB2": ">u2",
    "SignedMSB2": ">i2",
    "UnsignedLSB2": "<u2",
    "SignedLSB2": "<i2",
    "IEEE754MSB4": ">f4",
    "IEEE754LSB4": "<f4",
}


# ----------------------------------------------------------------------------- 
# TUNABLE PARAMETERS
# -----------------------------------------------------------------------------

# Approximate physical crater radius range in metres after TMC-scale
# normalization. These are detector limits, not scientific crater catalog
# limits. Increase/decrease after inspecting cnsfm_debug.png.
MIN_CRATER_RADIUS_M = 30.0
MAX_CRATER_RADIUS_M = 500.0

MAX_CRATERS = 120
K_NEIGHBORS = 6

# Hough detector.
HOUGH_DP = 1.2
HOUGH_PARAM1 = 70
HOUGH_PARAM2 = 16
HOUGH_MIN_DIST_M = 80.0

# Local descriptor matching.
DESCRIPTOR_RATIO = 0.82
DESCRIPTOR_ABS_DISTANCE = 1.55

# Geometric verification.
RANSAC_REPROJ_THRESHOLD = 6.0
RANSAC_MAX_ITERS = 5000

# LoFTR.
LOFTR_PRETRAINED = "outdoor"
LOFTR_CONFIDENCE_MIN = 0.20

# Aggregate confidence weights.
# CNSFM gets meaningful weight but does not dominate LoFTR.
LOFTR_WEIGHT = 0.60
CNSFM_WEIGHT = 0.40

# Minimum dimensions for LoFTR.
MAX_LOFTR_SIDE = 1600

# Strong OHRC -> TMC visual degradation. The OHRC is already reduced to
# physical TMC scale first; these settings deliberately make it look much
# more like the coarse TMC product before matching.
OHRC_PIXEL_BLOCK = 3          # 3x3 block averaging / pixelation
OHRC_GRAY_LEVELS = 32         # quantize to coarse intensity levels
OHRC_DARKEN_FACTOR = 0.62     # lower apparent brightness
OHRC_GAMMA = 1.12              # slightly suppress bright pixels
OHRC_BLUR_SIGMA = 1.0         # sensor-like smoothing before pixelation

# TMC preprocessing is intentionally softer than the old CLAHE path so the
# OHRC degradation is not immediately undone by aggressive contrast stretch.
TMC_GAMMA = 1.05

# TMC crop padding in native TMC pixels.
TMC_PAD_PX = 20

# Save outputs in the working directory.
RESULT_IMAGE = os.path.join(ROOT, "hybrid_correspondence_result.png")
DEBUG_IMAGE = os.path.join(ROOT, "cnsfm_debug.png")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Crater:
    x: float
    y: float
    r: float
    score: float


# =============================================================================
# 1. PDS4 METADATA / DISK I/O
# =============================================================================

def parse_pds4_metadata(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def strip_ns(tag):
        return tag.split("}")[-1] if "}" in tag else tag

    meta = {}

    for elem in root.iter():
        tag = strip_ns(elem.tag)

        if tag == "Axis_Array":
            axis_name = ""
            elements = None

            for child in elem:
                ctag = strip_ns(child.tag)

                if ctag == "axis_name" and child.text:
                    axis_name = child.text.strip()

                elif ctag == "elements" and child.text:
                    try:
                        elements = int(child.text.strip())
                    except ValueError:
                        pass

            if elements is not None:
                if axis_name.lower() == "line":
                    meta["lines"] = elements
                elif axis_name.lower() in ("sample", "samples"):
                    meta["samples"] = elements

        elif tag == "Element_Array":
            for child in elem:
                if strip_ns(child.tag) == "data_type" and child.text:
                    raw = child.text.strip()
                    if raw in DTYPE_MAP:
                        meta["dtype"] = DTYPE_MAP[raw]

    # Find geographic corner fields robustly.
    wanted = {
        "upper_left_latitude": "UL_LAT",
        "upper_left_longitude": "UL_LON",
        "upper_right_latitude": "UR_LAT",
        "upper_right_longitude": "UR_LON",
        "lower_right_latitude": "LR_LAT",
        "lower_right_longitude": "LR_LON",
        "lower_left_latitude": "LL_LAT",
        "lower_left_longitude": "LL_LON",
    }

    values = {}

    for elem in root.iter():
        tag = strip_ns(elem.tag).lower()

        if tag in wanted and elem.text:
            try:
                values[wanted[tag]] = float(elem.text.strip())
            except ValueError:
                pass

    # Some labels may contain slightly different naming. Fall back to
    # searching serialized XML if needed.
    if len(values) < 8:
        raw_text = ET.tostring(root, encoding="utf-8").decode("utf-8")

        patterns = {
            "UL_LAT": r"upper_left_latitude[^>]*>([-+0-9.eE]+)",
            "UL_LON": r"upper_left_longitude[^>]*>([-+0-9.eE]+)",
            "UR_LAT": r"upper_right_latitude[^>]*>([-+0-9.eE]+)",
            "UR_LON": r"upper_right_longitude[^>]*>([-+0-9.eE]+)",
            "LR_LAT": r"lower_right_latitude[^>]*>([-+0-9.eE]+)",
            "LR_LON": r"lower_right_longitude[^>]*>([-+0-9.eE]+)",
            "LL_LAT": r"lower_left_latitude[^>]*>([-+0-9.eE]+)",
            "LL_LON": r"lower_left_longitude[^>]*>([-+0-9.eE]+)",
        }

        for key, pattern in patterns.items():
            if key not in values:
                m = re.search(pattern, raw_text, re.IGNORECASE)
                if m:
                    values[key] = float(m.group(1))

    required = [
        "UL_LAT", "UL_LON",
        "UR_LAT", "UR_LON",
        "LR_LAT", "LR_LON",
        "LL_LAT", "LL_LON",
    ]

    missing = [x for x in required if x not in values]
    if missing:
        raise ValueError(
            f"Could not locate geographic corners {missing} in {xml_path}"
        )

    meta["corners"] = {
        "UL": (values["UL_LAT"], values["UL_LON"]),
        "UR": (values["UR_LAT"], values["UR_LON"]),
        "LR": (values["LR_LAT"], values["LR_LON"]),
        "LL": (values["LL_LAT"], values["LL_LON"]),
    }

    path_lower = xml_path.lower()

    if "ohr" in path_lower:
        meta["gsd"] = CURRENT_GSD["OHRC"]
        meta["name"] = "OHRC"
    elif "tmc" in path_lower:
        meta["gsd"] = CURRENT_GSD["TMC-2"]
        meta["name"] = "TMC-2"
    else:
        raise ValueError("Could not identify OHRC/TMC from XML path.")

    img_path = os.path.splitext(xml_path)[0] + ".img"
    if not os.path.exists(img_path):
        img_path = os.path.splitext(xml_path)[0] + ".IMG"

    if not os.path.exists(img_path):
        img_path = None

    meta["img_path"] = img_path

    if "lines" not in meta or "samples" not in meta:
        raise ValueError(f"Could not parse image dimensions from {xml_path}")

    if "dtype" not in meta:
        raise ValueError(f"Could not parse image datatype from {xml_path}")

    return meta


def read_pds4_window(filepath, r_start, r_end, c_start, c_end,
                     total_samples, dtype):
    """
    Read a bounded rectangular raster window directly from disk.

    PDS4 calibrated products used here are simple line/sample rasters.
    """
    r_start = max(0, int(r_start))
    r_end = max(r_start, int(r_end))
    c_start = max(0, int(c_start))
    c_end = max(c_start, int(c_end))

    rows = r_end - r_start
    cols = c_end - c_start

    out = np.zeros((rows, cols), dtype=dtype)

    itemsize = np.dtype(dtype).itemsize
    row_stride = total_samples * itemsize
    bytes_per_row = cols * itemsize

    with open(filepath, "rb") as f:
        for i, row in enumerate(range(r_start, r_end)):
            offset = row * row_stride + c_start * itemsize
            f.seek(offset)
            raw = f.read(bytes_per_row)

            if len(raw) == bytes_per_row:
                out[i] = np.frombuffer(raw, dtype=dtype)

    return out


def read_decimated_window(filepath, r_start, r_end, c_start, c_end,
                          total_samples, dtype, step):
    """
    Read every 'step'-th source row.

    This avoids loading the full OHRC strip into RAM. It still performs disk
    reads over the requested rows, but the returned array is much smaller.
    """
    r_start = max(0, int(r_start))
    r_end = max(r_start, int(r_end))
    c_start = max(0, int(c_start))
    c_end = max(c_start, int(c_end))
    step = max(1, int(step))

    rows = (max(0, r_end - r_start - 1) // step) + 1
    cols = c_end - c_start

    out = np.zeros((rows, cols), dtype=dtype)

    itemsize = np.dtype(dtype).itemsize
    row_stride = total_samples * itemsize
    bytes_per_row = cols * itemsize

    with open(filepath, "rb") as f:
        out_i = 0

        for row in range(r_start, r_end, step):
            offset = row * row_stride + c_start * itemsize
            f.seek(offset)
            raw = f.read(bytes_per_row)

            if len(raw) == bytes_per_row:
                out[out_i] = np.frombuffer(raw, dtype=dtype)
            out_i += 1

    return out


# =============================================================================
# 2. GEOMETRY
# =============================================================================

def to_local_metric(lat, lon, lat0, lon0):
    """
    Local equirectangular metric approximation.

    The current pair is equatorial, so this is suitable for the prototype.
    It is not the final camera/terrain model.
    """
    phi = math.radians(lat)
    lam = math.radians(lon)
    phi0 = math.radians(lat0)
    lam0 = math.radians(lon0)

    x = R_MOON * (lam - lam0) * math.cos(phi0)
    y = R_MOON * (phi - phi0)

    return x, y


def compute_ohrc_footprint_in_tmc(ohrc_meta, tmc_meta):
    """
    Approximate mapping:
        geographic metric coordinates -> TMC pixels

    Uses all four corners rather than only the centroid.
    """
    order = ["UL", "UR", "LR", "LL"]

    ohrc_lats = [ohrc_meta["corners"][k][0] for k in order]
    ohrc_lons = [ohrc_meta["corners"][k][1] for k in order]

    lat0 = float(np.mean(ohrc_lats))
    lon0 = float(np.mean(ohrc_lons))

    tmc_metric = np.array(
        [
            to_local_metric(
                tmc_meta["corners"][k][0],
                tmc_meta["corners"][k][1],
                lat0,
                lon0,
            )
            for k in order
        ],
        dtype=np.float32,
    )

    tmc_pixels = np.array(
        [
            [0, 0],
            [tmc_meta["samples"] - 1, 0],
            [tmc_meta["samples"] - 1, tmc_meta["lines"] - 1],
            [0, tmc_meta["lines"] - 1],
        ],
        dtype=np.float32,
    )

    H_metric_to_tmc = cv2.getPerspectiveTransform(
        tmc_metric, tmc_pixels
    )

    ohrc_metric = np.array(
        [
            to_local_metric(
                ohrc_meta["corners"][k][0],
                ohrc_meta["corners"][k][1],
                lat0,
                lon0,
            )
            for k in order
        ],
        dtype=np.float32,
    )

    ohrc_in_tmc = cv2.perspectiveTransform(
        ohrc_metric.reshape(-1, 1, 2),
        H_metric_to_tmc,
    ).reshape(-1, 2)

    pad = TMC_PAD_PX

    c_min = max(
        0,
        int(np.floor(np.min(ohrc_in_tmc[:, 0]))) - pad
    )
    c_max = min(
        tmc_meta["samples"],
        int(np.ceil(np.max(ohrc_in_tmc[:, 0]))) + pad
    )
    r_min = max(
        0,
        int(np.floor(np.min(ohrc_in_tmc[:, 1]))) - pad
    )
    r_max = min(
        tmc_meta["lines"],
        int(np.ceil(np.max(ohrc_in_tmc[:, 1]))) + pad
    )

    if r_max <= r_min or c_max <= c_min:
        raise RuntimeError("Projected OHRC footprint produced an invalid TMC crop.")

    return {
        "ohrc_polygon_tmc_px": ohrc_in_tmc,
        "tmc_bbox": (r_min, r_max, c_min, c_max),
        "proj_origin": (lat0, lon0),
        "H_metric_to_tmc": H_metric_to_tmc,
    }


def map_tmc_crop_to_ohrc(ohrc_meta, tmc_meta, geom):
    """
    Find the OHRC pixel rectangle corresponding to the TMC crop.

    This is used to extract the same physical region from OHRC.
    """
    r_min, r_max, c_min, c_max = geom["tmc_bbox"]

    order = ["UL", "UR", "LR", "LL"]

    lat0, lon0 = geom["proj_origin"]

    tmc_metric = np.array(
        [
            to_local_metric(
                tmc_meta["corners"][k][0],
                tmc_meta["corners"][k][1],
                lat0,
                lon0,
            )
            for k in order
        ],
        dtype=np.float32,
    )

    tmc_pixels = np.array(
        [
            [0, 0],
            [tmc_meta["samples"] - 1, 0],
            [tmc_meta["samples"] - 1, tmc_meta["lines"] - 1],
            [0, tmc_meta["lines"] - 1],
        ],
        dtype=np.float32,
    )

    H_tmc_to_metric = cv2.getPerspectiveTransform(
        tmc_pixels, tmc_metric
    )

    tmc_crop_corners = np.array(
        [
            [c_min, r_min],
            [c_max - 1, r_min],
            [c_max - 1, r_max - 1],
            [c_min, r_max - 1],
        ],
        dtype=np.float32,
    )

    metric_points = cv2.perspectiveTransform(
        tmc_crop_corners.reshape(-1, 1, 2),
        H_tmc_to_metric,
    ).reshape(-1, 2)

    ohrc_metric = np.array(
        [
            to_local_metric(
                ohrc_meta["corners"][k][0],
                ohrc_meta["corners"][k][1],
                lat0,
                lon0,
            )
            for k in order
        ],
        dtype=np.float32,
    )

    ohrc_pixels = np.array(
        [
            [0, 0],
            [ohrc_meta["samples"] - 1, 0],
            [ohrc_meta["samples"] - 1, ohrc_meta["lines"] - 1],
            [0, ohrc_meta["lines"] - 1],
        ],
        dtype=np.float32,
    )

    H_metric_to_ohrc = cv2.getPerspectiveTransform(
        ohrc_metric, ohrc_pixels
    )

    ohrc_points = cv2.perspectiveTransform(
        metric_points.reshape(-1, 1, 2),
        H_metric_to_ohrc,
    ).reshape(-1, 2)

    x_min = max(0, int(np.floor(np.min(ohrc_points[:, 0]))))
    x_max = min(
        ohrc_meta["samples"],
        int(np.ceil(np.max(ohrc_points[:, 0]))) + 1,
    )

    y_min = max(0, int(np.floor(np.min(ohrc_points[:, 1]))))
    y_max = min(
        ohrc_meta["lines"],
        int(np.ceil(np.max(ohrc_points[:, 1]))) + 1,
    )

    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError("Could not map TMC crop back into OHRC.")

    return {
        "ohrc_bbox": (y_min, y_max, x_min, x_max),
        "ohrc_polygon": ohrc_points,
    }


# =============================================================================
# 3. IMAGE PREPROCESSING
# =============================================================================

def normalize_percentile_clahe(img):
    img_f = img.astype(np.float32)

    p1, p99 = np.percentile(img_f, (1.0, 99.0))

    if p99 <= p1:
        out = np.zeros_like(img_f, dtype=np.uint8)
    else:
        out = np.clip(
            (img_f - p1) / (p99 - p1) * 255.0,
            0,
            255,
        ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(out)


def histogram_match(source, reference):
    """Match the grayscale distribution of source to reference."""
    src = source.astype(np.uint8).ravel()
    ref = reference.astype(np.uint8).ravel()

    src_values, src_idx, src_counts = np.unique(
        src, return_inverse=True, return_counts=True
    )
    ref_values, ref_counts = np.unique(
        ref, return_counts=True
    )

    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    mapped = np.interp(src_cdf, ref_cdf, ref_values)
    out = mapped[src_idx].reshape(source.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def pixelate_and_tone_down_ohrc(ohrc_img, tmc_reference):
    """
    Aggressively make the physically resampled OHRC resemble coarse TMC.

    Pipeline:
      1. match OHRC intensity distribution to TMC
      2. sensor-like blur
      3. 3x3 block averaging / pixelation
      4. coarse 32-level quantization
      5. darken + mild gamma compression
      6. restore image dimensions with nearest-neighbour blocks

    This is deliberate domain degradation, not a claim about the exact TMC
    instrument transfer function.
    """
    x = normalize_percentile_clahe(ohrc_img)
    ref = normalize_percentile_clahe(tmc_reference)

    # Replace aggressive CLAHE appearance with TMC's overall histogram.
    x = histogram_match(x, ref)

    if OHRC_BLUR_SIGMA > 0:
        x = cv2.GaussianBlur(x, (0, 0), OHRC_BLUR_SIGMA)

    block = max(1, int(OHRC_PIXEL_BLOCK))
    h, w = x.shape
    small_w = max(8, int(round(w / block)))
    small_h = max(8, int(round(h / block)))

    # Area reduction creates actual coarse pixels; nearest resize keeps them
    # visibly block-like instead of inventing fine detail again.
    small = cv2.resize(
        x,
        (small_w, small_h),
        interpolation=cv2.INTER_AREA,
    )

    levels = max(2, int(OHRC_GRAY_LEVELS))
    step = 255.0 / (levels - 1)
    small = np.round(small.astype(np.float32) / step) * step
    small = np.clip(small, 0, 255).astype(np.uint8)

    # Darken after quantization so the requested TMC-like low brightness is
    # not washed out by a later contrast stretch.
    y = small.astype(np.float32) / 255.0
    y = np.power(np.clip(y, 0.0, 1.0), OHRC_GAMMA)
    y *= OHRC_DARKEN_FACTOR
    small = np.clip(y * 255.0, 0, 255).astype(np.uint8)

    return cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )


def prepare_tmc_reference(tmc_img):
    """Gentle TMC normalization without aggressive CLAHE."""
    x = tmc_img.astype(np.float32)
    p1, p99 = np.percentile(x, (1.0, 99.0))
    if p99 <= p1:
        y = np.zeros_like(x, dtype=np.uint8)
    else:
        y = np.clip((x - p1) / (p99 - p1) * 255.0, 0, 255).astype(np.uint8)

    y = cv2.GaussianBlur(y, (0, 0), 0.7)
    yf = np.power(y.astype(np.float32) / 255.0, TMC_GAMMA)
    return np.clip(yf * 255.0, 0, 255).astype(np.uint8)


def gaussian_downsample(img, scale):
    """
    Downsample OHRC to TMC physical resolution.

    scale = OHRC_GSD / TMC_GSD ~= 0.0326.
    """
    scale = float(scale)

    if scale >= 1.0:
        return cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    # Gaussian prefilter before severe reduction to suppress aliasing.
    sigma = max(0.8, 0.5 / scale)

    blurred = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    new_w = max(32, int(round(img.shape[1] * scale)))
    new_h = max(32, int(round(img.shape[0] * scale)))

    return cv2.resize(
        blurred,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def resize_pair_for_loftr(img0, img1):
    """
    Keep both images at the same physical scale and limit dimensions.

    TMC is the reference coordinate system.
    """
    h0, w0 = img0.shape
    h1, w1 = img1.shape

    scale0 = min(
        1.0,
        MAX_LOFTR_SIDE / max(h0, w0),
    )
    scale1 = min(
        1.0,
        MAX_LOFTR_SIDE / max(h1, w1),
    )

    scale = min(scale0, scale1)

    if scale < 1.0:
        new_w0 = max(32, int(round(w0 * scale)))
        new_h0 = max(32, int(round(h0 * scale)))
        new_w1 = max(32, int(round(w1 * scale)))
        new_h1 = max(32, int(round(h1 * scale)))

        img0 = cv2.resize(
            img0, (new_w0, new_h0), interpolation=cv2.INTER_AREA
        )
        img1 = cv2.resize(
            img1, (new_w1, new_h1), interpolation=cv2.INTER_AREA
        )

    return img0, img1, scale


def pad_to_div8(img):
    h, w = img.shape

    nh = ((h + 7) // 8) * 8
    nw = ((w + 7) // 8) * 8

    padded = np.zeros((nh, nw), dtype=np.uint8)
    padded[:h, :w] = img

    return padded, (h, w)


# =============================================================================
# 4. LoFTR BRANCH
# =============================================================================

def run_loftr_branch(img0, img1):
    print("\n[LoFTR] Starting appearance branch...")

    img0, img1, resize_scale = resize_pair_for_loftr(img0, img1)

    img0_pad, shape0 = pad_to_div8(img0)
    img1_pad, shape1 = pad_to_div8(img1)

    h0, w0 = shape0
    h1, w1 = shape1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LoFTR] Device: {device}")

    matcher = LoFTR(pretrained=LOFTR_PRETRAINED).to(device).eval()

    t0 = (
        torch.from_numpy(img0_pad)
        .float()[None, None]
        .to(device)
        / 255.0
    )

    t1 = (
        torch.from_numpy(img1_pad)
        .float()[None, None]
        .to(device)
        / 255.0
    )

    with torch.no_grad():
        output = matcher({
            "image0": t0,
            "image1": t1,
        })

    kpts0 = output["keypoints0"].cpu().numpy()
    kpts1 = output["keypoints1"].cpu().numpy()
    conf = output["confidence"].cpu().numpy()

    valid = (
        (kpts0[:, 0] >= 0)
        & (kpts0[:, 0] < w0)
        & (kpts0[:, 1] >= 0)
        & (kpts0[:, 1] < h0)
        & (kpts1[:, 0] >= 0)
        & (kpts1[:, 0] < w1)
        & (kpts1[:, 1] >= 0)
        & (kpts1[:, 1] < h1)
    )

    kpts0 = kpts0[valid]
    kpts1 = kpts1[valid]
    conf = conf[valid]

    # Keep only reasonably confident raw LoFTR matches.
    conf_mask = conf >= LOFTR_CONFIDENCE_MIN

    filtered_kpts0 = kpts0[conf_mask]
    filtered_kpts1 = kpts1[conf_mask]
    filtered_conf = conf[conf_mask]

    if len(filtered_kpts0) >= 8:
        pts0 = filtered_kpts0
        pts1 = filtered_kpts1

        H, mask = cv2.findHomography(
            pts0,
            pts1,
            cv2.USAC_MAGSAC,
            RANSAC_REPROJ_THRESHOLD,
            maxIters=RANSAC_MAX_ITERS,
        )

        if mask is None:
            mask = np.zeros(len(pts0), dtype=np.uint8)
        else:
            mask = mask.ravel().astype(bool)

    else:
        H = None
        mask = np.zeros(len(filtered_kpts0), dtype=bool)

    inlier_count = int(np.sum(mask))
    raw_count = len(kpts0)
    filtered_count = len(filtered_kpts0)

    inlier_ratio = (
        inlier_count / filtered_count
        if filtered_count > 0
        else 0.0
    )

    if inlier_count >= 4 and H is not None:
        in0 = filtered_kpts0[mask]
        in1 = filtered_kpts1[mask]

        ones = np.ones((len(in0), 1), dtype=np.float32)
        hom = np.hstack([in0, ones])

        projected = (H @ hom.T).T

        valid_z = np.abs(projected[:, 2]) > 1e-8
        projected = projected[valid_z]
        actual = in1[valid_z]

        projected = projected[:, :2] / projected[:, 2:3]

        errors = np.linalg.norm(
            projected - actual,
            axis=1,
        )

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        median_error = float(np.median(errors))

    else:
        rmse = float("inf")
        median_error = float("inf")

    mean_conf = (
        float(np.mean(filtered_conf))
        if len(filtered_conf) > 0
        else 0.0
    )

    # Heuristic LoFTR quality score.
    #
    # Components:
    # - confidence
    # - inlier ratio
    # - absolute inlier support
    # - low geometric error
    #
    # This is deliberately bounded to [0,1].
    confidence_component = np.clip(mean_conf, 0.0, 1.0)

    ratio_component = np.clip(
        inlier_ratio / 0.75,
        0.0,
        1.0,
    )

    support_component = np.clip(
        inlier_count / 50.0,
        0.0,
        1.0,
    )

    if np.isfinite(rmse):
        error_component = math.exp(-rmse / 8.0)
    else:
        error_component = 0.0

    loftr_score = (
        0.30 * confidence_component
        + 0.30 * ratio_component
        + 0.25 * support_component
        + 0.15 * error_component
    )

    print(
        f"[LoFTR] Raw matches       : {raw_count}"
    )
    print(
        f"[LoFTR] Usable matches    : {filtered_count}"
    )
    print(
        f"[LoFTR] Inliers            : {inlier_count}"
    )
    print(
        f"[LoFTR] Inlier ratio       : {inlier_ratio:.4f}"
    )
    print(
        f"[LoFTR] Mean confidence    : {mean_conf:.4f}"
    )
    print(
        f"[LoFTR] RMSE               : "
        f"{rmse:.3f}" if np.isfinite(rmse)
        else "[LoFTR] RMSE               : inf"
    )
    print(
        f"[LoFTR] Branch score       : {loftr_score:.4f}"
    )

    # Convert points back to the pre-LoFTR resized image coordinate system.
    # Since both images received the same final scale, one scale is sufficient.
    if resize_scale > 0:
        filtered_kpts0_original = filtered_kpts0 / resize_scale
        filtered_kpts1_original = filtered_kpts1 / resize_scale
    else:
        filtered_kpts0_original = filtered_kpts0
        filtered_kpts1_original = filtered_kpts1

    return {
        "raw_matches": raw_count,
        "matches": filtered_count,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "mean_confidence": mean_conf,
        "rmse": rmse,
        "median_error": median_error,
        "score": float(np.clip(loftr_score, 0.0, 1.0)),
        "points0": filtered_kpts0_original,
        "points1": filtered_kpts1_original,
        "mask": mask,
        "H": H,
        "display_img0": img0,
        "display_img1": img1,
    }


# =============================================================================
# 5. CNSFM-STYLE CRATER DETECTION
# =============================================================================

def detect_craters(img):
    """
    Hough-based crater candidate detector.

    This is an engineering approximation of the crater-neighbourhood idea.
    It is not a claim of exact CNSFM paper reproduction.
    """
    # The input has already been domain-degraded. Use only a gentle normalize
    # here; aggressive CLAHE can destroy the deliberately coarse appearance.
    norm = img.astype(np.uint8)
    smooth = cv2.GaussianBlur(norm, (5, 5), 1.2)

    h, w = smooth.shape

    min_dim = min(h, w)

    min_r = max(
        2,
        int(round(MIN_CRATER_RADIUS_M / CURRENT_GSD["TMC-2"]))
    )
    max_r = max(
        min_r + 2,
        int(round(MAX_CRATER_RADIUS_M / CURRENT_GSD["TMC-2"]))
    )

    max_r = min(max_r, max(8, int(min_dim * 0.35)))

    min_dist = max(
        5,
        int(round(HOUGH_MIN_DIST_M / CURRENT_GSD["TMC-2"]))
    )

    circles = cv2.HoughCircles(
        smooth,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=min_dist,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(np.float32)

    candidates = []

    # Local ring-support score.
    for x, y, r in circles:
        x = float(x)
        y = float(y)
        r = float(r)

        if (
            x - r < 1
            or y - r < 1
            or x + r >= w - 1
            or y + r >= h - 1
        ):
            continue

        # Fast annulus sampling instead of allocating a full HxW mask for
        # every Hough circle. This makes the CNSFM branch practical on the
        # multi-thousand-pixel lunar crops.
        angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        rr = r * 1.00
        xs = np.rint(x + rr * np.cos(angles)).astype(np.int32)
        ys = np.rint(y + rr * np.sin(angles)).astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        ring_values = smooth[ys[valid], xs[valid]]

        if len(ring_values) < 10:
            continue

        ring_contrast = float(np.std(ring_values)) / 64.0

        # Sample a small interior grid rather than slicing a large patch.
        iy0 = max(0, int(y - 0.5 * r))
        iy1 = min(h, int(y + 0.5 * r))
        ix0 = max(0, int(x - 0.5 * r))
        ix1 = min(w, int(x + 0.5 * r))

        if iy1 <= iy0 or ix1 <= ix0:
            continue

        gy = np.linspace(iy0, iy1 - 1, 8).astype(np.int32)
        gx = np.linspace(ix0, ix1 - 1, 8).astype(np.int32)
        interior = smooth[np.ix_(gy, gx)]
        interior_std = float(np.std(interior)) / 64.0

        score = (
            0.65 * np.clip(ring_contrast, 0.0, 1.0)
            + 0.35 * np.clip(interior_std, 0.0, 1.0)
        )

        candidates.append(
            Crater(
                x=x,
                y=y,
                r=r,
                score=float(score),
            )
        )

    # Sort by detector support and perform simple NMS.
    candidates.sort(
        key=lambda c: c.score,
        reverse=True,
    )

    selected = []

    for c in candidates:
        keep = True

        for s in selected:
            distance = math.hypot(
                c.x - s.x,
                c.y - s.y,
            )

            min_allowed = 0.45 * max(c.r, s.r)

            if distance < min_allowed:
                keep = False
                break

        if keep:
            selected.append(c)

        if len(selected) >= MAX_CRATERS:
            break

    return selected


# =============================================================================
# 6. CNSFM-STYLE LOCAL CRATER NEIGHBOURHOOD DESCRIPTOR
# =============================================================================

def pairwise_distance_matrix(points):
    n = len(points)

    if n == 0:
        return np.empty((0, 0), dtype=np.float32)

    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def local_crater_descriptor(craters, index, k=K_NEIGHBORS):
    """
    Build a local geometric descriptor around one crater.

    Descriptor components:
      1. normalized neighbour distances
      2. neighbour radius ratios
      3. pairwise neighbour distances
      4. circular angular-gap pattern

    Absolute image orientation is not used directly.
    """
    center = craters[index]

    if len(craters) <= 1:
        return None

    points = np.array(
        [[c.x, c.y] for c in craters],
        dtype=np.float32,
    )

    radii = np.array(
        [c.r for c in craters],
        dtype=np.float32,
    )

    center_point = np.array(
        [center.x, center.y],
        dtype=np.float32,
    )

    distances = np.linalg.norm(
        points - center_point[None, :],
        axis=1,
    )

    order = np.argsort(distances)

    order = [
        j for j in order
        if j != index
    ][:k]

    if len(order) < 3:
        return None

    neighbor_points = points[order]
    neighbor_radii = radii[order]

    d = distances[order]

    local_scale = max(
        float(np.median(d)),
        1e-6,
    )

    norm_d = d / local_scale

    radius_ratio = neighbor_radii / max(center.r, 1e-6)

    # Pairwise distances between neighbours, normalized by the same local scale.
    pd = pairwise_distance_matrix(neighbor_points)

    tri = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            tri.append(pd[i, j] / local_scale)

    tri = np.asarray(tri, dtype=np.float32)

    # Angles of neighbours around the center.
    angles = np.arctan2(
        neighbor_points[:, 1] - center.y,
        neighbor_points[:, 0] - center.x,
    )

    angles = np.sort(
        (angles + 2.0 * np.pi) % (2.0 * np.pi)
    )

    gaps = np.diff(
        np.concatenate(
            [angles, angles[:1] + 2.0 * np.pi]
        )
    )

    gaps = gaps / (2.0 * np.pi)

    descriptor = np.concatenate(
        [
            np.sort(norm_d),
            np.sort(radius_ratio),
            np.sort(tri),
            np.sort(gaps),
        ]
    ).astype(np.float32)

    return descriptor


def build_crater_descriptors(craters):
    descriptors = []

    for i in range(len(craters)):
        d = local_crater_descriptor(
            craters,
            i,
            K_NEIGHBORS,
        )

        if d is not None:
            descriptors.append(
                (i, d)
            )

    return descriptors


def pad_descriptor(a, length):
    out = np.zeros(length, dtype=np.float32)
    n = min(length, len(a))
    out[:n] = a[:n]
    return out


def descriptor_distance(a, b):
    length = max(len(a), len(b))

    aa = pad_descriptor(a, length)
    bb = pad_descriptor(b, length)

    # Robust L1-like normalized distance.
    return float(
        np.mean(
            np.abs(aa - bb)
            / (1.0 + np.abs(aa) + np.abs(bb))
        )
    )


def match_crater_graphs(craters0, craters1):
    desc0 = build_crater_descriptors(craters0)
    desc1 = build_crater_descriptors(craters1)

    if not desc0 or not desc1:
        return [], 0.0

    candidates = []

    for i0, d0 in desc0:
        distances = []

        for i1, d1 in desc1:
            dist = descriptor_distance(d0, d1)
            distances.append(
                (dist, i1)
            )

        distances.sort(
            key=lambda x: x[0]
        )

        best_dist, best_i1 = distances[0]

        if len(distances) > 1:
            second_dist = distances[1][0]
        else:
            second_dist = best_dist + 1.0

        ratio = (
            best_dist / max(second_dist, 1e-6)
        )

        if (
            ratio <= DESCRIPTOR_RATIO
            and best_dist <= DESCRIPTOR_ABS_DISTANCE
        ):
            candidates.append(
                (
                    best_dist,
                    ratio,
                    i0,
                    best_i1,
                )
            )

    # Greedy one-to-one assignment.
    candidates.sort(
        key=lambda x: (x[0], x[1])
    )

    used0 = set()
    used1 = set()
    matches = []

    for dist, ratio, i0, i1 in candidates:
        if i0 in used0 or i1 in used1:
            continue

        used0.add(i0)
        used1.add(i1)

        matches.append(
            {
                "i0": i0,
                "i1": i1,
                "distance": dist,
                "ratio": ratio,
            }
        )

    if not matches:
        return [], 0.0

    mean_descriptor_distance = float(
        np.mean(
            [m["distance"] for m in matches]
        )
    )

    descriptor_score = math.exp(
        -mean_descriptor_distance
    )

    return matches, float(
        np.clip(descriptor_score, 0.0, 1.0)
    )


# =============================================================================
# 7. CNSFM GEOMETRIC VERIFICATION
# =============================================================================

def verify_crater_matches(craters0, craters1, matches):
    if len(matches) < 3:
        return {
            "matches": matches,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse": float("inf"),
            "score": 0.0,
            "mask": np.zeros(
                len(matches),
                dtype=bool,
            ),
            "M": None,
        }

    pts0 = np.array(
        [
            [craters0[m["i0"]].x, craters0[m["i0"]].y]
            for m in matches
        ],
        dtype=np.float32,
    )

    pts1 = np.array(
        [
            [craters1[m["i1"]].x, craters1[m["i1"]].y]
            for m in matches
        ],
        dtype=np.float32,
    )

    # Affine-partial is more permissive than a pure similarity transform
    # while still enforcing local geometric consistency.
    M, mask = cv2.estimateAffinePartial2D(
        pts0,
        pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        maxIters=RANSAC_MAX_ITERS,
        confidence=0.995,
        refineIters=20,
    )

    if M is None or mask is None:
        return {
            "matches": matches,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse": float("inf"),
            "score": 0.0,
            "mask": np.zeros(
                len(matches),
                dtype=bool,
            ),
            "M": None,
        }

    mask = mask.ravel().astype(bool)

    inlier_count = int(np.sum(mask))
    inlier_ratio = (
        inlier_count / len(matches)
        if matches
        else 0.0
    )

    if inlier_count >= 2:
        in0 = pts0[mask]
        in1 = pts1[mask]

        pred = cv2.transform(
            in0.reshape(-1, 1, 2),
            M,
        ).reshape(-1, 2)

        errors = np.linalg.norm(
            pred - in1,
            axis=1,
        )

        rmse = float(
            np.sqrt(np.mean(errors ** 2))
        )
    else:
        rmse = float("inf")

    support_component = np.clip(
        inlier_count / 12.0,
        0.0,
        1.0,
    )

    ratio_component = np.clip(
        inlier_ratio / 0.70,
        0.0,
        1.0,
    )

    if np.isfinite(rmse):
        error_component = math.exp(
            -rmse / 10.0
        )
    else:
        error_component = 0.0

    score = (
        0.40 * support_component
        + 0.40 * ratio_component
        + 0.20 * error_component
    )

    return {
        "matches": matches,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
        "score": float(np.clip(score, 0.0, 1.0)),
        "mask": mask,
        "M": M,
    }


def run_cnsfm_branch(img0, img1):
    print("\n[CNSFM] Starting crater-neighbourhood branch...")

    craters0 = detect_craters(img0)
    craters1 = detect_craters(img1)

    print(
        f"[CNSFM] Craters in image A: {len(craters0)}"
    )
    print(
        f"[CNSFM] Craters in image B: {len(craters1)}"
    )

    if len(craters0) < 3 or len(craters1) < 3:
        print(
            "[CNSFM] Too few crater candidates for reliable graph matching."
        )

        return {
            "craters0": craters0,
            "craters1": craters1,
            "matches": [],
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse": float("inf"),
            "score": 0.0,
            "mask": np.zeros(0, dtype=bool),
            "M": None,
        }

    graph_matches, descriptor_score = match_crater_graphs(
        craters0,
        craters1,
    )

    print(
        f"[CNSFM] Graph candidate matches: "
        f"{len(graph_matches)}"
    )
    print(
        f"[CNSFM] Descriptor score: "
        f"{descriptor_score:.4f}"
    )

    verified = verify_crater_matches(
        craters0,
        craters1,
        graph_matches,
    )

    # Combine descriptor similarity with geometric verification.
    final_score = (
        0.35 * descriptor_score
        + 0.65 * verified["score"]
    )

    verified["score"] = float(
        np.clip(final_score, 0.0, 1.0)
    )

    verified["craters0"] = craters0
    verified["craters1"] = craters1
    verified["descriptor_score"] = descriptor_score

    print(
        f"[CNSFM] Geometric inliers: "
        f"{verified['inliers']}"
    )
    print(
        f"[CNSFM] Inlier ratio: "
        f"{verified['inlier_ratio']:.4f}"
    )

    if np.isfinite(verified["rmse"]):
        print(
            f"[CNSFM] RMSE: "
            f"{verified['rmse']:.3f} px"
        )
    else:
        print("[CNSFM] RMSE: inf")

    print(
        f"[CNSFM] Branch score: "
        f"{verified['score']:.4f}"
    )

    return verified


# =============================================================================
# 8. CROSS-BRANCH AGGREGATION
# =============================================================================

def aggregate_confidence(loftr_result, cnsfm_result):
    """
    Aggregate two independent evidence sources.

    IMPORTANT:
    This is a heuristic confidence score, not a calibrated probability.
    """
    loftr_score = float(
        np.clip(loftr_result["score"], 0.0, 1.0)
    )

    cnsfm_score = float(
        np.clip(cnsfm_result["score"], 0.0, 1.0)
    )

    weighted = (
        LOFTR_WEIGHT * loftr_score
        + CNSFM_WEIGHT * cnsfm_score
    )

    # Agreement bonus:
    # If both branches independently support the same correspondence,
    # confidence increases slightly.
    agreement = 1.0 - abs(
        loftr_score - cnsfm_score
    )

    agreement_bonus = 0.10 * agreement

    aggregate = np.clip(
        weighted + agreement_bonus,
        0.0,
        1.0,
    )

    # Common-region decision is deliberately conservative.
    loftr_good = (
        loftr_result["inliers"] >= 12
        and loftr_result["inlier_ratio"] >= 0.20
    )

    cnsfm_good = (
        cnsfm_result["inliers"] >= 4
        and cnsfm_result["inlier_ratio"] >= 0.30
    )

    if loftr_good and cnsfm_good:
        decision = "STRONG MATCH"
    elif loftr_good or cnsfm_good:
        decision = "POSSIBLE MATCH"
    else:
        decision = "WEAK / UNVERIFIED"

    return {
        "loftr_score": loftr_score,
        "cnsfm_score": cnsfm_score,
        "agreement": agreement,
        "aggregate": float(aggregate),
        "decision": decision,
    }


# =============================================================================
# 9. VISUALIZATION
# =============================================================================

def draw_craters(img, craters, color=(255, 180, 0)):
    out = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2BGR,
    )

    for c in craters:
        cv2.circle(
            out,
            (int(round(c.x)), int(round(c.y))),
            int(round(c.r)),
            color,
            1,
            cv2.LINE_AA,
        )

        cv2.circle(
            out,
            (int(round(c.x)), int(round(c.y))),
            2,
            color,
            -1,
        )

    return out


def save_cnsfm_debug(img0, img1, cnsfm_result):
    a = draw_craters(
        img0,
        cnsfm_result["craters0"],
    )

    b = draw_craters(
        img1,
        cnsfm_result["craters1"],
    )

    canvas_h = max(a.shape[0], b.shape[0])
    canvas_w = a.shape[1] + b.shape[1]

    canvas = np.zeros(
        (canvas_h, canvas_w, 3),
        dtype=np.uint8,
    )

    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:] = b

    offset = a.shape[1]

    matches = cnsfm_result["matches"]
    mask = cnsfm_result["mask"]

    for i, m in enumerate(matches):
        c0 = cnsfm_result["craters0"][m["i0"]]
        c1 = cnsfm_result["craters1"][m["i1"]]

        p0 = (
            int(round(c0.x)),
            int(round(c0.y)),
        )

        p1 = (
            int(round(c1.x + offset)),
            int(round(c1.y)),
        )

        if i < len(mask) and mask[i]:
            color = (0, 255, 0)
            thickness = 2
        else:
            color = (0, 0, 255)
            thickness = 1

        cv2.line(
            canvas,
            p0,
            p1,
            color,
            thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        DEBUG_IMAGE,
        canvas,
    )

    print(
        f"[Saved] CNSFM debug image: {DEBUG_IMAGE}"
    )


def save_hybrid_visualization(
    img0,
    img1,
    loftr_result,
    cnsfm_result,
    aggregate,
):
    """
    Main visual result.

    LoFTR inliers are drawn in green.
    LoFTR outliers are drawn in red.
    CNSFM inlier crater matches are drawn in blue.
    """
    a = cv2.cvtColor(
        img0,
        cv2.COLOR_GRAY2BGR,
    )

    b = cv2.cvtColor(
        img1,
        cv2.COLOR_GRAY2BGR,
    )

    h = max(
        a.shape[0],
        b.shape[0],
    )

    canvas = np.zeros(
        (h, a.shape[1] + b.shape[1], 3),
        dtype=np.uint8,
    )

    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:] = b

    offset = a.shape[1]

    # LoFTR.
    pts0 = loftr_result["points0"]
    pts1 = loftr_result["points1"]
    mask = loftr_result["mask"]

    for i, (p0, p1) in enumerate(
        zip(pts0, pts1)
    ):
        x0, y0 = p0
        x1, y1 = p1

        if i < len(mask) and mask[i]:
            color = (0, 255, 0)
            thickness = 1
        else:
            color = (0, 0, 180)
            thickness = 1

        cv2.line(
            canvas,
            (
                int(round(x0)),
                int(round(y0)),
            ),
            (
                int(round(x1 + offset)),
                int(round(y1)),
            ),
            color,
            thickness,
            cv2.LINE_AA,
        )

    # CNSFM crater matches on top.
    cnsfm_mask = cnsfm_result["mask"]

    for i, m in enumerate(
        cnsfm_result["matches"]
    ):
        c0 = cnsfm_result["craters0"][m["i0"]]
        c1 = cnsfm_result["craters1"][m["i1"]]

        p0 = (
            int(round(c0.x)),
            int(round(c0.y)),
        )

        p1 = (
            int(round(c1.x + offset)),
            int(round(c1.y)),
        )

        if i < len(cnsfm_mask) and cnsfm_mask[i]:
            color = (255, 200, 0)
            thickness = 2

            cv2.circle(
                canvas,
                p0,
                4,
                color,
                -1,
            )

            cv2.circle(
                canvas,
                p1,
                4,
                color,
                -1,
            )

            cv2.line(
                canvas,
                p0,
                p1,
                color,
                thickness,
                cv2.LINE_AA,
            )

    # Add metrics in the top-left.
    lines = [
        f"LoFTR: {loftr_result['inliers']} inliers / "
        f"{loftr_result['matches']} matches",
        f"LoFTR ratio: {loftr_result['inlier_ratio']:.2%}",
        f"CNSFM: {cnsfm_result['inliers']} inliers / "
        f"{len(cnsfm_result['matches'])} graph matches",
        f"CNSFM ratio: {cnsfm_result['inlier_ratio']:.2%}",
        f"LoFTR score: {aggregate['loftr_score']:.3f}",
        f"CNSFM score: {aggregate['cnsfm_score']:.3f}",
        f"AGGREGATE: {aggregate['aggregate']:.3f}",
        f"DECISION: {aggregate['decision']}",
    ]

    y = 24

    for text in lines:
        cv2.putText(
            canvas,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

        y += 23

    cv2.imwrite(
        RESULT_IMAGE,
        canvas,
    )

    print(
        f"[Saved] Hybrid visualization: {RESULT_IMAGE}"
    )


# =============================================================================
# 10. MAIN PIPELINE
# =============================================================================

def run_hybrid(ohrc_xml, tmc_xml):
    print("=" * 78)
    print("SIH PS 26166 - HYBRID LoFTR + CNSFM CORRESPONDENCE")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Step 1: Parse metadata.
    # -------------------------------------------------------------------------
    print("\n[1/7] Parsing PDS4 metadata...")

    ohrc = parse_pds4_metadata(ohrc_xml)
    tmc = parse_pds4_metadata(tmc_xml)

    print(
        f"OHRC: {ohrc['lines']} x {ohrc['samples']} | "
        f"{ohrc['gsd']} m/px | {ohrc['dtype']}"
    )

    print(
        f"TMC : {tmc['lines']} x {tmc['samples']} | "
        f"{tmc['gsd']} m/px | {tmc['dtype']}"
    )

    # -------------------------------------------------------------------------
    # Step 2: Find physical overlap.
    # -------------------------------------------------------------------------
    print("\n[2/7] Mapping OHRC footprint into TMC...")

    geom = compute_ohrc_footprint_in_tmc(
        ohrc,
        tmc,
    )

    r_min, r_max, c_min, c_max = geom["tmc_bbox"]

    print(
        f"TMC crop: rows [{r_min}:{r_max}], "
        f"cols [{c_min}:{c_max}]"
    )

    print(
        "OHRC footprint projected to TMC:"
    )

    for p in geom["ohrc_polygon_tmc_px"]:
        print(
            f"    x={p[0]:.2f}, y={p[1]:.2f}"
        )

    # -------------------------------------------------------------------------
    # Step 3: Map crop back to OHRC.
    # -------------------------------------------------------------------------
    print("\n[3/7] Mapping the same physical crop into OHRC...")

    ohrc_geom = map_tmc_crop_to_ohrc(
        ohrc,
        tmc,
        geom,
    )

    oy0, oy1, ox0, ox1 = ohrc_geom["ohrc_bbox"]

    print(
        f"OHRC crop: rows [{oy0}:{oy1}], "
        f"cols [{ox0}:{ox1}]"
    )

    # -------------------------------------------------------------------------
    # Step 4: Read the same physical region.
    # -------------------------------------------------------------------------
    print("\n[4/7] Reading bounded image regions...")

    tmc_crop = read_pds4_window(
        tmc["img_path"],
        r_min,
        r_max,
        c_min,
        c_max,
        tmc["samples"],
        tmc["dtype"],
    )

    # OHRC source step needed to approximately reach TMC resolution.
    scale_ratio = ohrc["gsd"] / tmc["gsd"]

    # If OHRC GSD is 0.20 and TMC GSD is 6.13:
    #     scale_ratio ~= 0.0326
    # so one output pixel corresponds to about 30.65 OHRC pixels.
    source_step = max(
        1,
        int(round(1.0 / scale_ratio)),
    )

    print(
        f"OHRC -> TMC scale ratio: "
        f"{scale_ratio:.6f}"
    )

    print(
        f"OHRC source row decimation: "
        f"every {source_step} rows"
    )

    ohrc_crop = read_decimated_window(
        ohrc["img_path"],
        oy0,
        oy1,
        ox0,
        ox1,
        ohrc["samples"],
        ohrc["dtype"],
        source_step,
    )

    if ohrc_crop.size == 0 or tmc_crop.size == 0:
        raise RuntimeError(
            "One of the extracted image crops is empty."
        )

    # Match OHRC horizontal physical scale exactly by resize.
    target_w = max(
        32,
        int(round(
            ohrc_crop.shape[1]
            * ohrc["gsd"]
            / tmc["gsd"]
        )),
    )

    target_h = max(
        32,
        int(round(
            ohrc_crop.shape[0]
            * source_step
            * ohrc["gsd"]
            / tmc["gsd"]
        )),
    )

    ohrc_tmc_scale = cv2.resize(
        ohrc_crop,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    # -------------------------------------------------------------------------
    # Step 5: Common preprocessing.
    # -------------------------------------------------------------------------
    print("\n[5/7] Normalizing both sensors...")

    print(
        f"Applying strong OHRC->TMC degradation: "
        f"{OHRC_PIXEL_BLOCK}x pixel blocks, "
        f"{OHRC_GRAY_LEVELS} gray levels, "
        f"brightness x{OHRC_DARKEN_FACTOR:.2f}"
    )

    ohrc_prep = pixelate_and_tone_down_ohrc(
        ohrc_tmc_scale,
        tmc_crop,
    )

    tmc_prep = prepare_tmc_reference(
        tmc_crop
    )

    # The geographic crop should already have approximately the same physical
    # footprint. If dimensions differ because of corner/footprint approximation,
    # keep the actual images rather than forcing one image to the other.
    print(
        f"Prepared OHRC: {ohrc_prep.shape}"
    )

    print(
        f"Prepared TMC : {tmc_prep.shape}"
    )

    # -------------------------------------------------------------------------
    # Step 6: Run both independent branches.
    # -------------------------------------------------------------------------
    print("\n[6/7] Running LoFTR + CNSFM branches...")

    loftr_result = run_loftr_branch(
        ohrc_prep,
        tmc_prep,
    )

    cnsfm_result = run_cnsfm_branch(
        ohrc_prep,
        tmc_prep,
    )

    # -------------------------------------------------------------------------
    # Step 7: Aggregate.
    # -------------------------------------------------------------------------
    print("\n[7/7] Aggregating correspondence evidence...")

    aggregate = aggregate_confidence(
        loftr_result,
        cnsfm_result,
    )

    print("\n" + "=" * 78)
    print("FINAL HYBRID METRICS")
    print("=" * 78)

    print(
        f"LoFTR score       : "
        f"{aggregate['loftr_score']:.4f}"
    )

    print(
        f"CNSFM score       : "
        f"{aggregate['cnsfm_score']:.4f}"
    )

    print(
        f"Branch agreement  : "
        f"{aggregate['agreement']:.4f}"
    )

    print(
        f"Aggregate score   : "
        f"{aggregate['aggregate']:.4f}"
    )

    print(
        f"Decision          : "
        f"{aggregate['decision']}"
    )

    if np.isfinite(loftr_result["rmse"]):
        print(
            f"LoFTR RMSE        : "
            f"{loftr_result['rmse']:.3f} px"
        )
    else:
        print("LoFTR RMSE        : inf")

    if np.isfinite(cnsfm_result["rmse"]):
        print(
            f"CNSFM RMSE        : "
            f"{cnsfm_result['rmse']:.3f} px"
        )
    else:
        print("CNSFM RMSE        : inf")

    print("=" * 78)

    # Debug/visual outputs.
    save_cnsfm_debug(
        ohrc_prep,
        tmc_prep,
        cnsfm_result,
    )

    save_hybrid_visualization(
        ohrc_prep,
        tmc_prep,
        loftr_result,
        cnsfm_result,
        aggregate,
    )

    return {
        "ohrc": ohrc,
        "tmc": tmc,
        "geometry": geom,
        "ohrc_geometry": ohrc_geom,
        "ohrc_image": ohrc_prep,
        "tmc_image": tmc_prep,
        "loftr": loftr_result,
        "cnsfm": cnsfm_result,
        "aggregate": aggregate,
    }


# =============================================================================
# OHRC vs OHRC SANITY / EARLIER SAME-SENSOR EXAMPLE
# =============================================================================

def run_ohrc_vs_ohrc(ohrc_xml):
    """
    Run the SAME LoFTR + CNSFM pipeline on two overlapping OHRC crops.

    This reproduces the earlier same-sensor experiment using the current OHRC
    product. Since only one current OHRC product is configured here, the two
    inputs are overlapping crops from that product, not two independent
    acquisitions.

    The OHRC crops are deliberately degraded in the same way used for the
    OHRC->TMC experiment: strong downsampling/pixelation, coarse intensity
    quantization, blur and darkening. This lets us test whether the structural
    branch still works when appearance information is heavily degraded.
    """
    print("\n" + "=" * 78)
    print("OHRC vs OHRC - SAME PIPELINE SANITY TEST")
    print("=" * 78)

    meta = parse_pds4_metadata(ohrc_xml)
    print(
        f"OHRC: {meta['lines']} x {meta['samples']} | "
        f"{meta['gsd']} m/px | {meta['dtype']}"
    )

    # Two overlapping windows. The second is shifted so the matcher must
    # recover the correspondence instead of seeing identical arrays.
    # Keep the sanity test deliberately small so we never hold huge OHRC arrays
    # and their preprocessing copies in RAM.
    crop_h = min(4000, meta['lines'])
    crop_w = min(4000, meta['samples'])

    y0 = max(0, (meta['lines'] - crop_h) // 2)
    x0 = max(0, (meta['samples'] - crop_w) // 2)

    shift_y = min(600, max(200, crop_h // 8))
    shift_x = min(600, max(200, crop_w // 8))

    y1 = min(meta['lines'] - crop_h, y0 + shift_y)
    x1 = min(meta['samples'] - crop_w, x0 + shift_x)

    print(f"Crop A: rows [{y0}:{y0 + crop_h}], cols [{x0}:{x0 + crop_w}]")
    print(f"Crop B: rows [{y1}:{y1 + crop_h}], cols [{x1}:{x1 + crop_w}]")

    # IMPORTANT: keep the SAME 12000x12000 geographic crop as before, but do
    # not materialize both native-resolution crops in RAM. Read every 4th
    # source pixel/row into a compact representation first. The physical
    # region is unchanged; only the in-memory sampling is reduced.
    read_step = 4
    print(f"Low-RAM source sampling: every {read_step}th pixel/row")

    a = read_decimated_window(
        meta['img_path'], y0, y0 + crop_h, x0, x0 + crop_w,
        meta['samples'], meta['dtype'], read_step
    )
    a_n = normalize_percentile_clahe(a)
    a_p = pixelate_and_tone_down_ohrc(a_n, a_n)
    del a, a_n
    import gc
    gc.collect()

    b = read_decimated_window(
        meta['img_path'], y1, y1 + crop_h, x1, x1 + crop_w,
        meta['samples'], meta['dtype'], read_step
    )
    b_n = normalize_percentile_clahe(b)
    b_p = pixelate_and_tone_down_ohrc(b_n, b_n)
    del b, b_n
    gc.collect()

    # Keep LoFTR input bounded. This is the matcher resolution, NOT the
    # geographic crop size printed above.
    a_p, b_p, _ = resize_pair_for_loftr(a_p, b_p)

    print(f"Prepared OHRC A: {a_p.shape}")
    print(f"Prepared OHRC B: {b_p.shape}")

    print("\nRunning LoFTR + CNSFM on OHRC vs OHRC...")
    loftr_result = run_loftr_branch(a_p, b_p)
    cnsfm_result = run_cnsfm_branch(a_p, b_p)
    aggregate = aggregate_confidence(loftr_result, cnsfm_result)

    print("\n" + "=" * 78)
    print("OHRC vs OHRC METRICS")
    print("=" * 78)
    print(f"LoFTR score      : {aggregate['loftr_score']:.4f}")
    print(f"CNSFM score      : {aggregate['cnsfm_score']:.4f}")
    print(f"Branch agreement : {aggregate['agreement']:.4f}")
    print(f"Aggregate score  : {aggregate['aggregate']:.4f}")
    print(f"Decision         : {aggregate['decision']}")
    print(f"LoFTR matches    : {loftr_result['matches']}")
    print(f"LoFTR inliers    : {loftr_result['inliers']}")
    print(f"LoFTR ratio      : {loftr_result['inlier_ratio']:.4f}")
    print(f"LoFTR RMSE       : {loftr_result['rmse']:.3f}")
    print(f"CNSFM matches    : {len(cnsfm_result['matches'])}")
    print(f"CNSFM inliers    : {cnsfm_result['inliers']}")
    print(f"CNSFM ratio      : {cnsfm_result['inlier_ratio']:.4f}")
    print(f"CNSFM RMSE       : {cnsfm_result['rmse']:.3f}")
    print("=" * 78)

    save_cnsfm_debug(a_p, b_p, cnsfm_result)
    save_hybrid_visualization(
        a_p, b_p, loftr_result, cnsfm_result, aggregate
    )

    return {
        'image_a': a_p,
        'image_b': b_p,
        'loftr': loftr_result,
        'cnsfm': cnsfm_result,
        'aggregate': aggregate,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\nChoose experiment:")
    print("1 = OHRC vs TMC")
    print("2 = OHRC vs OHRC (earlier same-sensor example)")

    choice = input("Enter 1 or 2 [default=1]: ").strip()

    if choice == "2":
        run_ohrc_vs_ohrc(OHRC_XML)
    else:
        run_hybrid(OHRC_XML, TMC_XML)
