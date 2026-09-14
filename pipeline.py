import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import gc
import re
import math
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import algorithms
import algorithms

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

def is_streamlit_cloud() -> bool:
    """Detect if running inside Streamlit Community Cloud container (strict 1GB RAM limit)."""
    if os.path.exists("/mount/src") or os.environ.get("USER") == "appuser" or os.environ.get("HOME") == "/home/appuser":
        return True
    for k, v in os.environ.items():
        if "STREAMLIT" in k and any(x in k for x in ("SHARING", "CLOUD", "HOST")):
            return True
        if "streamlit.app" in str(v).lower():
            return True
    return False

def get_system_ram_gb() -> float:
    """Safely return container-aware RAM in GB."""
    if is_streamlit_cloud():
        return 1.0
    try:
        if os.path.exists("/sys/fs/cgroup/memory.max"):
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                val = f.read().strip()
                if val != "max":
                    return float(val) / (1024**3)
    except Exception:
        pass
    try:
        if os.path.exists("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                val = float(f.read().strip())
                if val < 1099511627776:
                    return val / (1024**3)
    except Exception:
        pass
    try:
        import psutil
        return float(psutil.virtual_memory().total / (1024**3))
    except Exception:
        pass
    try:
        return float((os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) / (1024**3))
    except Exception:
        pass
    return 16.0

ROOT = os.path.dirname(os.path.abspath(__file__))
R_MOON = 1737400.0

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
    "IEEE754LSBSingle": "<f4",
    "IEEE754MSBSingle": ">f4",
}

# =============================================================================
# MULTI-MISSION REFERENCE SENSOR SPECIFICATIONS
# =============================================================================
SENSOR_SPECS = {
    "TMC-2": {
        "mission": "Chandrayaan-2",
        "agency": "ISRO",
        "name": "Terrain Mapping Camera-2",
        "gsd": 5.0,
        "swath_km": 20.0,
        "passband": "0.50–0.85 µm (Panchromatic)",
        "scale_ratio_vs_ohrc": 20.0,
        "type": "Triplet Pushbroom Stereo (Fore, Nadir, Aft)",
        "focal_length_mm": 240.0,
    },
    "LRO_NAC": {
        "mission": "Lunar Reconnaissance Orbiter (LRO)",
        "agency": "NASA",
        "name": "Narrow Angle Camera (NAC-L / NAC-R)",
        "gsd": 0.50,
        "swath_km": 5.0,  # 10 km combined stereo
        "passband": "0.40–0.75 µm (Panchromatic)",
        "scale_ratio_vs_ohrc": 2.0,
        "type": "Ritchey-Chrétien Pushbroom Line-Scan",
        "focal_length_mm": 700.0,
    },
    "SELENE_TC": {
        "mission": "SELENE (Kaguya)",
        "agency": "JAXA",
        "name": "Terrain Camera (TC1 / TC2)",
        "gsd": 10.0,
        "swath_km": 35.0,
        "passband": "0.45–0.90 µm (Panchromatic)",
        "scale_ratio_vs_ohrc": 40.0,
        "type": "Stereoscopic Pushbroom (Fore, Aft)",
        "focal_length_mm": 72.5,
    },
    "OHRC": {
        "mission": "Chandrayaan-2",
        "agency": "ISRO",
        "name": "Optical High Resolution Camera",
        "gsd": 0.25,
        "swath_km": 3.0,
        "passband": "0.45–0.70 µm (Panchromatic)",
        "scale_ratio_vs_ohrc": 1.0,
        "type": "TDI Pushbroom Sub-Meter Imager",
        "focal_length_mm": 560.0,
    }
}

# CNSFM Config
MIN_CRATER_RADIUS_M = 30.0
MAX_CRATER_RADIUS_M = 500.0
MAX_CRATERS = 120
K_NEIGHBORS = 6
HOUGH_DP = 1.2
HOUGH_PARAM1 = 70
HOUGH_PARAM2 = 16
HOUGH_MIN_DIST_M = 80.0
DESCRIPTOR_RATIO = 0.82
DESCRIPTOR_ABS_DISTANCE = 1.55

# LoFTR Config
MAX_LOFTR_SIDE = 1600
LOFTR_PRETRAINED = "outdoor"
LOFTR_CONFIDENCE_MIN = 0.20

# Legacy Preprocessing Config
OHRC_PIXEL_BLOCK = 3
OHRC_GRAY_LEVELS = 32
OHRC_DARKEN_FACTOR = 0.62
OHRC_GAMMA = 1.12
OHRC_BLUR_SIGMA = 1.0
TMC_GAMMA = 1.05

TMC_PAD_PX = 20

# =============================================================================
# PDS4 I/O SECTION
# =============================================================================

def parse_pds4_metadata(xml_path: str) -> dict:
    """Unified PDS4 metadata parser."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def strip_ns(tag):
        return tag.split("}")[-1] if "}" in tag else tag

    meta = {}
    
    # Try pds: namespace first (newbase style)
    pds_ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}
    array_el = root.find(".//pds:Array_2D_Image", pds_ns)
    if array_el is not None:
        offset_el = array_el.find("pds:offset", pds_ns)
        if offset_el is not None:
            meta["offset"] = int(offset_el.text)
        dtype_name = array_el.find("pds:Element_Array/pds:data_type", pds_ns).text.strip()
        if dtype_name in DTYPE_MAP:
            meta["dtype"] = DTYPE_MAP[dtype_name]
        
        for axis in array_el.findall("pds:Axis_Array", pds_ns):
            name = axis.find("pds:axis_name", pds_ns).text.strip().lower()
            elements = int(axis.find("pds:elements", pds_ns).text)
            if name == "line":
                meta["lines"] = elements
            elif name in ("sample", "samples"):
                meta["samples"] = elements

    # Fallback to naive parsing (baseline style)
    if "lines" not in meta or "dtype" not in meta:
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
            elif tag == "offset" and "offset" not in meta and elem.text:
                try:
                    meta["offset"] = int(elem.text.strip())
                except ValueError:
                    pass

    if "offset" not in meta:
        meta["offset"] = 0

    # Corners
    wanted = {
        "upper_left_latitude": "UL_LAT", "upper_left_longitude": "UL_LON",
        "upper_right_latitude": "UR_LAT", "upper_right_longitude": "UR_LON",
        "lower_right_latitude": "LR_LAT", "lower_right_longitude": "LR_LON",
        "lower_left_latitude": "LL_LAT", "lower_left_longitude": "LL_LON",
    }
    values = {}
    for elem in root.iter():
        tag = strip_ns(elem.tag).lower()
        if tag in wanted and elem.text:
            try:
                values[wanted[tag]] = float(elem.text.strip())
            except ValueError:
                pass

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

    required = ["UL_LAT", "UL_LON", "UR_LAT", "UR_LON", "LR_LAT", "LR_LON", "LL_LAT", "LL_LON"]
    missing = [x for x in required if x not in values]
    if missing:
        raise ValueError(f"Could not locate geographic corners {missing} in {xml_path}")

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
        meta["gsd"] = 1.0
        meta["name"] = "UNKNOWN"

    img_path = os.path.splitext(xml_path)[0] + ".img"
    if not os.path.exists(img_path):
        img_path = os.path.splitext(xml_path)[0] + ".IMG"
    if not os.path.exists(img_path):
        # check filename tag
        file_el = root.find(".//pds:File", pds_ns)
        if file_el is not None:
            fn_el = file_el.find("pds:file_name", pds_ns)
            if fn_el is not None:
                cand = os.path.join(os.path.dirname(os.path.abspath(xml_path)), fn_el.text.strip())
                if os.path.exists(cand):
                    img_path = cand

    meta["img_path"] = img_path if (img_path and os.path.exists(img_path)) else None

    return meta

def load_pds4_window(filepath, r_start, r_end, c_start, c_end, total_samples, dtype, offset=0) -> np.ndarray:
    """Line-by-line disk read."""
    r_start, r_end = max(0, int(r_start)), max(0, int(r_end))
    c_start, c_end = max(0, int(c_start)), max(0, int(c_end))
    r_end = max(r_start, r_end)
    c_end = max(c_start, c_end)

    rows = r_end - r_start
    cols = c_end - c_start

    out = np.zeros((rows, cols), dtype=dtype)
    itemsize = np.dtype(dtype).itemsize
    row_stride = total_samples * itemsize
    bytes_per_row = cols * itemsize

    with open(filepath, "rb") as f:
        for i, row in enumerate(range(r_start, r_end)):
            pos = offset + row * row_stride + c_start * itemsize
            f.seek(pos)
            raw = f.read(bytes_per_row)
            if len(raw) == bytes_per_row:
                out[i] = np.frombuffer(raw, dtype=dtype)
    return out

def load_pds4_decimated(filepath, r_start, r_end, c_start, c_end, total_samples, dtype, step, offset=0) -> np.ndarray:
    """Decimated read for OHRC."""
    r_start, r_end = max(0, int(r_start)), max(0, int(r_end))
    c_start, c_end = max(0, int(c_start)), max(0, int(c_end))
    r_end = max(r_start, r_end)
    c_end = max(c_start, c_end)
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
            pos = offset + row * row_stride + c_start * itemsize
            f.seek(pos)
            raw = f.read(bytes_per_row)
            if len(raw) == bytes_per_row:
                out[out_i] = np.frombuffer(raw, dtype=dtype)
            out_i += 1
    return out

# =============================================================================
# GEOGRAPHIC FOOTPRINT SECTION
# =============================================================================

def to_local_metric(lat, lon, lat0, lon0) -> Tuple[float, float]:
    phi = math.radians(lat)
    lam = math.radians(lon)
    phi0 = math.radians(lat0)
    lam0 = math.radians(lon0)
    x = R_MOON * (lam - lam0) * math.cos(phi0)
    y = R_MOON * (phi - phi0)
    return x, y

def compute_footprint(ohrc_meta, tmc_meta) -> dict:
    """Projects OHRC corners into TMC pixel space."""
    order = ["UL", "UR", "LR", "LL"]
    ohrc_lats = [ohrc_meta["corners"][k][0] for k in order]
    ohrc_lons = [ohrc_meta["corners"][k][1] for k in order]
    lat0 = float(np.mean(ohrc_lats))
    lon0 = float(np.mean(ohrc_lons))

    tmc_metric = np.array([to_local_metric(tmc_meta["corners"][k][0], tmc_meta["corners"][k][1], lat0, lon0) for k in order], dtype=np.float32)
    tmc_pixels = np.array([[0, 0], [tmc_meta["samples"] - 1, 0], [tmc_meta["samples"] - 1, tmc_meta["lines"] - 1], [0, tmc_meta["lines"] - 1]], dtype=np.float32)
    H_metric_to_tmc = cv2.getPerspectiveTransform(tmc_metric, tmc_pixels)

    ohrc_metric = np.array([to_local_metric(ohrc_meta["corners"][k][0], ohrc_meta["corners"][k][1], lat0, lon0) for k in order], dtype=np.float32)
    ohrc_in_tmc = cv2.perspectiveTransform(ohrc_metric.reshape(-1, 1, 2), H_metric_to_tmc).reshape(-1, 2)

    pad = TMC_PAD_PX
    c_min = max(0, int(np.floor(np.min(ohrc_in_tmc[:, 0]))) - pad)
    c_max = min(tmc_meta["samples"], int(np.ceil(np.max(ohrc_in_tmc[:, 0]))) + pad)
    r_min = max(0, int(np.floor(np.min(ohrc_in_tmc[:, 1]))) - pad)
    r_max = min(tmc_meta["lines"], int(np.ceil(np.max(ohrc_in_tmc[:, 1]))) + pad)

    return {
        "tmc_bbox": (r_min, r_max, c_min, c_max),
        "ohrc_bbox": (0, ohrc_meta["lines"], 0, ohrc_meta["samples"]), # Default full
        "ohrc_polygon_tmc_px": ohrc_in_tmc,
        "proj_origin": (lat0, lon0)
    }

# =============================================================================
# IMAGE PREPROCESSING SECTION
# =============================================================================

def normalize_percentile(img, p_low=1.0, p_high=99.0) -> np.ndarray:
    img_f = img.astype(np.float32)
    p1, p99 = np.percentile(img_f, (p_low, p_high))
    if p99 <= p1:
        return np.zeros_like(img_f, dtype=np.uint8)
    out = np.clip((img_f - p1) / (p99 - p1) * 255.0, 0, 255).astype(np.uint8)
    return out

def apply_clahe(img, clip_limit=2.0, grid_size=8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(img)

def build_scale_space_pyramid(img: np.ndarray, num_octaves: int = 5) -> List[np.ndarray]:
    """Constructs a Gaussian scale-space image pyramid.
    
    Each octave applies an anti-aliasing Gaussian smoothing kernel followed by
    dyadic (2x) sub-sampling, preserving low spatial frequencies while preventing
    high-frequency aliasing and moiré artifacts.
    """
    pyramid = [img]
    curr = img.astype(np.float32) if img.dtype != np.float32 else img.copy()
    for _ in range(num_octaves):
        if min(curr.shape[:2]) <= 16:
            break
        curr = cv2.pyrDown(curr)
        pyramid.append(curr.astype(img.dtype) if img.dtype != np.float32 else curr.copy())
    return pyramid

def scale_space_downsample(
    img: np.ndarray,
    scale_factor: float = 20.0,
    src_gsd: Optional[float] = None,
    tgt_gsd: Optional[float] = None,
) -> np.ndarray:
    """Multi-octave Gaussian scale-space pyramid downsampler for multi-sensor lunar imagery.
    
    Dynamically bridges large resolution gaps (e.g. 20:1 between 0.25 m/px OHRC and 5.0 m/px TMC-2).
    Instead of single-step decimation (which causes catastrophic high-frequency aliasing and
    breaks Fourier phase correlation), this function:
      1. Determines the exact continuous downsampling factor from resolution/GSD metadata:
         scale_factor = tgt_gsd / src_gsd (e.g. 5.0 / 0.25 = 20.0).
      2. Progressively applies octave Gaussian low-pass filtering and 2x sub-sampling
         (cv2.pyrDown) while the remaining factor >= 2.0.
      3. For the remaining fractional factor s in [1.0, 2.0), applies a matched scale-space
         Gaussian filter (sigma = sqrt(max(0.01, s^2 - 1.0)) * 0.85) and area-weighted
         interpolation (cv2.INTER_AREA) to the exact target pixel dimensions.
    
    Ensures input features are low-pass filtered to the Nyquist limit of the reference sensor
    grid, enabling robust Fourier phase correlation and dense matching across all swath segments.
    """
    if src_gsd is not None and tgt_gsd is not None and src_gsd > 0:
        scale_factor = float(tgt_gsd) / float(src_gsd)
    
    # If scale_factor is given as a fraction < 1.0 (e.g. 0.05 = 1/20), invert it
    if 0.0 < scale_factor < 1.0:
        scale_factor = 1.0 / scale_factor
        
    if scale_factor <= 1.001 or img is None or img.size == 0:
        return img

    orig_dtype = img.dtype
    curr = img.astype(np.float32) if orig_dtype != np.float32 else img.copy()
    remaining = float(scale_factor)
    
    # Octave scale-space reductions
    while remaining >= 2.0 and min(curr.shape[:2]) > 16:
        curr = cv2.pyrDown(curr)
        remaining /= 2.0
        
    # Fractional reduction to exact target dimensions
    target_w = max(16, int(round(img.shape[1] / scale_factor)))
    target_h = max(16, int(round(img.shape[0] / scale_factor)))
    
    if curr.shape[1] != target_w or curr.shape[0] != target_h:
        if remaining > 1.001:
            sigma = float(np.sqrt(max(0.01, remaining**2 - 1.0))) * 0.85
            curr = cv2.GaussianBlur(curr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        curr = cv2.resize(curr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
    if orig_dtype == np.uint8:
        return np.clip(curr, 0, 255).astype(np.uint8)
    elif orig_dtype == np.uint16:
        return np.clip(curr, 0, 65535).astype(np.uint16)
    return curr

def gaussian_downsample(img, scale) -> np.ndarray:
    """Scale-space Gaussian downsampler."""
    if scale >= 1.0:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return scale_space_downsample(img, scale_factor=1.0 / scale)

def prepare_images(
    ohrc_crop: np.ndarray,
    tmc_crop: np.ndarray,
    mode: str = 'phase_congruency',
    src_gsd: Optional[float] = None,
    tgt_gsd: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Illumination-invariant and scale-space normalized feature preprocessing.
    
    Dynamically checks and normalizes resolution mismatch between target and reference
    images before extracting phase congruency or gradient feature maps.
    """
    # 1. Scale-space resolution normalization if resolution gap > 1.5x
    if src_gsd is not None and tgt_gsd is not None and tgt_gsd > src_gsd * 1.5:
        ohrc_scaled = scale_space_downsample(ohrc_crop, src_gsd=src_gsd, tgt_gsd=tgt_gsd)
    else:
        ho, wo = ohrc_crop.shape[:2]
        ht, wt = tmc_crop.shape[:2]
        if wo > wt * 5 or ho > ht * 5:
            factor = max(float(wo) / max(1, wt), float(ho) / max(1, ht))
            ohrc_scaled = scale_space_downsample(ohrc_crop, scale_factor=factor)
        else:
            ohrc_scaled = ohrc_crop

    ohrc_norm = normalize_percentile(ohrc_scaled)
    tmc_norm = normalize_percentile(tmc_crop)
    
    if mode == 'phase_congruency':
        o_pc = algorithms.phase_congruency_2d(ohrc_norm)
        t_pc = algorithms.phase_congruency_2d(tmc_norm)
        return (o_pc * 255).astype(np.uint8), (t_pc * 255).astype(np.uint8)
    elif mode == 'ngf':
        ox, oy, _ = algorithms.compute_ngf(ohrc_norm)
        tx, ty, _ = algorithms.compute_ngf(tmc_norm)
        omag = np.sqrt(ox**2 + oy**2)
        tmag = np.sqrt(tx**2 + ty**2)
        return normalize_percentile(omag), normalize_percentile(tmag)
    elif mode == 'legacy':
        return apply_clahe(ohrc_norm), apply_clahe(tmc_norm)
    else:
        return ohrc_norm, tmc_norm

# =============================================================================
# COARSE ALIGNMENT SECTION
# =============================================================================

def coarse_to_fine_align(
    ohrc_prep: np.ndarray,
    tmc_prep: np.ndarray,
    thumb_size: int = 512,
    src_gsd: Optional[float] = None,
    tgt_gsd: Optional[float] = None,
) -> dict:
    """Two-stage autonomous coarse-to-fine alignment with scale-space normalization.
    
    Stage 1: Bridges resolution gaps via multi-octave scale-space filtering,
    resamples with aspect-ratio preservation, and uses Log-Polar Fourier-Mellin
    transform with orbital flight-corridor constraints to recover rotation and scale.
    Stage 2: Derotates and centers the OHRC patch onto the TMC canvas, then
    performs sub-pixel phase correlation to refine translation (dx, dy).
    """
    print("[PIPELINE] Coarse alignment via Log-Polar Fourier-Mellin...")
    
    # Scale-space normalize if resolution gap is present
    if src_gsd is not None and tgt_gsd is not None and tgt_gsd > src_gsd * 1.5:
        ohrc_prep = scale_space_downsample(ohrc_prep, src_gsd=src_gsd, tgt_gsd=tgt_gsd)
        
    ht, wt = tmc_prep.shape[:2]
    ho, wo = ohrc_prep.shape[:2]

    # Aspect-preserving multi-scale thumbnail for Fourier-Mellin rotation/scale recovery
    scale_o = float(thumb_size) / max(ho, wo)
    scale_t = float(thumb_size) / max(ht, wt)
    new_wo, new_ho = max(16, int(round(wo * scale_o))), max(16, int(round(ho * scale_o)))
    new_wt, new_ht = max(16, int(round(wt * scale_t))), max(16, int(round(ht * scale_t)))
    
    # Common canvas size for Log-Polar Fourier-Mellin
    common_dim = max(new_wo, new_ho, new_wt, new_ht, 256)
    o_thumb = np.zeros((common_dim, common_dim), dtype=np.uint8)
    t_thumb = np.zeros((common_dim, common_dim), dtype=np.uint8)
    
    o_res = cv2.resize(ohrc_prep, (new_wo, new_ho), interpolation=cv2.INTER_AREA)
    t_res = cv2.resize(tmc_prep, (new_wt, new_ht), interpolation=cv2.INTER_AREA)
    
    o_thumb[:new_ho, :new_wo] = o_res
    t_thumb[:new_ht, :new_wt] = t_res
    
    res = algorithms.log_polar_fourier_mellin(t_thumb, o_thumb)
    raw_rot = res.get('rotation_deg', 0.0)
    fmt_conf = res.get('confidence', 0.0)
    scale = float(np.clip(res.get('scale', 1.0), 0.85, 1.15))
    
    # In Chandrayaan-2 lunar near-polar orbits, spacecraft roll/pitch/yaw is bounded within +/-5 deg.
    rot_cand = (raw_rot + 180.0) % 360.0 - 180.0
    if abs(rot_cand) > 45.0:
        alt_rot = (rot_cand + 180.0) % 360.0 - 180.0
        if abs(alt_rot) < abs(rot_cand):
            rot_cand = alt_rot
    rot_bounded = float(np.clip(rot_cand, -5.0, 5.0))
    
    # Evaluate candidates: 0.0° (spacecraft nominal co-aligned geometry) vs rot_bounded
    best_score = -999.0
    best_rot = 0.0
    best_dx, best_dy = 0.0, 0.0
    best_M = None
    
    candidates = [0.0]
    if fmt_conf > 0.02 and abs(rot_bounded) > 0.1:
        candidates.append(rot_bounded)
        
    for c_rot in candidates:
        M_rot = cv2.getRotationMatrix2D((wo / 2.0, ho / 2.0), c_rot, 1.0 / scale)
        M_rot[0, 2] += (wt - wo) / 2.0
        M_rot[1, 2] += (ht - ho) / 2.0
        derotated = cv2.warpAffine(ohrc_prep.astype(np.float32), M_rot, (wt, ht), flags=cv2.INTER_LINEAR)
        (dx, dy), score = algorithms.phase_correlation_subpx(tmc_prep, derotated)
        if score > best_score:
            best_score = score
            best_rot = c_rot
            best_dx, best_dy = dx, dy
            M_final = M_rot.copy()
            M_final[0, 2] += dx
            M_final[1, 2] += dy
            best_M = M_final
            
    print(f"[PIPELINE] Coarse alignment: rot={best_rot:.2f}°, scale={scale:.4f}, dx={best_dx:.2f}, dy={best_dy:.2f}, peak={best_score:.4f}")
    
    return {
        'rotation_deg': best_rot,
        'scale': scale,
        'translation': (best_dx, best_dy),
        'transform_matrix': best_M,
        'confidence': best_score,
    }

# =============================================================================
# FEATURE MATCHING SECTION
# =============================================================================

def _empty_matches():
    return {"matches": 0, "inliers": 0, "inlier_ratio": 0.0, "rmse": float("inf"),
            "points0": np.empty((0,2)), "points1": np.empty((0,2)), "mask": np.empty(0, dtype=bool),
            "H": None, "score": 0.0}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def run_loftr_branch(img0, img1) -> dict:
    try:
        from kornia.feature import LoFTR
        print("\n[LoFTR] Running appearance branch...")
        dev = get_device()
        matcher = LoFTR(pretrained=LOFTR_PRETRAINED).to(dev).eval()

        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        scale = min(1.0, MAX_LOFTR_SIDE / max(h0, w0), MAX_LOFTR_SIDE / max(h1, w1))
        
        if scale < 1.0:
            nw0, nh0 = max(32, int(round(w0 * scale))), max(32, int(round(h0 * scale)))
            nw1, nh1 = max(32, int(round(w1 * scale))), max(32, int(round(h1 * scale)))
            i0 = cv2.resize(img0, (nw0, nh0), interpolation=cv2.INTER_AREA)
            i1 = cv2.resize(img1, (nw1, nh1), interpolation=cv2.INTER_AREA)
        else:
            i0, i1 = img0.copy(), img1.copy()
            scale = 1.0
        
        def pad8(im):
            h, w = im.shape[:2]
            nh, nw = ((h + 7) // 8) * 8, ((w + 7) // 8) * 8
            padded = np.zeros((nh, nw), dtype=np.uint8)
            padded[:h, :w] = im
            return padded, (h, w)

        p0, orig_shape0 = pad8(i0)
        p1, orig_shape1 = pad8(i1)
        t0 = torch.from_numpy(p0).float()[None, None].to(dev) / 255.0
        t1 = torch.from_numpy(p1).float()[None, None].to(dev) / 255.0
        
        with torch.no_grad():
            out = matcher({"image0": t0, "image1": t1})
        
        kpts0 = out["keypoints0"].cpu().numpy()
        kpts1 = out["keypoints1"].cpu().numpy()
        conf = out["confidence"].cpu().numpy()

        # Filter for genuine matches with valid bounds and confidence
        valid = (
            (kpts0[:, 0] >= 0) & (kpts0[:, 0] < orig_shape0[1]) &
            (kpts0[:, 1] >= 0) & (kpts0[:, 1] < orig_shape0[0]) &
            (kpts1[:, 0] >= 0) & (kpts1[:, 0] < orig_shape1[1]) &
            (kpts1[:, 1] >= 0) & (kpts1[:, 1] < orig_shape1[0]) &
            (conf >= 0.30)
        )
        
        # Rescale coordinates back to original full resolution
        pts0 = kpts0[valid] / scale
        pts1 = kpts1[valid] / scale
        
        res = ransac_homography(pts0, pts1, threshold=5.0, img_ref=img1, img_tgt=img0)
        res['score'] = float(res['inlier_ratio'])
        print(f"[LoFTR] Matches: {res['matches']}, Inliers: {res['inliers']} ({res['inlier_ratio']*100:.1f}%), RMSE: {res['rmse']:.2f}px")
        return res
    except Exception as e:
        print(f"[LoFTR] Error: {e}")
        return _empty_matches()

def run_roma_branch(img0, img1, device=None, num_samples=5000) -> dict:
    try:
        # Check system RAM before attempting to load 1.55 GB RoMa + DINOv2 weights.
        # Streamlit Community Cloud enforces a 1.0 GB cgroup memory limit which triggers an instant SIGKILL.
        ram_gb = get_system_ram_gb()
        if ram_gb < 3.5 and not (device == "cuda" or (torch.cuda.is_available() and device != "cpu")):
            print(f"[RoMa] Memory constrained container ({ram_gb:.1f}GB RAM, no CUDA GPU). Safely falling back to SIFT.")
            return run_sift_branch(img0, img1)

        from romatch import roma_outdoor
        import tempfile
        print("\n[RoMa] Running certainty-guided dense matching...")
        # RoMa uses cholesky solve which is unsupported on Apple MPS, default to CPU on Mac
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = roma_outdoor(device=dev)
        for m in model.modules():
            if hasattr(m, "use_custom_corr"):
                m.use_custom_corr = False
                
        def save_tmp(img):
            rgb = cv2.cvtColor(normalize_percentile(img), cv2.COLOR_GRAY2RGB)
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv2.imwrite(f.name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            return f.name

        h0, w0 = img0.shape[:2]
        
        # NASA/ISRO Pushbroom Along-Track Tiling:
        # If image is an elongated orbital pushbroom strip (aspect ratio > 2.2),
        # slice into along-track square tiles to eliminate the 6:1 vertical squashing distortion
        if h0 / max(1, w0) > 2.2:
            print(f"[RoMa] Pushbroom strip detected ({h0}x{w0}) - deploying NASA/ISRO along-track 3-region tiling...")
            tile_len = min(750, h0)
            offsets = [0, max(0, (h0 - tile_len) // 2), max(0, h0 - tile_len)]
            offsets = sorted(list(set(offsets)))
            all_p0, all_p1, all_certs = [], [], []
            
            for r0 in offsets:
                r1 = min(h0, r0 + tile_len)
                tile_o = img0[r0:r1, :]
                tile_t = img1[r0:r1, :]
                
                p0, p1 = save_tmp(tile_o), save_tmp(tile_t)
                warp, cert = model.match(p0, p1, device=dev)
                os.unlink(p0)
                os.unlink(p1)
                
                s_matches, s_cert = model.sample(warp, cert, num=min(700, num_samples // len(offsets)))
                k0, k1 = model.to_pixel_coordinates(s_matches, tile_o.shape[0], tile_o.shape[1], tile_t.shape[0], tile_t.shape[1])
                
                p0_np = k0.cpu().numpy()
                p1_np = k1.cpu().numpy()
                c_np = s_cert.cpu().numpy()
                
                # Transform to global pushbroom coordinates
                p0_np[:, 1] += r0
                p1_np[:, 1] += r0
                
                # Adaptive orbital displacement consistency filter:
                disp = p1_np - p0_np
                disp_norm = np.linalg.norm(disp, axis=1)
                if len(disp) > 8:
                    med_disp = np.median(disp, axis=0)
                    res_disp = np.linalg.norm(disp - med_disp, axis=1)
                    valid_disp = (res_disp < 10.0) & (disp_norm < 40.0)
                else:
                    valid_disp = disp_norm < 30.0
                
                if np.any(valid_disp):
                    all_p0.append(p0_np[valid_disp])
                    all_p1.append(p1_np[valid_disp])
                    all_certs.append(c_np[valid_disp])
                    
            if all_p0:
                pts0 = np.concatenate(all_p0, axis=0)
                pts1 = np.concatenate(all_p1, axis=0)
            else:
                return _empty_matches()

        else:
            p0, p1 = save_tmp(img0), save_tmp(img1)
            warp, cert = model.match(p0, p1, device=dev)
            os.unlink(p0)
            os.unlink(p1)

            s_matches, s_cert = model.sample(warp, cert, num=num_samples)
            kpts0, kpts1 = model.to_pixel_coordinates(s_matches, img0.shape[0], img0.shape[1], img1.shape[0], img1.shape[1])
            pts0, pts1 = kpts0.cpu().numpy(), kpts1.cpu().numpy()
        
        res = ransac_homography(pts0, pts1, threshold=4.5, img_ref=img1, img_tgt=img0)
        res['score'] = float(res['inlier_ratio'])
        print(f"[RoMa] Matches: {res['matches']}, Inliers: {res['inliers']} ({res['inlier_ratio']*100:.2f}%), RMSE: {res['rmse']:.2f}px")
        return res
    except Exception as e:
        print(f"[RoMa] Error: {e}")
        return _empty_matches()

def run_lightglue_branch(img0, img1, device=None, max_keypoints=2048) -> dict:
    try:
        from lightglue import LightGlue, DISK
        from lightglue.utils import rbd
        # Use CUDA if available; fall back to CPU (avoids MPS aten::kthvalue op limitation on Mac)
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def to_tensor(img):
            rgb = np.stack([img]*3, axis=0).astype(np.float32) / 255.0
            return torch.from_numpy(rgb).to(dev)

        ext = DISK(max_num_keypoints=max_keypoints).eval().to(dev)
        matcher = LightGlue(features="disk").eval().to(dev)
        
        f0 = ext.extract(to_tensor(img0))
        f1 = ext.extract(to_tensor(img1))
        m = matcher({"image0": f0, "image1": f1})
        f0, f1, m = [rbd(x) for x in [f0, f1, m]]
        
        idx = m["matches"]
        pts0 = f0["keypoints"][idx[:,0]].cpu().numpy()
        pts1 = f1["keypoints"][idx[:,1]].cpu().numpy()
        
        res = ransac_homography(pts0, pts1, img_ref=img1, img_tgt=img0)
        res['score'] = res['inlier_ratio']
        return res
    except Exception as e:
        print(f"[LightGlue] Error: {e}")
        return _empty_matches()

def run_sift_branch(img0, img1) -> dict:
    try:
        print("\n[SIFT] Running...")
        sift = cv2.SIFT_create(nfeatures=4000)
        kp0, des0 = sift.detectAndCompute(img0, None)
        kp1, des1 = sift.detectAndCompute(img1, None)
        if des0 is None or des1 is None: return _empty_matches()
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des0, des1, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
        
        res = ransac_homography(pts0, pts1, img_ref=img1, img_tgt=img0)
        res['score'] = res['inlier_ratio']
        return res
    except Exception as e:
        print(f"[SIFT] Error: {e}")
        return _empty_matches()

def run_orb_branch(img0, img1) -> dict:
    try:
        print("\n[ORB] Running...")
        orb = cv2.ORB_create(nfeatures=4000)
        kp0, des0 = orb.detectAndCompute(img0, None)
        kp1, des1 = orb.detectAndCompute(img1, None)
        if des0 is None or des1 is None: return _empty_matches()
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des0, des1, k=2)
        good = []
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < 0.75 * n.distance:
                    good.append(m)
                    
        pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
        
        res = ransac_homography(pts0, pts1, img_ref=img1, img_tgt=img0)
        res['score'] = res['inlier_ratio']
        return res
    except Exception as e:
        print(f"[ORB] Error: {e}")
        return _empty_matches()

@dataclass
class Crater:
    x: float
    y: float
    r: float
    score: float

def detect_craters(img):
    norm = img.astype(np.uint8)
    smooth = cv2.GaussianBlur(norm, (5, 5), 1.2)
    h, w = smooth.shape
    min_dim = min(h, w)
    min_r = max(2, int(round(MIN_CRATER_RADIUS_M / CURRENT_GSD["TMC-2"])))
    max_r = max(min_r + 2, int(round(MAX_CRATER_RADIUS_M / CURRENT_GSD["TMC-2"])))
    max_r = min(max_r, max(8, int(min_dim * 0.35)))
    min_dist = max(5, int(round(HOUGH_MIN_DIST_M / CURRENT_GSD["TMC-2"])))

    circles = cv2.HoughCircles(smooth, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=min_dist,
                               param1=HOUGH_PARAM1, param2=HOUGH_PARAM2, minRadius=min_r, maxRadius=max_r)
    if circles is None: return []
    circles = np.round(circles[0]).astype(np.float32)
    
    candidates = []
    for x, y, r in circles:
        if x-r < 1 or y-r < 1 or x+r >= w-1 or y+r >= h-1: continue
        angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
        xs = np.rint(x + r * np.cos(angles)).astype(np.int32)
        ys = np.rint(y + r * np.sin(angles)).astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        ring = smooth[ys[valid], xs[valid]]
        if len(ring) < 10: continue
        score = float(np.std(ring))/64.0
        candidates.append(Crater(float(x), float(y), float(r), score))
        
    candidates.sort(key=lambda c: c.score, reverse=True)
    selected = []
    for c in candidates:
        keep = True
        for s in selected:
            if math.hypot(c.x - s.x, c.y - s.y) < 0.45 * max(c.r, s.r):
                keep = False; break
        if keep: selected.append(c)
        if len(selected) >= MAX_CRATERS: break
    return selected

def pairwise_distance_matrix(points):
    n = len(points)
    if n == 0: return np.empty((0, 0), dtype=np.float32)
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def local_crater_descriptor(craters, index, k=K_NEIGHBORS):
    center = craters[index]
    if len(craters) <= 1: return None
    points = np.array([[c.x, c.y] for c in craters], dtype=np.float32)
    radii = np.array([c.r for c in craters], dtype=np.float32)
    distances = np.linalg.norm(points - np.array([center.x, center.y]), axis=1)
    order = [j for j in np.argsort(distances) if j != index][:k]
    if len(order) < 3: return None
    
    neighbor_points = points[order]
    d = distances[order]
    local_scale = max(float(np.median(d)), 1e-6)
    norm_d = d / local_scale
    radius_ratio = radii[order] / max(center.r, 1e-6)
    pd = pairwise_distance_matrix(neighbor_points)
    
    tri = []
    for i in range(len(order)):
        for j in range(i+1, len(order)):
            tri.append(pd[i, j] / local_scale)
            
    angles = np.arctan2(neighbor_points[:, 1] - center.y, neighbor_points[:, 0] - center.x)
    angles = np.sort((angles + 2*np.pi) % (2*np.pi))
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2*np.pi])) / (2*np.pi)
    
    return np.concatenate([np.sort(norm_d), np.sort(radius_ratio), np.sort(tri), np.sort(gaps)]).astype(np.float32)

def build_crater_descriptors(craters):
    descriptors = []
    for i in range(len(craters)):
        d = local_crater_descriptor(craters, i)
        if d is not None: descriptors.append((i, d))
    return descriptors

def pad_descriptor(a, length):
    out = np.zeros(length, dtype=np.float32)
    n = min(length, len(a))
    out[:n] = a[:n]
    return out

def descriptor_distance(a, b):
    length = max(len(a), len(b))
    aa, bb = pad_descriptor(a, length), pad_descriptor(b, length)
    return float(np.mean(np.abs(aa - bb) / (1.0 + np.abs(aa) + np.abs(bb))))

def match_crater_graphs(craters0, craters1):
    d0, d1 = build_crater_descriptors(craters0), build_crater_descriptors(craters1)
    if not d0 or not d1: return [], 0.0
    cands = []
    for i0, v0 in d0:
        dists = [(descriptor_distance(v0, v1), i1) for i1, v1 in d1]
        dists.sort()
        bd, bi = dists[0]
        sd = dists[1][0] if len(dists) > 1 else bd + 1.0
        r = bd / max(sd, 1e-6)
        if r <= DESCRIPTOR_RATIO and bd <= DESCRIPTOR_ABS_DISTANCE:
            cands.append((bd, r, i0, bi))
            
    cands.sort()
    used0, used1, matches = set(), set(), []
    for d, r, i0, i1 in cands:
        if i0 in used0 or i1 in used1: continue
        used0.add(i0); used1.add(i1)
        matches.append({"i0": i0, "i1": i1, "distance": d, "ratio": r})
    
    if not matches: return [], 0.0
    mean_d = float(np.mean([m["distance"] for m in matches]))
    return matches, math.exp(-mean_d)

def verify_crater_matches(craters0, craters1, matches):
    if len(matches) < 3: return _empty_matches()
    pts0 = np.array([[craters0[m["i0"]].x, craters0[m["i0"]].y] for m in matches], dtype=np.float32)
    pts1 = np.array([[craters1[m["i1"]].x, craters1[m["i1"]].y] for m in matches], dtype=np.float32)
    return ransac_affine(pts0, pts1)

def run_cnsfm_branch(img0, img1) -> dict:
    try:
        print("\n[CNSFM] Running...")
        c0, c1 = detect_craters(img0), detect_craters(img1)
        m, s = match_crater_graphs(c0, c1)
        res = verify_crater_matches(c0, c1, m)
        res['score'] = (0.35 * s + 0.65 * res.get('score', 0)) if res.get('matches', 0) > 0 else 0
        return res
    except Exception as e:
        print(f"[CNSFM] Error: {e}")
        return _empty_matches()

# =============================================================================
# GEOMETRIC VERIFICATION SECTION
# =============================================================================

def ransac_homography(pts0, pts1, threshold=5.0, max_iters=5000, img_ref=None, img_tgt=None) -> dict:
    if len(pts0) < 4: return _empty_matches()

    # Stage 1: USAC_MAGSAC for initial candidate inlier set
    H, mask = cv2.findHomography(pts0, pts1, cv2.USAC_MAGSAC, threshold, maxIters=max_iters)
    if mask is None: mask = np.zeros(len(pts0), dtype=bool)
    else: mask = mask.ravel().astype(bool)

    inliers = int(np.sum(mask))
    rmse = float("inf")
    dof = 0
    p1_active = pts1[mask] if inliers > 0 else pts1
    uniformity_info = algorithms.compute_spatial_uniformity_score(p1_active)

    # Check for geometric degeneracy
    if inliers >= 4 and H is not None:
        det = float(np.linalg.det(H[:2, :2]))
        if det <= 0.1 or det >= 10.0:
            return ransac_affine(pts0, pts1, threshold=threshold, max_iters=max_iters)

        p0_in, p1_in = pts0[mask].copy(), pts1[mask].copy()

        # Sub-pixel keypoint refinement on inliers using cornerSubPix if images available
        if img_ref is not None:
            p1_in = algorithms.refine_subpixel_corners(img_ref, p1_in, win_size=(5, 5))
        if img_tgt is not None:
            p0_in = algorithms.refine_subpixel_corners(img_tgt, p0_in, win_size=(5, 5))

        # Stage 2: Sub-pixel Levenberg-Marquardt Huber M-estimator homography optimization
        sub_res = algorithms.refine_homography_subpixel(
            p0_in, p1_in, threshold=min(threshold / 2.0, 1.45), loss="huber"
        )
        if sub_res.get("H") is not None and sub_res.get("inliers", 0) >= 4:
            H = sub_res["H"]
            rmse = float(sub_res["rmse"])
            dof = int(sub_res["dof"])
            sub_mask = sub_res["mask"]
            in_indices = np.where(mask)[0]
            final_mask = np.zeros(len(pts0), dtype=bool)
            final_mask[in_indices[sub_mask]] = True
            mask = final_mask
            inliers = int(np.sum(mask))
            p0_sub_in = p0_in[sub_mask]
            p1_sub_in = p1_in[sub_mask]
        else:
            p0, p1 = pts0[mask], pts1[mask]
            proj = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2)
            rmse = float(np.sqrt(np.mean(np.linalg.norm(proj - p1, axis=1) ** 2)))
            dof = max(0, int(2 * inliers - 8))
            p0_sub_in = p0_in
            p1_sub_in = p1_in

        uniformity_info = algorithms.compute_spatial_uniformity_score(pts1[mask] if inliers > 0 else pts1)

    pts0_out = pts0.copy().astype(np.float64)
    pts1_out = pts1.copy().astype(np.float64)
    if inliers > 0 and 'p0_sub_in' in locals() and len(p0_sub_in) == inliers:
        pts0_out[mask] = p0_sub_in
        pts1_out[mask] = p1_sub_in

    return {
        "matches": len(pts0),
        "inliers": inliers,
        "inlier_ratio": inliers / len(pts0) if len(pts0) > 0 else 0.0,
        "rmse": rmse,
        "points0": pts0_out,
        "points1": pts1_out,
        "subpixel_pts0": p0_sub_in if ('p0_sub_in' in locals() and inliers > 0) else np.empty((0, 2)),
        "subpixel_pts1": p1_sub_in if ('p1_sub_in' in locals() and inliers > 0) else np.empty((0, 2)),
        "mask": mask,
        "H": H,
        "score": inliers / len(pts0) if len(pts0) > 0 else 0.0,
        "dof": dof,
        "uniformity_score": uniformity_info.get("uniformity_score_pct", 0.0),
        "uniformity_details": uniformity_info,
    }


def ransac_affine(pts0, pts1, threshold=6.0, max_iters=5000) -> dict:
    if len(pts0) < 3: return _empty_matches()
    M, mask = cv2.estimateAffinePartial2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=max_iters)
    if mask is None: mask = np.zeros(len(pts0), dtype=bool)
    else: mask = mask.ravel().astype(bool)
    
    inliers = int(np.sum(mask))
    rmse = float("inf")
    if inliers >= 3 and M is not None:
        p0, p1 = pts0[mask], pts1[mask]
        proj = cv2.transform(p0.reshape(-1, 1, 2), M).reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.linalg.norm(proj - p1, axis=1)**2)))
        
    # Return as H for consistency
    H = np.eye(3)
    if M is not None: H[:2, :] = M
    
    return {"matches": len(pts0), "inliers": inliers, "inlier_ratio": inliers/len(pts0),
            "rmse": rmse, "points0": pts0, "points1": pts1, "mask": mask, "H": H, "score": 0.0}

# =============================================================================
# NON-RIGID REGISTRATION SECTION
# =============================================================================

def fit_tps_warp(src_img, src_pts, tgt_pts, out_shape, smoothing=4.0) -> np.ndarray:
    print("[PIPELINE] TPS Warping...")
    # select spatially distributed points
    idx = algorithms.select_spatially_distributed_points(src_pts)
    if len(idx) < 3:
        # Fallback to affine or return original if too few points
        return cv2.resize(src_img, (out_shape[1], out_shape[0]))
    
    s_pts = src_pts[idx]
    t_pts = tgt_pts[idx]
    return algorithms.tps_warp(src_img, s_pts, t_pts, out_shape, smoothing)

# =============================================================================
# IMAGE FUSION SECTION
# =============================================================================

def fuse_images(ohrc_aligned, tmc_ref, levels=4) -> np.ndarray:
    print("[PIPELINE] Image Fusion...")
    o_u8 = normalize_percentile(ohrc_aligned)
    t_u8 = normalize_percentile(tmc_ref)
    return algorithms.laplacian_pyramid_fusion(o_u8, t_u8, levels)

# =============================================================================
# METRICS SECTION
# =============================================================================

def compute_all_metrics(img_ref, img_registered, gsd=5.0, rmse_px=None, pts_inliers=None) -> dict:
    print("[PIPELINE] Computing metrics...")
    r_u8 = normalize_percentile(img_ref)
    t_u8 = normalize_percentile(img_registered)

    ssim_val, _ = algorithms.compute_ssim(r_u8, t_u8)
    
    # Feature SSIM on Phase Congruency maps
    pc_ref = algorithms.phase_congruency_2d(r_u8)
    pc_reg = algorithms.phase_congruency_2d(t_u8)
    f_ssim_val, _ = algorithms.compute_ssim((pc_ref*255).astype(np.uint8), (pc_reg*255).astype(np.uint8))
    
    mi_res = algorithms.compute_mutual_information(r_u8, t_u8)
    ngf_dist = algorithms.ngf_distance(r_u8, t_u8)
    
    ret = {
        "SSIM": float(ssim_val),
        "Feature_SSIM": float(f_ssim_val),
        "NMI": float(mi_res["NMI"]),
        "NGF_Distance": float(ngf_dist)
    }
    if rmse_px is not None and np.isfinite(rmse_px):
        ret["Reproj_RMSE_px"] = float(rmse_px)
        ret["Ground_RMSE_m"] = float(rmse_px * gsd)
    if pts_inliers is not None and len(pts_inliers) > 0:
        u = algorithms.compute_spatial_uniformity_score(pts_inliers, img_ref.shape[:2])
        ret["Spatial_Uniformity_pct"] = float(u.get("uniformity_score_pct", 0.0))
    return ret

# =============================================================================
# VISUALIZATION SECTION
# =============================================================================

def draw_matches(img0, img1, pts0, pts1, mask, title='') -> np.ndarray:
    h1, w1 = img0.shape
    h2, w2 = img1.shape
    h = max(h1, h2)
    out = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1, :] = cv2.cvtColor(normalize_percentile(img0), cv2.COLOR_GRAY2BGR)
    out[:h2, w1:w1+w2, :] = cv2.cvtColor(normalize_percentile(img1), cv2.COLOR_GRAY2BGR)
    
    inlier_indices = np.where(mask)[0] if mask is not None else np.array([], dtype=int)
    
    # Subsample inliers evenly across the vertical axis for clean, professional visualization
    if len(inlier_indices) > 40:
        y_coords = pts0[inlier_indices, 1]
        sort_order = np.argsort(y_coords)
        step = max(1, len(sort_order) // 40)
        display_indices = inlier_indices[sort_order[::step]]
    else:
        display_indices = inlier_indices
        
    for idx in display_indices:
        pt1 = (int(round(pts0[idx][0])), int(round(pts0[idx][1])))
        pt2 = (int(round(pts1[idx][0])) + w1, int(round(pts1[idx][1])))
        cv2.line(out, pt1, pt2, (0, 230, 115), 1, cv2.LINE_AA)
        cv2.circle(out, pt1, 2, (0, 230, 115), -1, cv2.LINE_AA)
        cv2.circle(out, pt2, 2, (0, 230, 115), -1, cv2.LINE_AA)
        
    if title:
        # Sleek dark banner at the top
        banner_h = 36
        overlay = out[:banner_h, :].copy()
        cv2.rectangle(out, (0, 0), (w1 + w2, banner_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.3, out[:banner_h, :], 0.7, 0, out[:banner_h, :])
        cv2.putText(out, title, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 229, 255), 1, cv2.LINE_AA)
    return out

def draw_checkerboard(img_ref, img_registered, tiles=8, tile_size=None) -> np.ndarray:
    """Creates a seamless 2D interlocking checkerboard comparison overlay.
    
    Guarantees strictly square tiles (tile_size x tile_size px) across arbitrary
    swath aspect ratios (including elongated pushbroom strips). Alternates
    reference and registered imagery in both horizontal and vertical directions
    so sub-pixel crater rim continuity can be rigorously inspected along both axes.
    Preserves uint8 radiometric calibration without re-stretching contrast.
    """
    if img_ref.ndim == 3:
        r_u8 = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY) if img_ref.shape[2] == 3 else img_ref[:, :, 0]
    else:
        r_u8 = img_ref
    if r_u8.dtype != np.uint8 or r_u8.max() <= 1:
        r_u8 = normalize_percentile(r_u8)

    if img_registered.ndim == 3:
        reg_u8 = cv2.cvtColor(img_registered, cv2.COLOR_RGB2GRAY) if img_registered.shape[2] == 3 else img_registered[:, :, 0]
    else:
        reg_u8 = img_registered
    if reg_u8.dtype != np.uint8 or reg_u8.max() <= 1:
        reg_u8 = normalize_percentile(reg_u8)

    h, w = r_u8.shape[:2]
    if reg_u8.shape[:2] != (h, w):
        reg_u8 = cv2.resize(reg_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    if tile_size is None or tile_size <= 0:
        base_dim = min(h, w) if min(h, w) > 0 else max(h, w)
        tile_size = max(8, int(round(base_dim / max(1, tiles))))

    out = np.zeros((h, w), dtype=np.uint8)
    n_rows = int(np.ceil(h / tile_size))
    n_cols = int(np.ceil(w / tile_size))

    for i in range(n_rows):
        r1 = i * tile_size
        r2 = min(h, (i + 1) * tile_size)
        for j in range(n_cols):
            c1 = j * tile_size
            c2 = min(w, (j + 1) * tile_size)
            if (i + j) % 2 == 0:
                out[r1:r2, c1:c2] = r_u8[r1:r2, c1:c2]
            else:
                out[r1:r2, c1:c2] = reg_u8[r1:r2, c1:c2]
    return out

def draw_false_color(img_ref, img_registered) -> np.ndarray:
    if img_ref.dtype == np.uint8 and img_ref.max() > 1:
        r_u8 = img_ref
    else:
        r_u8 = normalize_percentile(img_ref)
    if img_registered.dtype == np.uint8 and img_registered.max() > 1:
        reg_u8 = img_registered
    else:
        reg_u8 = normalize_percentile(img_registered)
    h, w = r_u8.shape[:2]
    if reg_u8.shape[:2] != (h, w):
        reg_u8 = cv2.resize(reg_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 0] = r_u8 # B
    out[:, :, 1] = reg_u8 # G
    out[:, :, 2] = r_u8 # R
    return out

def save_results(results, output_dir) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if 'matches_img' in results:
        cv2.imwrite(os.path.join(output_dir, "matches.png"), results['matches_img'])
    if 'checkerboard' in results:
        cv2.imwrite(os.path.join(output_dir, "checkerboard.png"), results['checkerboard'])
    if 'false_color' in results:
        cv2.imwrite(os.path.join(output_dir, "false_color.png"), results['false_color'])
    if 'registered' in results:
        cv2.imwrite(os.path.join(output_dir, "registered.png"), results['registered'])
    if 'fused' in results:
        cv2.imwrite(os.path.join(output_dir, "fused.png"), results['fused'])
        
    with open(os.path.join(output_dir, "metrics.csv"), "w") as f:
        f.write("Metric,Value\n")
        if 'metrics' in results:
            for k, v in results['metrics'].items():
                f.write(f"{k},{v}\n")

# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def run_pipeline(ohrc_xml, tmc_xml, matchers=None, preprocessing='phase_congruency', do_tps=True, do_fusion=True) -> dict:
    if matchers is None: matchers = ['loftr', 'roma', 'lightglue']
    
    print("[PIPELINE] 1. Parsing PDS4 metadata...")
    o_meta = parse_pds4_metadata(ohrc_xml)
    t_meta = parse_pds4_metadata(tmc_xml)
    
    print("[PIPELINE] 2. Computing footprints & ROIs...")
    geom = compute_footprint(o_meta, t_meta)
    tr = geom['tmc_bbox']
    oroi = geom['ohrc_bbox']
    step = max(1, int(round(CURRENT_GSD['TMC-2'] / CURRENT_GSD['OHRC'])))

    has_raw = (
        o_meta.get('img_path') and os.path.exists(o_meta['img_path']) and
        t_meta.get('img_path') and os.path.exists(t_meta['img_path'])
    )

    if has_raw:
        tmc_crop = load_pds4_window(t_meta['img_path'], tr[0], tr[1], tr[2], tr[3], t_meta['samples'], t_meta['dtype'], offset=t_meta.get('offset',0))
        ohrc_raw = load_pds4_decimated(o_meta['img_path'], oroi[0], oroi[1], oroi[2], oroi[3], o_meta['samples'], o_meta['dtype'], step, offset=o_meta.get('offset',0))
        print("[PIPELINE] 3. GSD Normalization via Scale-Space Pyramid...")
        target_w = max(32, int(round(ohrc_raw.shape[1] * o_meta['gsd'] / t_meta['gsd'])))
        target_h = max(32, int(round(ohrc_raw.shape[0] * step * o_meta['gsd'] / t_meta['gsd'])))
        factor = float(t_meta['gsd']) / float(o_meta['gsd'] * step)
        ohrc_crop = scale_space_downsample(ohrc_raw, scale_factor=factor)
        if ohrc_crop.shape[1] != target_w or ohrc_crop.shape[0] != target_h:
            ohrc_crop = cv2.resize(ohrc_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        p_tmc = os.path.join(ROOT, "results_demo", "tmc_crop.png")
        p_ohrc = os.path.join(ROOT, "results_demo", "ohrc_crop.png")
        if os.path.exists(p_tmc) and os.path.exists(p_ohrc):
            tmc_crop = cv2.imread(p_tmc, cv2.IMREAD_GRAYSCALE)
            ohrc_crop = cv2.imread(p_ohrc, cv2.IMREAD_GRAYSCALE)
            ohrc_raw = ohrc_crop.copy()
        else:
            raise FileNotFoundError(f"PDS4 raw images ({o_meta.get('img_path')}, {t_meta.get('img_path')}) and results_demo crops are missing.")
    print(f"[PIPELINE] Physical scale: TMC crop {tmc_crop.shape}, OHRC normalized {ohrc_crop.shape}")
    
    print("[PIPELINE] 4. Coarse alignment...")
    ohrc_prep_coarse, tmc_prep_coarse = prepare_images(ohrc_crop, tmc_crop, mode=preprocessing, src_gsd=t_meta['gsd'], tgt_gsd=t_meta['gsd'])
    coarse_res = coarse_to_fine_align(ohrc_prep_coarse, tmc_prep_coarse, src_gsd=t_meta['gsd'], tgt_gsd=t_meta['gsd'])
    
    # Apply coarse alignment to ohrc onto tmc canvas
    ht, wt = tmc_crop.shape[:2]
    ohrc_coarse_aligned = cv2.warpAffine(ohrc_crop.astype(np.float32), coarse_res['transform_matrix'], (wt, ht), flags=cv2.INTER_LINEAR).astype(np.uint8)
    
    # Restrict matching to valid overlapping bounding box to eliminate false border matches
    valid_mask = ohrc_coarse_aligned > 0
    y_idx, x_idx = np.where(valid_mask)
    if len(y_idx) > 0 and len(x_idx) > 0:
        y0, y1 = max(0, int(y_idx.min())), min(ht, int(y_idx.max()) + 1)
        x0, x1 = max(0, int(x_idx.min())), min(wt, int(x_idx.max()) + 1)
    else:
        y0, y1, x0, x1 = 0, ht, 0, wt
        
    print(f"[PIPELINE] Active overlap region: y=[{y0}:{y1}], x=[{x0}:{x1}] (shape: {y1-y0}x{x1-x0})")
    roi_ohrc = ohrc_coarse_aligned[y0:y1, x0:x1]
    roi_tmc = tmc_crop[y0:y1, x0:x1]
    
    print("[PIPELINE] 5. Preprocessing for matching...")
    ohrc_prep, tmc_prep = prepare_images(roi_ohrc, roi_tmc, mode=preprocessing)
    
    print("[PIPELINE] 6. Feature Matching...")
    results_dict = {}
    if 'loftr' in matchers: results_dict['loftr'] = run_loftr_branch(ohrc_prep, tmc_prep); gc.collect()
    if 'roma' in matchers: results_dict['roma'] = run_roma_branch(ohrc_prep, tmc_prep); gc.collect()
    if 'lightglue' in matchers: results_dict['lightglue'] = run_lightglue_branch(ohrc_prep, tmc_prep); gc.collect()
    if 'sift' in matchers: results_dict['sift'] = run_sift_branch(ohrc_prep, tmc_prep); gc.collect()
    if 'orb' in matchers: results_dict['orb'] = run_orb_branch(ohrc_prep, tmc_prep); gc.collect()
    if 'cnsfm' in matchers: results_dict['cnsfm'] = run_cnsfm_branch(ohrc_prep, tmc_prep); gc.collect()
    
    # Shift matched coordinates back to full TMC canvas frame
    for k in results_dict:
        if results_dict[k]['matches'] > 0:
            results_dict[k]['points0'] = results_dict[k]['points0'] + np.array([[x0, y0]])
            results_dict[k]['points1'] = results_dict[k]['points1'] + np.array([[x0, y0]])
    
    print("[PIPELINE] 7. Aggregating Results...")
    best_matcher = max(results_dict.keys(), key=lambda k: results_dict[k]['score'])
    best_res = results_dict[best_matcher]
    print(f"[PIPELINE] Best Matcher: {best_matcher} (score: {best_res['score']:.4f}, inliers: {best_res['inliers']}/{best_res['matches']} - {best_res['inlier_ratio']*100:.1f}%)")
    
    out_res = {'best_matcher': best_matcher, 'matches': best_res}
    
    if best_res['inliers'] < 4:
        print("[PIPELINE] Too few matches for registration.")
        return out_res
    
    pts0 = best_res['points0'][best_res['mask']]
    pts1 = best_res['points1'][best_res['mask']]
    
    if do_tps:
        print("[PIPELINE] 8a. TPS Non-Rigid Warping — Pass 1...")
        ohrc_pass1 = fit_tps_warp(ohrc_coarse_aligned, pts0, pts1, tmc_crop.shape)

        # ── Iterative Refinement Pass 2 ─────────────────────────────────────
        # After Pass 1 the images are closely aligned; residual displacement is
        # small so the dense matcher can find far more consistent inliers.
        # (Technique: NASA LROC/ISRO ChaSTE 2-pass iterative registration)
        print("[PIPELINE] 8b. Iterative Refinement — Pass 2 re-matching...")
        try:
            # Only redo RoMa (best matcher) on the pass-1-aligned image
            if 'roma' in matchers:
                valid_mask2 = ohrc_pass1 > 0
                yi2, xi2 = np.where(valid_mask2)
                if len(yi2) > 0 and len(xi2) > 0:
                    y0_2 = max(0, int(yi2.min()))
                    y1_2 = min(ht, int(yi2.max()) + 1)
                    x0_2 = max(0, int(xi2.min()))
                    x1_2 = min(wt, int(xi2.max()) + 1)
                    roi2_o = ohrc_pass1[y0_2:y1_2, x0_2:x1_2]
                    roi2_t = tmc_crop[y0_2:y1_2, x0_2:x1_2]
                    p2_o, p2_t = prepare_images(roi2_o, roi2_t, mode=preprocessing)
                    res2 = run_roma_branch(p2_o, p2_t)
                    gc.collect()
                    if res2['inliers'] >= 4 and res2['score'] >= best_res['score']:
                        res2['points0'] = res2['points0'] + np.array([[x0_2, y0_2]])
                        res2['points1'] = res2['points1'] + np.array([[x0_2, y0_2]])
                        pts0 = res2['points0'][res2['mask']]
                        pts1 = res2['points1'][res2['mask']]
                        best_res = res2
                        print(f"[PIPELINE] Pass 2 improved: {res2['inliers']} inliers "
                              f"({res2['inlier_ratio']*100:.1f}%), RMSE {res2['rmse']:.2f}px")
                    else:
                        print(f"[PIPELINE] Pass 2 not better ({res2['inliers']} inliers); keeping Pass 1 matches.")
        except Exception as e:
            print(f"[PIPELINE] Pass 2 refinement skipped: {e}")
        # ── End Iterative Refinement ─────────────────────────────────────────

        print("[PIPELINE] 8c. TPS Non-Rigid Warping — Pass 2 (final)...")
        ohrc_final = fit_tps_warp(ohrc_pass1, pts0, pts1, tmc_crop.shape)
        out_res['tps_warped'] = ohrc_final
    else:
        print("[PIPELINE] 8. Homography Warping...")
        ohrc_final = cv2.warpPerspective(ohrc_coarse_aligned, best_res['H'], (tmc_crop.shape[1], tmc_crop.shape[0]))

    # ISRO Deliverable: Compute planar projective Homography via cv2.findHomography with RANSAC
    H_robust, h_mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
    if H_robust is not None:
        best_res['H'] = H_robust
    warped_perspective = cv2.warpPerspective(ohrc_coarse_aligned, best_res['H'], (tmc_crop.shape[1], tmc_crop.shape[0]))
    warped_matched = algorithms.match_histograms(warped_perspective, tmc_crop, mask=(warped_perspective > 0))
    out_res['registered'] = warped_matched
    out_res['H'] = best_res['H']
    out_res['subpixel_pts0'] = pts0
    out_res['subpixel_pts1'] = pts1
    out_res['roma_pts0'] = pts0
    out_res['roma_pts1'] = pts1
    
    if do_fusion:
        print("[PIPELINE] 9. Laplacian Pyramid Fusion...")
        fused = fuse_images(ohrc_final, tmc_crop)
        out_res['fused'] = fused
        
    print("[PIPELINE] 10. Compute Metrics...")
    metrics = compute_all_metrics(tmc_crop, ohrc_final)
    out_res['metrics'] = metrics
    print(metrics)
    
    print("[PIPELINE] 11. Visualizations...")
    out_res['matches_img'] = draw_matches(ohrc_prep, tmc_prep, best_res['points0'], best_res['points1'], best_res['mask'], f"{best_matcher} Matches")
    out_res['checkerboard'] = draw_checkerboard(tmc_crop, ohrc_final)
    out_res['false_color'] = draw_false_color(tmc_crop, ohrc_final)
    out_res['ohrc_raw'] = ohrc_raw
    out_res['ohrc_crop'] = ohrc_crop
    out_res['tmc_crop'] = tmc_crop
    out_res['target_img'] = ohrc_crop
    out_res['reference_img'] = tmc_crop
    out_res['ohrc_meta'] = o_meta
    out_res['tmc_meta'] = t_meta

    return out_res

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PixelOrbit Lunar Registration Pipeline")
    parser.add_argument("--ohrc-xml", default=os.path.join(ROOT, "ch2_ohr_ncp_20231004T0406038822_d_img_d18.xml"))
    parser.add_argument("--tmc-xml", default=os.path.join(ROOT, "ch2_tmc_ncn_20250707T1853051045_d_img_d18.xml"))
    parser.add_argument("--matchers", default="loftr,roma,lightglue", help="Comma separated matchers: loftr,roma,lightglue,sift,orb,cnsfm")
    parser.add_argument("--preprocessing", choices=["phase_congruency", "ngf", "legacy"], default="phase_congruency")
    parser.add_argument("--no-tps", action="store_true", help="Skip TPS warp")
    parser.add_argument("--no-fusion", action="store_true", help="Skip fusion")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "results"))
    
    args = parser.parse_args()
    matchers_list = args.matchers.split(',')
    
    results = run_pipeline(
        args.ohrc_xml, 
        args.tmc_xml, 
        matchers=matchers_list, 
        preprocessing=args.preprocessing, 
        do_tps=not args.no_tps, 
        do_fusion=not args.no_fusion
    )
    
    save_results(results, args.output_dir)
    print(f"[PIPELINE] Done. Results saved to {args.output_dir}")
