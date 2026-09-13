"""
better_branches.py
pip install romatch                                    # RoMa -- try this first
pip install git+https://github.com/cvg/LightGlue.git   # LightGlue -- lighter/faster fallback

WHY THESE, INSTEAD OF LoFTR + Hough-circle CNSFM
-------------------------------------------------
Two independent benchmarks point the same direction:

1. A Chandrayaan-2-specific study (Makharia et al. 2025, arXiv 2509.04775)
   testing SIFT/ASIFT/AKAZE/RIFT2/SuperGlue on cross-sensor lunar pairs
   found SuperGlue-family matchers winning by a wide margin, and on their
   hardest cross-modality pair, EVERY classical method (including RIFT2,
   which is specifically built for cross-modal radiometric differences)
   failed outright -- only the SuperGlue-class matcher registered it.

2. A 2026 SAR-optical review (Zhang et al., arXiv 2502.01002) built a
   10,850-pair benchmark spanning 0.16m-10m resolution and ran 16 SOTA
   matchers. Two results matter directly for us:
     - LoFTR specifically had the fewest matches and the most mismatches
       "in scenes with significant image content variation" -- our exact
       situation (crisp OHRC vs genuinely blurry TMC).
     - At their most extreme resolution gap (sub-meter vs 10m -- still
       smaller than our ~30x OHRC/TMC gap), nearly every algorithm failed
       outright, and RoMa was the ONLY one that produced correct, usable
       matches.

So: RoMa first (best evidence at extreme resolution gaps), LightGlue as a
lighter/faster fallback (still clearly better than LoFTR per both studies
above). Kept the GSD-matching + CLAHE preprocessing from before -- that
part already matches published best practice, nothing to change there.

HONEST EXPECTATION: our OHRC/TMC gap (~0.2m vs ~6.13m, ~30x) is LARGER
than the worst case tested in either benchmark above, where even the best
method (RoMa) only succeeded on some of the hardest pairs, with sparse,
unevenly-distributed matches. A modest number of matches that survive
BOTH RANSAC and an independent geometry cross-check (see the earlier
run_correspondence.py additions) is a legitimate, defensible outcome here
-- not a failure. Chasing "hundreds of dense matches" at this resolution
gap is chasing something no published method reliably achieves yet.
"""

import os
import tempfile
import numpy as np
import cv2
import torch
import xml.etree.ElementTree as ET


# =============================================================================
# OPTION A (try first): RoMa -- dense matcher, best published evidence at
# extreme resolution gaps specifically.
# =============================================================================

def run_roma_branch(img0, img1, device=None, num_samples=5000):
    """
    Drop-in replacement for run_loftr_branch(img0, img1) from the hybrid
    script -- same grayscale-uint8-array-in, metrics-dict-out contract.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from romatch import roma_outdoor
    roma_model = roma_outdoor(device=dev)

    # Some installed versions of romatch default ConvRefiner.use_custom_corr
    # to True in the "outdoor" preset, which tries to `import local_corr` --
    # an optional compiled CUDA extension that's NOT part of a plain
    # `pip install romatch` (it needs `pip install romatch[fused-local-corr]`
    # and a CUDA build toolchain). Force it off everywhere so we always use
    # the plain PyTorch correlation path instead of crashing on that import.
    for module in roma_model.modules():
        if hasattr(module, "use_custom_corr"):
            module.use_custom_corr = False

    # RoMa's DINOv2 backbone expects RGB and reads from disk paths -- our
    # crops are already-normalized single-channel uint8 arrays, so
    # replicate to 3 channels and round-trip through a temp PNG.
    def save_tmp(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(f.name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return f.name

    h0, w0 = img0.shape
    h1, w1 = img1.shape
    path0, path1 = save_tmp(img0), save_tmp(img1)

    try:
        warp, certainty = roma_model.match(path0, path1, device=dev)
        matches, certainty = roma_model.sample(warp, certainty, num=num_samples)
        kpts0, kpts1 = roma_model.to_pixel_coordinates(matches, h0, w0, h1, w1)
        pts0 = kpts0.cpu().numpy()
        pts1 = kpts1.cpu().numpy()
    finally:
        os.unlink(path0)
        os.unlink(path1)

    print(f"[RoMa] dense-sampled matches: {len(pts0)}")
    return _ransac_and_metrics(pts0, pts1, "RoMa")


# =============================================================================
# OPTION B (lighter/faster fallback): LightGlue -- still clearly ahead of
# LoFTR per both benchmarks above, much cheaper than RoMa.
# =============================================================================

def run_lightglue_branch(img0, img1, device=None, max_keypoints=2048):
    """Same drop-in contract, using DISK (permissive license) + LightGlue."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from lightglue import LightGlue, DISK
    from lightglue.utils import rbd

    def to_tensor(img):
        rgb = np.stack([img] * 3, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).to(dev)

    extractor = DISK(max_num_keypoints=max_keypoints).eval().to(dev)
    matcher = LightGlue(features="disk").eval().to(dev)

    feats0 = extractor.extract(to_tensor(img0))
    feats1 = extractor.extract(to_tensor(img1))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"]
    idx = matches01["matches"]
    pts0 = kpts0[idx[:, 0]].cpu().numpy()
    pts1 = kpts1[idx[:, 1]].cpu().numpy()

    print(f"[LightGlue] raw matches: {len(pts0)}")
    return _ransac_and_metrics(pts0, pts1, "LightGlue")


def _ransac_and_metrics(pts0, pts1, label):
    if len(pts0) < 4:
        print(f"[{label}] fewer than 4 matches -- cannot run RANSAC")
        return {"matches": len(pts0), "inliers": 0, "inlier_ratio": 0.0,
                "rmse": float("inf"), "points0": pts0, "points1": pts1,
                "mask": np.zeros(len(pts0), dtype=bool), "H": None}

    H, mask = cv2.findHomography(pts0, pts1, cv2.USAC_MAGSAC, 6.0, maxIters=5000)
    mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(pts0), dtype=bool)
    n_inliers = int(mask.sum())

    rmse = float("inf")
    if n_inliers >= 4 and H is not None:
        in0, in1 = pts0[mask], pts1[mask]
        proj = cv2.perspectiveTransform(in0.reshape(-1, 1, 2), H).reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.linalg.norm(proj - in1, axis=1) ** 2)))

    ratio = n_inliers / len(pts0)
    print(f"[{label}] inliers: {n_inliers}/{len(pts0)}  ratio: {ratio:.4f}  RMSE: {rmse:.3f}px")

    return {"matches": len(pts0), "inliers": n_inliers, "inlier_ratio": ratio,
            "rmse": rmse, "points0": pts0, "points1": pts1, "mask": mask, "H": H}


# =============================================================================
# The PUBLISHED preprocessing recipe for OHRC-class registration (Makharia
# et al.), to use INSTEAD of the Hough-circle CNSFM branch. This enhances
# crater/edge structure as an input to the matcher above, rather than
# trying to detect and graph-match craters as a separate subsystem.
# =============================================================================

def enhance_craters_edges(img, clahe_clip=2.0, clahe_grid=8, dilate_ksize=3):
    """
    CLAHE -> inversion -> morphological dilation, applied to BOTH images
    before matching. Per the published recipe: CLAHE boosts local contrast
    without blowing out noise in flat regions, inversion pushes shadowed
    crater floors and bright rims further apart, and dilation thickens rim
    edges into stronger, more repeatable keypoint structure.
    """
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    out = clahe.apply(img)
    out = 255 - out
    kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
    out = cv2.dilate(out, kernel, iterations=1)
    return out


# =============================================================================
# Wiring into the existing hybrid script:
#
#   from better_branches import run_roma_branch, enhance_craters_edges
#
#   # try RoMa directly on the existing GSD-matched, CLAHE'd crops first:
#   result = run_roma_branch(ohrc_prep, tmc_prep)
#
#   # if matches are still thin, try the edge-enhanced variant as well and
#   # compare -- don't assume either preprocessing wins without checking:
#   result_edge = run_roma_branch(enhance_craters_edges(ohrc_prep),
#                                  enhance_craters_edges(tmc_prep))
#
#   # drop run_cnsfm_branch() entirely for now -- it's Stage 7 "research
#   # extension" territory (per the original project staging), not part of
#   # the required baseline, and the Hough-circle detector it's built on
#   # isn't reliable enough on TMC's blur to be trustworthy signal yet.
# =============================================================================

# PDS4 Element_Array/data_type -> numpy dtype string
_PDS4_DTYPES = {
    "UnsignedByte": "uint8",
    "SignedByte": "int8",
    "UnsignedLSB2": "<u2",
    "UnsignedMSB2": ">u2",
    "SignedLSB2": "<i2",
    "SignedMSB2": ">i2",
    "UnsignedLSB4": "<u4",
    "UnsignedMSB4": ">u4",
    "SignedLSB4": "<i4",
    "SignedMSB4": ">i4",
    "IEEE754LSBSingle": "<f4",
    "IEEE754MSBSingle": ">f4",
}

_PDS_NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}


def parse_pds4_label(xml_path):
    """
    Parse a Chandrayaan-2 (or any PDS4) .xml label and pull out what's
    needed to read the companion raw .img array: file name, byte offset,
    dtype, and (lines, samples) shape. The label and the .img file are
    expected to sit in the same directory.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    file_el = root.find(".//pds:File", _PDS_NS)
    file_name = file_el.find("pds:file_name", _PDS_NS).text.strip()

    array_el = root.find(".//pds:Array_2D_Image", _PDS_NS)
    offset = int(array_el.find("pds:offset", _PDS_NS).text)
    dtype_name = array_el.find("pds:Element_Array/pds:data_type", _PDS_NS).text.strip()
    if dtype_name not in _PDS4_DTYPES:
        raise ValueError(f"Unrecognized PDS4 data_type '{dtype_name}' in {xml_path}")
    np_dtype = _PDS4_DTYPES[dtype_name]

    dims = {}
    for axis in array_el.findall("pds:Axis_Array", _PDS_NS):
        name = axis.find("pds:axis_name", _PDS_NS).text.strip()
        elements = int(axis.find("pds:elements", _PDS_NS).text)
        seq = int(axis.find("pds:sequence_number", _PDS_NS).text)
        dims[seq] = (name, elements)
    lines = dims[1][1]     # sequence 1 = Line
    samples = dims[2][1]   # sequence 2 = Sample

    img_path = os.path.join(os.path.dirname(os.path.abspath(xml_path)), file_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Label references '{file_name}' but it isn't next to the .xml "
            f"at {img_path}. Keep the .img and .xml in the same folder.")

    return {"img_path": img_path, "dtype": np_dtype,
            "shape": (lines, samples), "offset": offset}


def load_pds4_image(xml_path, roi=None, downsample=None):
    """
    Memory-map a PDS4 .img via its .xml label and return an 8-bit
    grayscale numpy array ready for CLAHE/matching.

    roi: optional (row_start, row_end, col_start, col_end) pixel window,
         applied BEFORE materializing into RAM -- use this for the huge
         TMC/OHRC full-strip products so you don't try to load gigabytes
         at once.
    downsample: optional integer stride (e.g. 4 keeps every 4th pixel),
         applied after roi, before the array is copied into RAM.
    """
    meta = parse_pds4_label(xml_path)
    lines, samples = meta["shape"]

    mm = np.memmap(meta["img_path"], dtype=meta["dtype"], mode="r",
                    offset=meta["offset"], shape=(lines, samples))

    if roi is not None:
        r0, r1, c0, c1 = roi
        view = mm[r0:r1, c0:c1]
    else:
        view = mm[:, :]

    if downsample and downsample > 1:
        view = view[::downsample, ::downsample]

    arr = np.array(view)  # materialize the (cropped/downsampled) slice only

    if arr.dtype != np.uint8:
        # Robust contrast stretch to 8-bit (1st-99th percentile), rather
        # than a naive min-max, so a few hot/dead pixels don't wash out
        # the whole dynamic range.
        arr_f = arr.astype(np.float32)
        lo, hi = np.percentile(arr_f, (1, 99))
        arr = np.clip((arr_f - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    return arr


def parse_pds4_corners(xml_path):
    """
    Pull the four corner (lat, lon) pairs out of a Chandrayaan-2 PDS4 label's
    isda:Geometry_Parameters block. Prefers Refined_Corner_Coordinates,
    falls back to System_Level_Coordinates if refined isn't present.
    """
    isda_ns = {"isda": "https://isda.issdc.gov.in/pds4/isda/v1"}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    corner_el = root.find(".//isda:Refined_Corner_Coordinates", isda_ns)
    if corner_el is None:
        corner_el = root.find(".//isda:System_Level_Coordinates", isda_ns)
    if corner_el is None:
        raise ValueError(f"No corner coordinates found in {xml_path}")

    def get(tag):
        return float(corner_el.find(f"isda:{tag}", isda_ns).text)

    return {
        "UL": (get("upper_left_latitude"), get("upper_left_longitude")),
        "UR": (get("upper_right_latitude"), get("upper_right_longitude")),
        "LL": (get("lower_left_latitude"), get("lower_left_longitude")),
        "LR": (get("lower_right_latitude"), get("lower_right_longitude")),
    }


def _bilinear_latlon(corners, u, v):
    """Forward map: normalized (u along Line, v along Sample) -> (lat, lon)."""
    ul, ur, ll, lr = corners["UL"], corners["UR"], corners["LL"], corners["LR"]
    w = [(1 - u) * (1 - v), (1 - u) * v, u * (1 - v), u * v]
    lat = w[0] * ul[0] + w[1] * ur[0] + w[2] * ll[0] + w[3] * lr[0]
    lon = w[0] * ul[1] + w[1] * ur[1] + w[2] * ll[1] + w[3] * lr[1]
    return lat, lon


def _bilinear_uv_for_latlon(corners, lat, lon, iters=6):
    """
    Inverse map: (lat, lon) -> normalized (u, v), via a decoupled linear
    initial guess (lat driven mostly by u, lon mostly by v -- true for a
    near-nadir pushbroom strip) refined by a few Newton steps against the
    exact bilinear forward map and its analytic Jacobian.

    This is an approximation, not rigorous photogrammetry -- it ignores
    terrain relief, orbit curvature, and any off-nadir viewing (e.g. OHRC's
    pitch in these labels is ~26.6 deg). Treat results as a starting point
    and pad generously with margin, not as ground truth.
    """
    ul, ur, ll, lr = corners["UL"], corners["UR"], corners["LL"], corners["LR"]

    mean_lat_u0 = (ul[0] + ur[0]) / 2.0
    mean_lat_u1 = (ll[0] + lr[0]) / 2.0
    slope = mean_lat_u1 - mean_lat_u0
    u = (lat - mean_lat_u0) / slope if slope else 0.5
    lon_v0 = (1 - u) * ul[1] + u * ll[1]
    lon_v1 = (1 - u) * ur[1] + u * lr[1]
    denom = lon_v0 - lon_v1
    v = (lon_v0 - lon) / denom if denom else 0.5

    for _ in range(iters):
        cur_lat, cur_lon = _bilinear_latlon(corners, u, v)
        d_lat = lat - cur_lat
        d_lon = lon - cur_lon

        dlat_du = (ll[0] - ul[0]) * (1 - v) + (lr[0] - ur[0]) * v
        dlat_dv = (ur[0] - ul[0]) * (1 - u) + (lr[0] - ll[0]) * u
        dlon_du = (ll[1] - ul[1]) * (1 - v) + (lr[1] - ur[1]) * v
        dlon_dv = (ur[1] - ul[1]) * (1 - u) + (lr[1] - ll[1]) * u

        det = dlat_du * dlon_dv - dlat_dv * dlon_du
        if abs(det) < 1e-12:
            break
        du = (d_lat * dlon_dv - d_lon * dlat_dv) / det
        dv = (dlat_du * d_lon - dlon_du * d_lat) / det
        u += du
        v += dv

    return u, v


def latlon_bounds_for_roi(xml_path, roi):
    """(lat_min, lat_max, lon_min, lon_max) covered by a pixel ROI, via the
    corner-based forward bilinear map."""
    meta = parse_pds4_label(xml_path)
    corners = parse_pds4_corners(xml_path)
    lines, samples = meta["shape"]
    r0, r1, c0, c1 = roi
    lats, lons = [], []
    for r in (r0, r1):
        for c in (c0, c1):
            u = r / (lines - 1)
            v = c / (samples - 1)
            lat, lon = _bilinear_latlon(corners, u, v)
            lats.append(lat)
            lons.append(lon)
    return min(lats), max(lats), min(lons), max(lons)


def estimate_matching_roi(target_xml, source_xml, source_roi, margin_frac=0.25):
    """
    Given a pixel ROI in the 'source' image, estimate the pixel ROI in
    'target' that covers the same ground footprint, using each label's
    corner lat/lon. Pads the result by margin_frac on every side since
    this is only an approximation (see _bilinear_uv_for_latlon caveats).
    """
    lat_min, lat_max, lon_min, lon_max = latlon_bounds_for_roi(source_xml, source_roi)
    target_meta = parse_pds4_label(target_xml)
    target_corners = parse_pds4_corners(target_xml)
    lines, samples = target_meta["shape"]

    rows, cols = [], []
    for lat in (lat_min, lat_max):
        for lon in (lon_min, lon_max):
            u, v = _bilinear_uv_for_latlon(target_corners, lat, lon)
            rows.append(u * (lines - 1))
            cols.append(v * (samples - 1))

    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    r_pad = (r1 - r0) * margin_frac + 1
    c_pad = (c1 - c0) * margin_frac + 1
    r0 = max(0, int(r0 - r_pad))
    r1 = min(lines, int(r1 + r_pad))
    c0 = max(0, int(c0 - c_pad))
    c1 = min(samples, int(c1 + c_pad))
    return (r0, r1, c0, c1)


def check_roi_overlap(ohrc_xml, ohrc_roi, tmc_xml, tmc_roi):
    """Print each ROI's estimated ground footprint and warn if they don't
    plausibly overlap, so a bad ROI pick is caught before wasting a RoMa run."""
    o_lat_min, o_lat_max, o_lon_min, o_lon_max = latlon_bounds_for_roi(ohrc_xml, ohrc_roi)
    t_lat_min, t_lat_max, t_lon_min, t_lon_max = latlon_bounds_for_roi(tmc_xml, tmc_roi)

    print(f"[geo-check] OHRC ROI ~ lat [{o_lat_min:.4f}, {o_lat_max:.4f}]  "
          f"lon [{o_lon_min:.4f}, {o_lon_max:.4f}]")
    print(f"[geo-check] TMC  ROI ~ lat [{t_lat_min:.4f}, {t_lat_max:.4f}]  "
          f"lon [{t_lon_min:.4f}, {t_lon_max:.4f}]")

    overlap = not (o_lat_max < t_lat_min or t_lat_max < o_lat_min or
                   o_lon_max < t_lon_min or t_lon_max < o_lon_min)
    if not overlap:
        print("[geo-check] WARNING: these two ROIs do not appear to cover "
              "the same ground area at all -- any matches RoMa finds are "
              "almost certainly spurious. Consider --tmc-roi auto (or "
              "--ohrc-roi auto) to derive one from the other.")
    return overlap


def run_demo(ohrc_prep, tmc_prep, save_path="match_overlay.png",
             tmc_pixel_size_m=6.13, display_height=900, show_outliers=False):
    """
    Runs the RoMa branch on two already-prepared (GSD-matched + CLAHE'd)
    grayscale uint8 arrays and saves the match overlay to save_path.

    This used to be bare code at module level, which is why ohrc_prep /
    tmc_prep showed up as "undefined": those names were never assigned
    anywhere in this file -- they were only ever meant to come from your
    existing hybrid script's GSD-matching + CLAHE step. Wrapping it in a
    function means importing this file no longer tries to execute that
    code (and fail on the missing names); you call run_demo(...) yourself
    once you actually have prepared images.

    Display fixes vs the earlier version:
      - Both panels are rescaled to the SAME display height so a tiny TMC
        crop next to a huge OHRC crop doesn't look absurd -- matching
        itself still runs on the original, native-resolution arrays; only
        the plotted coordinates are rescaled afterwards.
      - Only inlier correspondences are drawn by default (show_outliers=True
        brings back the faint rejected lines, but they add noise, not signal,
        for a presentation figure).
      - Lines are thin and points are small markers so a judge can actually
        see image content underneath, not just a mass of colored lines.
      - RMSE is also reported in meters (using tmc_pixel_size_m), which is
        a far more legible number to put in front of judges than raw pixels.
    """
    result = run_roma_branch(ohrc_prep, tmc_prep)

    rmse_m = result["rmse"] * tmc_pixel_size_m if np.isfinite(result["rmse"]) else float("inf")
    print(f"matches: {result['matches']}  inliers: {result['inliers']}  "
          f"ratio: {result['inlier_ratio']:.4f}  RMSE: {result['rmse']:.3f}px "
          f"(~{rmse_m:.1f} m)")

    import matplotlib
    matplotlib.use("Agg")  # headless-safe: never tries to open a GUI window
    import matplotlib.pyplot as plt

    def to_bgr_scaled(img, target_h):
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scale = target_h / bgr.shape[0]
        new_w = max(1, int(round(bgr.shape[1] * scale)))
        bgr = cv2.resize(bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)
        return bgr, scale

    ohrc_disp, ohrc_scale = to_bgr_scaled(ohrc_prep, display_height)
    tmc_disp, tmc_scale = to_bgr_scaled(tmc_prep, display_height)
    canvas = np.hstack([ohrc_disp, tmc_disp])
    offset = ohrc_disp.shape[1]

    mask = result["mask"]
    pts0, pts1 = result["points0"], result["points1"]
    idx_to_draw = np.where(mask)[0] if not show_outliers else np.arange(len(pts0))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(canvas[..., ::-1])
    for i in idx_to_draw:
        x0, y0 = pts0[i] * ohrc_scale
        x1, y1 = pts1[i] * tmc_scale
        keep = mask[i]
        color = "lime" if keep else "red"
        lw = 0.4 if keep else 0.2
        alpha = 0.6 if keep else 0.08
        ax.plot([x0, x1 + offset], [y0, y1], color=color, linewidth=lw, alpha=alpha)
        ax.plot(x0, y0, "o", color="lime", markersize=1.5, alpha=0.8)
        ax.plot(x1 + offset, y1, "o", color="lime", markersize=1.5, alpha=0.8)

    ax.set_title(f"{result['inliers']} inliers (of {result['matches']} sampled)  "
                 f"RMSE {result['rmse']:.2f}px (~{rmse_m:.1f} m)")
    ax.axis("off")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[run_demo] saved match overlay to {save_path}")

    return result


def _parse_roi(roi_str):
    if roi_str is None:
        return None
    parts = [int(p) for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("--ohrc-roi/--tmc-roi need 4 comma-separated ints: r0,r1,c0,c1")
    return tuple(parts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone demo: run RoMa matching on an OHRC/TMC PDS4 pair. "
                    "Pass the .xml LABEL files (not the .img files directly) -- "
                    "the label tells us the dtype, shape, and byte offset needed "
                    "to read the raw .img correctly.")
    parser.add_argument("ohrc_xml", help="Path to the OHRC PDS4 .xml label")
    parser.add_argument("tmc_xml", help="Path to the TMC PDS4 .xml label")
    parser.add_argument("--ohrc-roi", default=None,
                         help="r0,r1,c0,c1 pixel window into the OHRC strip "
                              "(strongly recommended -- full OHRC frames are "
                              "~100k x 12k pixels), or 'auto' to derive it "
                              "from --tmc-roi's ground footprint")
    parser.add_argument("--tmc-roi", default=None,
                         help="r0,r1,c0,c1 pixel window into the TMC strip, "
                              "or 'auto' to derive it from --ohrc-roi's "
                              "ground footprint (recommended: pick --ohrc-roi "
                              "manually around your feature of interest, then "
                              "use --tmc-roi auto)")
    parser.add_argument("--roi-margin", type=float, default=0.25,
                         help="Fractional padding applied to an auto-derived "
                              "ROI on every side (default 0.25 = 25%%)")
    parser.add_argument("--ohrc-downsample", type=int, default=1,
                         help="Stride to subsample the OHRC crop by after ROI crop")
    parser.add_argument("--tmc-downsample", type=int, default=1,
                         help="Stride to subsample the TMC crop by after ROI crop")
    parser.add_argument("--save-path", default="match_overlay.png",
                         help="Where to save the match overlay image "
                              "(default: match_overlay.png in the cwd)")
    parser.add_argument("--display-height", type=int, default=900,
                         help="Both panels are rescaled to this height (px) "
                              "for the figure only -- matching itself still "
                              "runs on the native-resolution crops")
    parser.add_argument("--show-outliers", action="store_true",
                         help="Also draw rejected (non-inlier) correspondences "
                              "faintly. Off by default -- they add noise, not "
                              "signal, in a presentation figure")
    parser.add_argument("--tmc-pixel-size-m", type=float, default=6.13,
                         help="TMC ground sample distance in meters/pixel, "
                              "used only to also report RMSE in meters "
                              "(default 6.13, per this project's TMC label)")
    args = parser.parse_args()

    if args.ohrc_roi == "auto" and args.tmc_roi == "auto":
        raise ValueError("Only one of --ohrc-roi / --tmc-roi can be 'auto' -- "
                          "the other needs a real pixel window to derive from.")

    ohrc_roi = _parse_roi(args.ohrc_roi) if args.ohrc_roi not in (None, "auto") else None
    tmc_roi = _parse_roi(args.tmc_roi) if args.tmc_roi not in (None, "auto") else None

    if args.tmc_roi == "auto":
        if ohrc_roi is None:
            raise ValueError("--tmc-roi auto requires a real --ohrc-roi to derive from.")
        tmc_roi = estimate_matching_roi(args.tmc_xml, args.ohrc_xml, ohrc_roi,
                                         margin_frac=args.roi_margin)
        print(f"[auto-roi] derived --tmc-roi {tmc_roi[0]},{tmc_roi[1]},"
              f"{tmc_roi[2]},{tmc_roi[3]} from the OHRC ROI's ground footprint")
    elif args.ohrc_roi == "auto":
        if tmc_roi is None:
            raise ValueError("--ohrc-roi auto requires a real --tmc-roi to derive from.")
        ohrc_roi = estimate_matching_roi(args.ohrc_xml, args.tmc_xml, tmc_roi,
                                          margin_frac=args.roi_margin)
        print(f"[auto-roi] derived --ohrc-roi {ohrc_roi[0]},{ohrc_roi[1]},"
              f"{ohrc_roi[2]},{ohrc_roi[3]} from the TMC ROI's ground footprint")

    if ohrc_roi is not None and tmc_roi is not None:
        check_roi_overlap(args.ohrc_xml, ohrc_roi, args.tmc_xml, tmc_roi)

    # ohrc_prep/tmc_prep now come from the actual PDS4 .img data, read via
    # the .xml label (dtype UnsignedByte / UnsignedLSB2 etc, not PNG/JPEG).
    ohrc_prep = load_pds4_image(args.ohrc_xml, roi=ohrc_roi,
                                 downsample=args.ohrc_downsample)
    tmc_prep = load_pds4_image(args.tmc_xml, roi=tmc_roi,
                                downsample=args.tmc_downsample)

    run_demo(ohrc_prep, tmc_prep, save_path=args.save_path,
             display_height=args.display_height, show_outliers=args.show_outliers,
             tmc_pixel_size_m=args.tmc_pixel_size_m)