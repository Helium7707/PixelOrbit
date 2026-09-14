"""
Comprehensive test suite for Phase 1: algorithms.py
Tests all 8 core algorithmic modules.
"""

import numpy as np
import cv2
import sys

from algorithms import (
    phase_correlation_subpx,
    log_polar_fourier_mellin,
    compute_ngf,
    ngf_distance,
    ngf_similarity_map,
    phase_congruency_2d,
    tps_warp,
    select_spatially_distributed_points,
    laplacian_pyramid_fusion,
    compute_ssim,
    compute_mutual_information,
    refine_subpixel_corners,
    refine_subpixel_phase_correlation,
    refine_homography_subpixel,
    spatial_grid_bucketing,
    compute_spatial_uniformity_score,
)

def create_synthetic_lunar_patch(size=256, seed=42):
    np.random.seed(seed)
    # Background texture
    img = np.random.normal(120, 15, (size, size)).astype(np.float32)
    # Add synthetic crater (rim + floor)
    cx, cy, r = size // 2, size // 2, size // 5
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    # Crater floor (dark)
    img[dist < r] -= 40
    # Crater rim (bright sunlit side)
    rim_mask = (dist >= r - 3) & (dist <= r + 3) & (x < cx)
    img[rim_mask] += 60
    # Crater shadow
    shadow_mask = (dist < r) & (x > cx - r // 2)
    img[shadow_mask] -= 30
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 1.0)


def test_phase_correlation():
    print("Testing 1: Phase Correlation (sub-pixel)...")
    base = create_synthetic_lunar_patch(256)
    shift_x, shift_y = 7.35, -4.60
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(base, M, (256, 256), borderMode=cv2.BORDER_REFLECT)
    
    (dx, dy), score = phase_correlation_subpx(base, shifted)
    print(f"  Expected: ({shift_x}, {shift_y}), Recovered: ({dx:.2f}, {dy:.2f}), Peak: {score:.3f}")
    assert abs(dx - shift_x) < 0.5, f"dx error too large: {dx} vs {shift_x}"
    assert abs(dy - shift_y) < 0.5, f"dy error too large: {dy} vs {shift_y}"
    assert score > 0.3, f"Score too low: {score}"
    print("  ✅ Phase Correlation passed.")


def test_fourier_mellin():
    print("Testing 2: Log-Polar Fourier-Mellin (Rot + Scale + Trans)...")
    base = create_synthetic_lunar_patch(256)
    angle = 5.0  # degrees
    scale = 1.05
    M = cv2.getRotationMatrix2D((128, 128), angle, scale)
    M[0, 2] += 4.0
    M[1, 2] += -3.0
    transformed = cv2.warpAffine(base, M, (256, 256), borderMode=cv2.BORDER_REFLECT)
    
    res = log_polar_fourier_mellin(base, transformed)
    rec_scale = res["scale"]
    rec_rot = res["rotation_deg"]
    print(f"  Target scale: {scale:.2f}, Recovered: {rec_scale:.2f}")
    print(f"  Target rot: {angle:.1f}°, Recovered: {rec_rot:.1f}°")
    print(f"  Confidence: {res['confidence']:.3f}")
    assert abs(rec_scale - scale) < 0.1, f"Scale recovery error: {rec_scale} vs {scale}"
    assert abs(rec_rot - angle) < 3.0 or abs((rec_rot - angle) % 360) < 3.0, f"Rot error: {rec_rot} vs {angle}"
    print("  ✅ Fourier-Mellin passed.")


def test_ngf():
    print("Testing 3: Normalized Gradient Fields...")
    base = create_synthetic_lunar_patch(256)
    # Test identical image -> distance close to 0
    dist_self = ngf_distance(base, base)
    print(f"  Self-distance: {dist_self:.6f} (expected ~0.0)")
    assert dist_self < 1e-4, f"Self NGF distance should be ~0, got {dist_self}"
    
    # Inverted contrast image (simulating shadow reversal)
    inverted = 255 - base
    dist_inv = ngf_distance(base, inverted)
    print(f"  Inverted-contrast distance: {dist_inv:.6f} (expected ~0.0 due to sign invariance)")
    assert dist_inv < 1e-4, f"NGF should be sign-invariant! Got {dist_inv}"
    
    # Shifted image -> higher distance
    M = np.float32([[1, 0, 20], [0, 1, 20]])
    shifted = cv2.warpAffine(base, M, (256, 256))
    dist_shifted = ngf_distance(base, shifted)
    print(f"  Shifted distance: {dist_shifted:.6f} (expected > 0.05)")
    assert dist_shifted > 0.05, f"Shifted NGF distance should be high, got {dist_shifted}"
    print("  ✅ NGF passed.")


def test_phase_congruency():
    print("Testing 4: Phase Congruency (Kovesi)...")
    base = create_synthetic_lunar_patch(128)
    pc = phase_congruency_2d(base, nscale=3, norient=4, min_wavelength=3.0)
    print(f"  PC shape: {pc.shape}, min: {pc.min():.3f}, max: {pc.max():.3f}, mean: {pc.mean():.3f}")
    assert pc.shape == (128, 128)
    assert 0.0 <= pc.min() and pc.max() <= 1.0
    assert pc.max() > 0.2, "Phase congruency failed to detect edge features"
    print("  ✅ Phase Congruency passed.")


def test_tps_warp():
    print("Testing 5: Thin Plate Spline (TPS) Warping...")
    base = create_synthetic_lunar_patch(128)
    
    # Grid of control points
    src_pts = np.array([
        [20, 20], [108, 20],
        [64, 64],
        [20, 108], [108, 108]
    ], dtype=np.float32)
    
    # Mild non-rigid perturbation
    tgt_pts = src_pts + np.array([
        [0, 0], [2, -1],
        [-3, 2],
        [1, 2], [0, 0]
    ], dtype=np.float32)
    
    warped = tps_warp(base, src_pts, tgt_pts, (128, 128), smoothing=5.0)
    assert warped.shape == (128, 128)
    assert warped.dtype == base.dtype
    
    # Test spatial point selection
    large_pts = np.random.uniform(0, 128, (100, 2))
    selected_idx = select_spatially_distributed_points(large_pts, n_bins=4, max_per_bin=3)
    assert len(selected_idx) <= 4 * 4 * 3
    assert len(selected_idx) > 5
    print(f"  Spatial selection filtered 100 points to {len(selected_idx)} well-spread points")
    print("  ✅ TPS Warping passed.")


def test_laplacian_fusion():
    print("Testing 6: Laplacian Pyramid Fusion...")
    base = create_synthetic_lunar_patch(256)
    # Low-pass version (like coarse TMC)
    tmc_sim = cv2.GaussianBlur(base, (15, 15), 4.0)
    # High-detail version (like OHRC)
    ohrc_sim = base
    
    fused = laplacian_pyramid_fusion(ohrc_sim, tmc_sim, levels=4)
    assert fused.shape == (256, 256)
    assert fused.dtype == np.uint8
    # Fused image should have higher gradient variance than the blurred TMC
    var_tmc = np.var(cv2.Sobel(tmc_sim, cv2.CV_64F, 1, 1))
    var_fused = np.var(cv2.Sobel(fused, cv2.CV_64F, 1, 1))
    print(f"  Detail gradient variance: TMC={var_tmc:.2f} -> Fused={var_fused:.2f}")
    assert var_fused > var_tmc, "Fusion failed to inject high-frequency details"
    print("  ✅ Laplacian Fusion passed.")


def test_ssim_and_mi():
    print("Testing 7 & 8: SSIM and Mutual Information...")
    base = create_synthetic_lunar_patch(128)
    
    # SSIM self test
    mssim_self, _ = compute_ssim(base, base)
    print(f"  SSIM self: {mssim_self:.4f} (expected 1.0)")
    assert mssim_self > 0.999
    
    # SSIM degraded
    noisy = np.clip(base.astype(np.int16) + np.random.normal(0, 25, base.shape), 0, 255).astype(np.uint8)
    mssim_noisy, _ = compute_ssim(base, noisy)
    print(f"  SSIM noisy: {mssim_noisy:.4f} (expected < 0.9)")
    assert mssim_noisy < mssim_self
    
    # Mutual Information self test
    mi_self = compute_mutual_information(base, base)
    print(f"  Self NMI: {mi_self['NMI']:.4f} (expected ~2.0)")
    assert mi_self['NMI'] > 1.8
    
    # Independent random noise
    random_noise = np.random.randint(0, 256, base.shape, dtype=np.uint8)
    mi_noise = compute_mutual_information(base, random_noise)
    print(f"  Uncorrelated NMI: {mi_noise['NMI']:.4f} (expected ~1.0)")
    assert mi_noise['NMI'] < 1.3
    print("  ✅ SSIM & MI passed.")


def test_subpixel_refinement():
    print("Testing 9: Sub-Pixel Keypoint & Homography Refinement (RMSE < 1.0 px)...")
    base = create_synthetic_lunar_patch(256)
    
    # 1. Test cornerSubPix wrapper
    pts = np.array([
        [128.0, 77.0],
        [100.0, 100.0],
        [80.0, 128.0],
        [100.0, 156.0]
    ], dtype=np.float32)
    refined_pts = refine_subpixel_corners(base, pts, win_size=(5, 5), max_shift=3.5)
    assert refined_pts.shape == pts.shape
    shifts = np.linalg.norm(refined_pts - pts, axis=1)
    assert np.all(shifts <= 3.5), f"Max shift exceeded: {shifts}"
    print(f"  CornerSubPix shifts: min={shifts.min():.3f}px, max={shifts.max():.3f}px")

    # 2. Test phase correlation subpixel patch refinement
    dx_true, dy_true = 0.45, -0.35
    M = np.float32([[1, 0, dx_true], [0, 1, dy_true]])
    shifted = cv2.warpAffine(base, M, (256, 256), borderMode=cv2.BORDER_REFLECT)
    pts_ref = np.array([[128.0, 128.0]], dtype=np.float32)
    pts_tgt = np.array([[128.0, 128.0]], dtype=np.float32)
    _, p_tgt_ref = refine_subpixel_phase_correlation(base, shifted, pts_ref, pts_tgt, patch_radius=24)
    rec_dx = p_tgt_ref[0, 0] - 128.0
    rec_dy = p_tgt_ref[0, 1] - 128.0
    print(f"  Phase correlation subpixel shift: true=({dx_true}, {dy_true}), recovered=({rec_dx:.2f}, {rec_dy:.2f})")
    assert abs(rec_dx - dx_true) < 0.35
    assert abs(rec_dy - dy_true) < 0.35

    # 3. Test Levenberg-Marquardt Huber homography refinement for RMSE < 1.0 px
    np.random.seed(42)
    pts0 = np.random.uniform(40, 216, (40, 2)).astype(np.float64)
    H_true = np.array([
        [1.02, -0.01, 12.5],
        [0.015, 0.98, -8.2],
        [1e-5, -2e-5, 1.0]
    ], dtype=np.float64)
    pts0_h = np.hstack([pts0, np.ones((len(pts0), 1))])
    proj = (H_true @ pts0_h.T).T
    pts1 = proj[:, :2] / proj[:, 2:3]
    # Add realistic sub-pixel Gaussian noise (sigma = 0.35 px)
    noise = np.random.normal(0, 0.35, pts1.shape)
    pts1 += noise
    # Add 2 outliers
    pts1[-1] += np.array([25.0, -30.0])
    pts1[-2] += np.array([-40.0, 35.0])

    res = refine_homography_subpixel(pts0, pts1, threshold=2.5, loss="huber")
    print(f"  Refined Homography Inliers: {res['inliers']}/40, RMSE: {res['rmse']:.4f} px, DOF: {res['dof']}")
    assert res["inliers"] >= 36, f"Expected >= 36 inliers, got {res['inliers']}"
    assert res["rmse"] < 1.0, f"Reprojection RMSE must be < 1.0 px, got {res['rmse']}"
    assert res["dof"] >= 60, f"Expected high DOF, got {res['dof']}"
    print("  ✅ Sub-Pixel Refinement passed (RMSE < 1.0 px).")


def test_spatial_uniformity():
    print("Testing 10: Spatial Grid Bucketing & Uniformity Evaluation...")
    img_shape = (1000, 1000)
    
    # 1. Test uniformly distributed points
    np.random.seed(123)
    grid_coords = []
    for r in range(10):
        for c in range(10):
            if (r + c) % 2 == 0 and len(grid_coords) < 40:
                grid_coords.append([c * 100 + 50 + np.random.uniform(-20, 20),
                                    r * 100 + 50 + np.random.uniform(-20, 20)])
    uniform_pts = np.array(grid_coords, dtype=np.float32)
    
    score_res = compute_spatial_uniformity_score(uniform_pts, img_shape=img_shape, grid_size=(10, 10))
    print(f"  Uniform test score: {score_res['uniformity_score_pct']}%, occupied: {score_res['occupied_cells']}/100, entropy: {score_res['shannon_entropy']}/{score_res['max_entropy']}")
    assert score_res["uniformity_score_pct"] >= 85.0, f"Expected score >= 85%, got {score_res['uniformity_score_pct']}"
    assert score_res["dispersion_ratio"] >= 0.90, f"Expected dispersion ratio >= 0.90, got {score_res['dispersion_ratio']}"
    assert score_res["occupied_cells"] >= 35, f"Expected >= 35 occupied cells, got {score_res['occupied_cells']}"

    # 2. Test clustered points (all inside cell (0, 0))
    clustered_pts = np.random.uniform(5, 50, (40, 2)).astype(np.float32)
    clustered_res = compute_spatial_uniformity_score(clustered_pts, img_shape=img_shape, grid_size=(10, 10))
    print(f"  Clustered test score: {clustered_res['uniformity_score_pct']}%, occupied: {clustered_res['occupied_cells']}/100")
    assert clustered_res["uniformity_score_pct"] < 50.0, f"Clustered score should be low, got {clustered_res['uniformity_score_pct']}"

    # 3. Test spatial_grid_bucketing filter
    dense_cluster = np.random.uniform(10, 80, (60, 2))
    dispersed = np.random.uniform(100, 900, (40, 2))
    all_pts = np.vstack([dense_cluster, dispersed])
    
    p0_sel, p1_sel, sel_idx = spatial_grid_bucketing(all_pts, all_pts, img_shape=img_shape, grid_size=(10, 10), max_per_cell=3, min_dist=10.0)
    print(f"  Spatial bucketing: {len(all_pts)} input points filtered to {len(sel_idx)} uniformly distributed points")
    assert len(sel_idx) < len(all_pts)
    assert len(sel_idx) >= 30
    
    cell_0_count = sum(1 for p in p1_sel if p[0] < 100 and p[1] < 100)
    assert cell_0_count <= 3, f"Max per cell exceeded: {cell_0_count} > 3"
    print("  ✅ Spatial Uniformity & Bucketing passed (score > 85%, clumping eliminated).")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING UNIT TESTS FOR algorithms.py (PHASE 1 + ISRO UPDATES)")
    print("=" * 60)
    test_phase_correlation()
    test_fourier_mellin()
    test_ngf()
    test_phase_congruency()
    test_tps_warp()
    test_laplacian_fusion()
    test_ssim_and_mi()
    test_subpixel_refinement()
    test_spatial_uniformity()
    print("=" * 60)
    print("ALL ALGORITHMS VERIFIED AND FUNCTIONING PERFECTLY!")
    print("=" * 60)
