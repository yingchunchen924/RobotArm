"""标定拟合纯函数测试（合成数据，无需摄像头/ROS）。"""

import importlib.util
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(modname, relpath):
    path = os.path.join(_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


cal = _load("calibrate_pixel_map", "scripts/board/calibrate_pixel_map.py")


def _make_samples_from_affine(A, pts, noise=0.0):
    """用已知仿射 A 从像素点生成样本，可叠加噪声。"""
    samples = []
    for i, (cx, cy) in enumerate(pts):
        x = A[0][0] * cx + A[0][1] * cy + A[0][2]
        y = A[1][0] * cx + A[1][1] * cy + A[1][2]
        if noise:
            # 确定性的小扰动（不用随机，保证可复现）
            x += noise * ((i % 3) - 1)
            y += noise * (((i + 1) % 3) - 1)
        samples.append(cal.Sample(cx, cy, x, y))
    return samples


# 一个已知的仿射：x = 0.0003*cx - 0.00001*cy - 0.05 ; y = 0.00001*cx - 0.0004*cy + 0.42
TRUE_A = [[0.0003, -0.00001, -0.05], [0.00001, -0.0004, 0.42]]
GRID = [(100, 80), (320, 80), (540, 80), (100, 240), (320, 240), (540, 240), (100, 400), (540, 400)]


def test_fit_affine_recovers_known_matrix_no_noise():
    samples = _make_samples_from_affine(TRUE_A, GRID)
    A = cal.fit_affine(samples)
    assert np.allclose(A, np.array(TRUE_A), atol=1e-9)


def test_fit_affine_residual_zero_no_noise():
    samples = _make_samples_from_affine(TRUE_A, GRID)
    A = cal.fit_affine(samples)
    stats = cal.residuals(lambda cx, cy: cal.apply_affine(A, cx, cy), samples)
    assert stats["rms_m"] < 1e-9
    assert stats["max_m"] < 1e-9
    assert stats["n"] == len(samples)


def test_fit_affine_robust_to_small_noise():
    samples = _make_samples_from_affine(TRUE_A, GRID, noise=0.001)  # 1mm 量级
    A = cal.fit_affine(samples)
    stats = cal.residuals(lambda cx, cy: cal.apply_affine(A, cx, cy), samples)
    # 残差应在噪声量级（< 2mm）
    assert stats["rms_m"] < 0.002


def test_fit_affine_needs_three_points():
    with pytest.raises(ValueError):
        cal.fit_affine([cal.Sample(0, 0, 0, 0), cal.Sample(1, 1, 1, 1)])


def test_apply_affine_matches_formula():
    A = np.array(TRUE_A)
    x, y = cal.apply_affine(A, 320, 240)
    assert x == pytest.approx(0.0003 * 320 - 0.00001 * 240 - 0.05)
    assert y == pytest.approx(0.00001 * 320 - 0.0004 * 240 + 0.42)


def test_fit_homography_recovers_affine_case():
    # 仿射是单应的特例：用仿射生成的点，单应也应零残差
    samples = _make_samples_from_affine(TRUE_A, GRID)
    H = cal.fit_homography(samples)
    stats = cal.residuals(lambda cx, cy: cal.apply_homography(H, cx, cy), samples)
    assert stats["rms_m"] < 1e-6


def test_homography_needs_four_points():
    with pytest.raises(ValueError):
        cal.fit_homography([cal.Sample(i, i, i, i) for i in range(3)])


def test_leave_one_out_small_for_clean_data():
    samples = _make_samples_from_affine(TRUE_A, GRID)
    loo = cal.leave_one_out_rms(
        samples, cal.fit_affine, lambda m: (lambda cx, cy: cal.apply_affine(m, cx, cy))
    )
    assert loo < 1e-6


def test_parse_samples_text_json_and_plain():
    text = """
    # comment
    {"cx": 100, "cy": 80, "tar_x": -0.02, "tar_y": 0.39}
    320 240 0.0 0.31
    540, 400, 0.06, 0.25
    """
    samples = cal.parse_samples_text(text)
    assert len(samples) == 3
    assert samples[0].cx == 100 and samples[0].tar_y == 0.39
    assert samples[1].cx == 320 and samples[1].tar_x == 0.0
    assert samples[2].cx == 540 and samples[2].tar_y == 0.25


def test_parse_pose():
    assert cal.parse_pose("91,135,0,0,90,30") == [91, 135, 0, 0, 90, 30]
    assert cal.parse_pose("91 135 0 0 90 30") == [91, 135, 0, 0, 90, 30]
    with pytest.raises(ValueError):
        cal.parse_pose("91,135,0")


def test_to_yaml_block_contains_affine_and_model():
    samples = _make_samples_from_affine(TRUE_A, GRID)
    A = cal.fit_affine(samples)
    stats = cal.residuals(lambda cx, cy: cal.apply_affine(A, cx, cy), samples)
    block = cal.to_yaml_block(A, stats, "2026-06-24")
    assert "model: affine" in block
    assert "affine:" in block
    assert "calibrated_at" in block
