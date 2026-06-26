# -*- coding: utf-8 -*-
"""
test_vectorization_bit_level.py — 向量化函数位级一致性测试

原则 P1 (数值保真红线) 的单元测试层门禁:
    每个向量化函数必须与逐点 (pointwise) 参考实现位级一致 (diff == 0).
    任何向量化重构若引入数值差异, 本测试将红灯.

覆盖函数:
    - _hex8_shape_grad_batch  vs  hex8_shape_grad (逐点)
    - _build_B_matrix_batch   vs  _build_B_matrix (逐点)
    - _gauss_hex8_batch       vs  _gauss_pointwise (逐点)
"""
import numpy as np
import pytest

from fcm.elements import (
    HEX8_NODES, GAUSS_2X2X2, hex8_shape_grad, _build_B_matrix,
    elastic_matrix_D,
)
from fcm.assembly import (
    _hex8_shape_grad_batch, _build_B_matrix_batch,
    _gauss_hex8_batch, _gauss_pointwise,
)


# ---------------------------------------------------------------------------
# 测试用例生成器
# ---------------------------------------------------------------------------
def _random_local_pts(n: int = 20, seed: int = 42) -> np.ndarray:
    """生成 n 个随机参考坐标 (ξ,η,ζ) ∈ [-1,1]³."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 3))


def _random_dN_dx(n: int = 15, npe: int = 8, seed: int = 7) -> np.ndarray:
    """生成 n 组随机 dN/dx (npe 节点)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n, 3, npe))


def _random_coords(seed: int = 99) -> np.ndarray:
    """生成一个随机但 Jacobian 正定的 Hex8 单元坐标 (8, 3)."""
    rng = np.random.default_rng(seed)
    # 基础立方体 + 小扰动, 保证 detJ > 0
    base = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    return base + rng.uniform(-0.05, 0.05, size=base.shape)


# ---------------------------------------------------------------------------
# _hex8_shape_grad_batch vs hex8_shape_grad
# ---------------------------------------------------------------------------
class TestHex8ShapeGradBatch:
    """向量化形函数梯度 vs 逐点实现 — 必须位级一致."""

    def test_at_gauss_points(self):
        """在 2×2×2 Gauss 点上对比."""
        pts = GAUSS_2X2X2.points
        dN_batch = _hex8_shape_grad_batch(pts)  # (8, 3, 8)
        for g, pt in enumerate(pts):
            dN_ref = hex8_shape_grad(pt[0], pt[1], pt[2])  # (3, 8)
            assert np.array_equal(dN_batch[g], dN_ref), (
                f"Gauss point {g}: shape grad diff = "
                f"{np.max(np.abs(dN_batch[g] - dN_ref)):.2e}"
            )

    def test_at_random_points(self):
        """在 20 个随机点上对比 — 位级一致 (diff == 0)."""
        pts = _random_local_pts(20)
        dN_batch = _hex8_shape_grad_batch(pts)
        for g, pt in enumerate(pts):
            dN_ref = hex8_shape_grad(pt[0], pt[1], pt[2])
            assert np.array_equal(dN_batch[g], dN_ref), (
                f"Random point {g}: shape grad diff = "
                f"{np.max(np.abs(dN_batch[g] - dN_ref)):.2e}"
            )

    def test_at_origin(self):
        """在原点 (0,0,0) 对比."""
        pt = np.array([[0.0, 0.0, 0.0]])
        dN_batch = _hex8_shape_grad_batch(pt)[0]
        dN_ref = hex8_shape_grad(0.0, 0.0, 0.0)
        assert np.array_equal(dN_batch, dN_ref)


# ---------------------------------------------------------------------------
# _build_B_matrix_batch vs _build_B_matrix
# ---------------------------------------------------------------------------
class TestBuildBMatrixBatch:
    """向量化 B 矩阵 vs 逐点实现 — 必须位级一致."""

    @pytest.mark.parametrize('npe', [8, 20, 32])
    def test_random_dN_dx(self, npe):
        """对 Hex8/20/32 节点数, 随机 dN/dx 对比 — 位级一致."""
        dN_dx = _random_dN_dx(15, npe)
        B_batch = _build_B_matrix_batch(dN_dx)  # (15, 6, 3*npe)
        for g in range(dN_dx.shape[0]):
            B_ref = _build_B_matrix(dN_dx[g], npe)  # (6, 3*npe)
            assert np.array_equal(B_batch[g], B_ref), (
                f"npe={npe} point {g}: B-matrix diff = "
                f"{np.max(np.abs(B_batch[g] - B_ref)):.2e}"
            )

    def test_shape(self):
        """输出形状正确."""
        dN_dx = _random_dN_dx(5, 8)
        B = _build_B_matrix_batch(dN_dx)
        assert B.shape == (5, 6, 24)


# ---------------------------------------------------------------------------
# _gauss_hex8_batch vs _gauss_pointwise
# ---------------------------------------------------------------------------
class TestGaussHex8Batch:
    """向量化 Gauss 积分 vs 逐点实现 — 必须数值一致 (diff < 1e-10)."""

    def test_unit_cube_ke(self):
        """单位立方体单元刚度矩阵对比."""
        coords = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64)
        D = elastic_matrix_D(2e5, 0.3)
        pts = GAUSS_2X2X2.points
        wts = GAUSS_2X2X2.weights

        ke_batch = np.zeros((24, 24), dtype=np.float64)
        _gauss_hex8_batch(pts, wts, coords, D, ke_batch)

        ke_ref = np.zeros((24, 24), dtype=np.float64)
        _gauss_pointwise(pts, wts, coords, 8, hex8_shape_grad, D, ke_ref)

        diff = np.max(np.abs(ke_batch - ke_ref))
        assert diff < 1e-10, (
            f"Unit cube ke: batch vs pointwise diff = {diff:.2e} (must < 1e-10)"
        )

    def test_perturbed_cube_ke(self):
        """扰动立方体单元刚度矩阵对比."""
        coords = _random_coords()
        D = elastic_matrix_D(2e5, 0.3)
        pts = GAUSS_2X2X2.points
        wts = GAUSS_2X2X2.weights

        ke_batch = np.zeros((24, 24), dtype=np.float64)
        _gauss_hex8_batch(pts, wts, coords, D, ke_batch)

        ke_ref = np.zeros((24, 24), dtype=np.float64)
        _gauss_pointwise(pts, wts, coords, 8, hex8_shape_grad, D, ke_ref)

        diff = np.max(np.abs(ke_batch - ke_ref))
        assert diff < 1e-10, (
            f"Perturbed cube ke: batch vs pointwise diff = {diff:.2e} (must < 1e-10)"
        )

    def test_ke_symmetry(self):
        """单元刚度矩阵必须对称."""
        coords = _random_coords()
        D = elastic_matrix_D(2e5, 0.3)
        pts = GAUSS_2X2X2.points
        wts = GAUSS_2X2X2.weights
        ke = np.zeros((24, 24), dtype=np.float64)
        _gauss_hex8_batch(pts, wts, coords, D, ke)
        assert np.allclose(ke, ke.T, atol=1e-12), (
            f"ke not symmetric: max asymmetry = {np.max(np.abs(ke - ke.T)):.2e}"
        )
