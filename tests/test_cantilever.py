# -*- coding: utf-8 -*-
"""Tests for Hex8 / Hex32 element formulations — shape functions, stiffness."""
import numpy as np
import pytest

from src.fem_base import (
    HEX8_NODES, HEX32_NODES,
    hex8_shape_func, hex8_element_stiffness,
    hex32_shape_func, hex32_element_stiffness,
)


class TestHex8ShapeFunctions:
    """Validate Hex8 trilinear shape functions."""

    def test_partition_of_unity(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            xi = rng.uniform(-1, 1)
            eta = rng.uniform(-1, 1)
            zeta = rng.uniform(-1, 1)
            N = hex8_shape_func(xi, eta, zeta)
            assert abs(N.sum() - 1.0) < 1e-12

    def test_kronecker_delta(self):
        for i in range(8):
            xi, eta, zeta = HEX8_NODES[i]
            N = hex8_shape_func(xi, eta, zeta)
            assert abs(N[i] - 1.0) < 1e-12
            for j in range(8):
                if j != i:
                    assert abs(N[j]) < 1e-12


class TestHex8Stiffness:
    """Validate Hex8 stiffness matrix properties."""

    @pytest.fixture
    def unit_hex_coords(self):
        return np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64)

    def test_symmetry(self, unit_hex_coords):
        ke = hex8_element_stiffness(unit_hex_coords, E=2e5, nu=0.3)
        asym = np.max(np.abs(ke - ke.T))
        assert asym < 1e-10, f"Max asymmetry: {asym:.2e}"

    def test_positive_semidefinite(self, unit_hex_coords):
        ke = hex8_element_stiffness(unit_hex_coords, E=2e5, nu=0.3)
        eigvals = np.linalg.eigvalsh(ke)
        n_near_zero = np.sum(np.abs(eigvals) < 1e-8)
        assert n_near_zero >= 6, f"Expected >=6 zero eigvals, got {n_near_zero}"
        assert np.all(eigvals >= -1e-10)

    def test_cantilever_tip_deflection(self, unit_hex_coords):
        # Single Hex8 cantilever: 10x1x1 beam, tip force Fy=-100
        coords = np.array([
            [0, 0, 0], [10, 0, 0], [10, 1, 0], [0, 1, 0],
            [0, 0, 1], [10, 0, 1], [10, 1, 1], [0, 1, 1],
        ], dtype=np.float64)
        E, nu = 2e5, 0.3
        ke = hex8_element_stiffness(coords, E, nu)
        F = np.zeros(24)
        fixed = [0, 1, 2, 9, 10, 11, 12, 13, 14, 21, 22, 23]
        F[4] = -25; F[7] = -25; F[16] = -25; F[19] = -25
        for d in fixed:
            ke[d, :] = 0; ke[:, d] = 0; ke[d, d] = 1.0; F[d] = 0
        u = np.linalg.solve(ke, F)
        uy_tip = (u[4] + u[7] + u[16] + u[19]) / 4
        # Analytical: delta = PL^3/(3EI), I = bh^3/12 = 1/12
        I = 1.0 / 12
        delta_expected = 100 * 10**3 / (3 * E * I)
        ratio = abs(uy_tip) / delta_expected
        # Single linear Hex8 is stiff under bending due to shear locking
        assert 0 < ratio < 1.0, f"Tip deflection ratio {ratio:.4f} outside (0,1)"


class TestHex32ShapeFunctions:
    """Validate Hex32 quadratic shape functions."""

    @pytest.fixture
    def flat_32_coords(self):
        coords = np.zeros((32, 3))
        coords[0] = [0, 0, 0]; coords[1] = [1, 0, 0]
        coords[2] = [1, 1, 0]; coords[3] = [0, 1, 0]
        coords[4] = [0, 0, 1]; coords[5] = [1, 0, 1]
        coords[6] = [1, 1, 1]; coords[7] = [0, 1, 1]
        coords[8] = [1/3, 0, 0]; coords[9] = [2/3, 0, 0]
        coords[10] = [1, 1/3, 0]; coords[11] = [1, 2/3, 0]
        coords[12] = [1/3, 1, 0]; coords[13] = [2/3, 1, 0]
        coords[14] = [0, 1/3, 0]; coords[15] = [0, 2/3, 0]
        coords[16] = [1/3, 0, 1]; coords[17] = [2/3, 0, 1]
        coords[18] = [1, 1/3, 1]; coords[19] = [1, 2/3, 1]
        coords[20] = [1/3, 1, 1]; coords[21] = [2/3, 1, 1]
        coords[22] = [0, 1/3, 1]; coords[23] = [0, 2/3, 1]
        coords[24] = [0, 0, 1/3]; coords[25] = [0, 0, 2/3]
        coords[26] = [1, 0, 1/3]; coords[27] = [1, 0, 2/3]
        coords[28] = [1, 1, 1/3]; coords[29] = [1, 1, 2/3]
        coords[30] = [0, 1, 1/3]; coords[31] = [0, 1, 2/3]
        return coords

    def test_partition_of_unity(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            xi = rng.uniform(-1, 1)
            eta = rng.uniform(-1, 1)
            zeta = rng.uniform(-1, 1)
            N = hex32_shape_func(xi, eta, zeta)
            assert abs(N.sum() - 1.0) < 1e-10

    def test_kronecker_delta_corners(self):
        for i in range(8):
            xi, eta, zeta = HEX32_NODES[i]
            N = hex32_shape_func(xi, eta, zeta)
            assert abs(N[i] - 1.0) < 1e-10
            for j in range(32):
                if j != i:
                    assert abs(N[j]) < 1e-10

    def test_symmetry(self, flat_32_coords):
        ke = hex32_element_stiffness(flat_32_coords, E=2e5, nu=0.3)
        asym = np.max(np.abs(ke - ke.T))
        assert asym < 1e-8, f"Max asymmetry: {asym:.2e}"

    def test_positive_semidefinite(self, flat_32_coords):
        ke = hex32_element_stiffness(flat_32_coords, E=2e5, nu=0.3)
        eigvals = np.linalg.eigvalsh(ke)
        n_near_zero = np.sum(np.abs(eigvals) < 1e-6)
        assert n_near_zero >= 6, f"Expected >=6 zero eigvals, got {n_near_zero}"
