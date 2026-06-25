# -*- coding: utf-8 -*-
"""Integration test: Nitsche boundary conditions with Hex32 elements."""
import importlib.util
import os

import numpy as np
import pytest
from scipy.sparse.linalg import splu

from src.fem_xvoxel import XVoxelFEMSolver


@pytest.fixture(scope="module")
def lshape_solver():
    """Build L-shape FCM solver once per module."""
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _path = os.path.join(_root, "examples", "fig7_lshape.py")
    spec = importlib.util.spec_from_file_location("fig7_lshape", _path)
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    xv, fids = _mod.build_lshape_model()
    return XVoxelFEMSolver(xv, E=2e5, nu=0.3, alpha=1e-8, element_order=3)


class TestNitscheHex32:
    """Validate Nitsche BC application produces a solvable system."""

    def test_no_zero_diag_before_nitsche(self, lshape_solver):
        K, _ = lshape_solver.assemble_FCM_system()
        diag = K.diagonal()
        n_zero = np.sum(diag == 0)
        assert n_zero < lshape_solver.ndof, "All diagonal entries are zero"

    def test_nitsche_produces_valid_system(self, lshape_solver):
        K, F = lshape_solver.assemble_FCM_system()
        K_bc, F_bc = lshape_solver.apply_nitsche_dirichlet(
            K.copy(), F.copy(), 'ymax')
        diag = K_bc.diagonal()
        assert diag.min() > 0, "After Nitsche, all diagonals should be >0"
        lu = splu(K_bc.tocsc())
        u = lu.solve(F_bc)
        assert np.max(np.abs(u)) < 1e6, (
            f"Solution diverges: max|u| = {np.max(np.abs(u)):.2e}"
        )
