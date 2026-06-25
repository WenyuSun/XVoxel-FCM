# -*- coding: utf-8 -*-
"""pytest configuration and shared fixtures for XVoxel-FCM tests."""
import os
import sys

# Ensure the repo root is on sys.path so tests can do absolute imports.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pytest
import numpy as np

# ---------------------------------------------------------------------------
#  Primitives fixtures
# ---------------------------------------------------------------------------

from src.primitives import Cube, CylinderZ, CylinderY, RoundCorner2D


@pytest.fixture
def unit_cube():
    """A 1×1×1 cube centered at origin."""
    return Cube(cx=0, cy=0, cz=0, sx=1.0, sy=1.0, sz=1.0)


@pytest.fixture
def small_cylinder():
    """A small Z-axis cylinder."""
    return CylinderZ(cx=0, cy=0, r=0.5, zmin=-1.0, zmax=1.0)


# ---------------------------------------------------------------------------
#  XVoxel model fixtures
# ---------------------------------------------------------------------------

from src.xvoxel import XVoxelModel


@pytest.fixture
def xvoxel_10():
    """10×10×10 XVoxel model, unit size, centered at origin."""
    return XVoxelModel(10, 10, 10, 1.0, 1.0, 1.0, origin=(-0.5, -0.5, -0.5))


@pytest.fixture
def xvoxel_10_with_body(xvoxel_10):
    """10³ XVoxel with a solid cube body added."""
    body = Cube(cx=0, cy=0, cz=0, sx=0.8, sy=0.8, sz=0.8)
    xvoxel_10.add_feature(body, nature=1, name="body")
    return xvoxel_10


@pytest.fixture
def xvoxel_10_with_hole(xvoxel_10_with_body):
    """10³ XVoxel with solid body + cylindrical hole."""
    hole = CylinderZ(cx=0, cy=0, r=0.15, zmin=-0.5, zmax=0.5)
    xvoxel_10_with_body.add_feature(hole, nature=-1, name="hole")
    return xvoxel_10_with_body


# ---------------------------------------------------------------------------
#  L-shape model fixture (lazy — only built when requested)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def lshape_model():
    """Session-scoped L-shape model from examples/fig7_lshape."""
    # Load via importlib to avoid sys.path issues
    import importlib.util
    _path = os.path.join(_repo_root, "examples", "fig7_lshape.py")
    spec = importlib.util.spec_from_file_location("fig7_lshape", _path)
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    xv, fids = _mod.build_lshape_model()
    return xv, fids, _mod


# ---------------------------------------------------------------------------
#  FEM solver fixtures
# ---------------------------------------------------------------------------

from src.fem_base import HexMesh, FEMSolver


@pytest.fixture
def hex_mesh_4():
    """4×4×4 unit-cube hex mesh."""
    return HexMesh(4, 4, 4, 1.0, 1.0, 1.0)


@pytest.fixture
def fem_solver_nu0(hex_mesh_4):
    """FEM solver on 4³ mesh with ν=0 (no Poisson)."""
    return FEMSolver(hex_mesh_4, E=2e11, nu=0.0)


@pytest.fixture
def fem_solver_nu03(hex_mesh_4):
    """FEM solver on 4³ mesh with ν=0.3."""
    return FEMSolver(hex_mesh_4, E=2e11, nu=0.3)


from src.fem_xvoxel import XVoxelFEMSolver


@pytest.fixture
def xvoxel_solver_hex8(xvoxel_10_with_body):
    """XVoxelFEMSolver with Hex8 elements."""
    return XVoxelFEMSolver(xvoxel_10_with_body, E=2e11, nu=0.3, alpha=1e-8,
                            element_order=1)
