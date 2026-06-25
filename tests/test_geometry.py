# -*- coding: utf-8 -*-
"""Tests for geometric corner/edge voxel nature in L-shape model."""
import numpy as np
import pytest

from src.primitives import Cube, RoundCorner2D
from src.xvoxel import XVoxelModel


class TestLShapeVoxelNature:
    """Validate voxel classification for the L-shape corner geometry."""

    @pytest.fixture
    def xv(self):
        NX, NY, NZ = 15, 15, 3
        LX, LY, LZ = 15.0, 15.0, 3.0
        origin = (0.0, 0.0, -LZ / 2)
        xv = XVoxelModel(NX, NY, NZ, LX, LY, LZ, origin=origin)

        vert = Cube(cx=1.5, cy=7.5, cz=0.0, sx=3.0, sy=15.0, sz=3.0)
        xv.add_feature(vert, nature=1, name="vertical")

        horiz = Cube(cx=7.5, cy=1.5, cz=0.0, sx=15.0, sy=3.0, sz=3.0)
        xv.add_feature(horiz, nature=1, name="horizontal")

        corner = RoundCorner2D(cx=3.0, cy=3.0, r=6.0,
                                zmin=-1.5, zmax=1.5,
                                sign_x=+1, sign_y=+1)
        fid_corner = xv.add_feature(corner, nature=+1, name="fillet")
        return xv, fid_corner

    def test_initial_voxel_counts(self, xv):
        xv_m, _ = xv
        n_s = np.sum(xv_m.voxel_nature == 1)
        n_v = np.sum(xv_m.voxel_nature == -1)
        n_b = np.sum(xv_m.voxel_nature == 0)
        total = n_s + n_v + n_b
        assert total == xv_m.n_voxels
        assert n_s > 0, "Expected some solid voxels"
        assert n_v > 0, "Expected some void voxels"

    def test_edit_fillet_radius(self, xv):
        xv_m, fid_corner = xv
        n_s_before = np.sum(xv_m.voxel_nature == 1)
        xv_m.edit_parameter(fid_corner, 'r', 5.0)
        n_s_after = np.sum(xv_m.voxel_nature == 1)
        assert n_s_after <= n_s_before, (
            f"Smaller fillet should not increase solid count "
            f"(was {n_s_before}, now {n_s_after})"
        )
