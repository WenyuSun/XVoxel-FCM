# -*- coding: utf-8 -*-
"""Tests for the XVoxel data structure — add, edit, delete features."""
import numpy as np
import pytest

from src.primitives import Cube, CylinderZ
from src.xvoxel import XVoxelModel
from src.pmc import pmc_point_3d


class TestXVoxelOperations:
    """Basic XVoxel CRUD: add, edit parameter, delete."""

    @pytest.fixture
    def xv(self):
        xv = XVoxelModel(10, 10, 10, 1.0, 1.0, 1.0, origin=(-0.5, -0.5, -0.5))
        body = Cube(cx=0, cy=0, cz=0, sx=0.8, sy=0.8, sz=0.8)
        xv.add_feature(body, nature=1, name="body")
        hole = CylinderZ(cx=0, cy=0, r=0.15, zmin=-0.5, zmax=0.5)
        xv.add_feature(hole, nature=-1, name="hole")
        return xv

    def test_add_body_creates_solid_voxels(self, xv):
        n_solid = np.sum(xv.voxel_nature == 1)
        assert n_solid > 0, "Expected some solid voxels after adding body"

    def test_add_hole_creates_void_voxels(self, xv):
        n_void = np.sum(xv.voxel_nature == -1)
        assert n_void > 0, "Expected some void voxels after adding hole"

    def test_pmc_center_is_void(self, xv):
        center_idx = xv._idx(5, 5, 5)
        status = pmc_point_3d(0, 0, 0, xv.voxel_attrs[center_idx], xv.features)
        assert status == -1, f"Expected void (-1) at center, got {status}"

    def test_pmc_corner_is_solid(self, xv):
        corner_idx = xv._idx(8, 8, 5)
        status = pmc_point_3d(0.3, 0.3, 0, xv.voxel_attrs[corner_idx], xv.features)
        assert status == 1, f"Expected solid (+1) at corner, got {status}"

    def test_edit_hole_radius(self, xv):
        n_void_before = np.sum(xv.voxel_nature == -1)
        hole_fid = xv.features[1].feature_id
        active = xv.edit_parameter(hole_fid, 'r', 0.25)
        n_void_after = np.sum(xv.voxel_nature == -1)
        assert n_void_after > n_void_before, (
            f"Larger hole should create more void voxels "
            f"(was {n_void_before}, now {n_void_after})"
        )
        assert len(active) > 0, "Edit should report affected voxels"

    def test_delete_hole_restores_solid(self, xv):
        hole_fid = xv.features[1].feature_id
        xv.delete_feature(hole_fid)
        center_idx = xv._idx(5, 5, 5)
        status = pmc_point_3d(0, 0, 0, xv.voxel_attrs[center_idx], xv.features)
        assert status == 1, (
            f"After hole deletion, center should be solid (+1), got {status}"
        )


class TestXVoxelIndexing:
    """Verify voxel indexing conventions."""

    def test_idx_ijk_roundtrip(self):
        xv = XVoxelModel(10, 10, 10, 1.0, 1.0, 1.0, origin=(-0.5, -0.5, -0.5))
        for i, j, k in [(0, 0, 0), (5, 5, 5), (9, 9, 9), (3, 7, 2)]:
            idx = xv._idx(i, j, k)
            ii, jj, kk = xv._ijk(idx)
            assert (i, j, k) == (ii, jj, kk)

    def test_voxel_center(self):
        xv = XVoxelModel(10, 10, 10, 1.0, 1.0, 1.0, origin=(-0.5, -0.5, -0.5))
        c = xv.voxel_center(0, 0, 0)
        assert np.allclose(c, [-0.45, -0.45, -0.45])
        c = xv.voxel_center(9, 9, 9)
        assert np.allclose(c, [0.45, 0.45, 0.45])

    def test_n_voxels(self):
        xv = XVoxelModel(10, 10, 10, 1.0, 1.0, 1.0)
        assert xv.n_voxels == 1000
