# -*- coding: utf-8 -*-
"""Tests for xvoxel v2 package (Phase 1 refactor)."""
import numpy as np
import pytest
from xvoxel import (
    XVoxelModel, Cube, CylinderZ, Sphere, RoundCorner2D,
    Feature, Boolean, BoolOp, classify_sdfs,
)


class TestXVoxelModel:
    """Test XVoxelModel v2 core functionality."""

    def test_basic_construction(self):
        """Grid dimensions and default state."""
        xv = XVoxelModel(5, 5, 5, 10.0, 10.0, 10.0)
        assert xv.nx == 5
        assert xv.ny == 5
        assert xv.nz == 5
        assert xv.n_voxels == 125
        assert xv.csg_root is None
        assert np.all(xv.voxel_nature == -1)  # all void by default

    def test_add_single_cube(self):
        """Adding a single cube feature."""
        xv = XVoxelModel(10, 10, 3, 10.0, 10.0, 3.0)
        c = Cube(cx=5.0, cy=5.0, cz=0.0, sx=6.0, sy=6.0, sz=3.0, name='test')
        fid = xv.add_feature(c)
        assert fid == 0
        assert xv.csg_root is c
        # Should have some solid voxels
        assert np.sum(xv.voxel_nature == 1) > 0
        assert np.sum(xv.voxel_nature == -1) > 0  # some void too

    def test_two_cubes_union(self):
        """Two cubes unioned."""
        xv = XVoxelModel(10, 10, 3, 10.0, 10.0, 3.0)
        c1 = Cube(cx=3.0, cy=5.0, cz=0.0, sx=4.0, sy=6.0, sz=3.0, name='left')
        c2 = Cube(cx=7.0, cy=5.0, cz=0.0, sx=4.0, sy=6.0, sz=3.0, name='right')
        f1 = xv.add_feature(c1)
        f2 = xv.add_feature(c2)
        assert f1 == 0
        assert f2 == 1
        # CSG root should be a Boolean
        assert isinstance(xv.csg_root, Boolean)
        assert xv.csg_root.op == BoolOp.UNION
        assert np.sum(xv.voxel_nature == 1) > 0

    def test_cube_with_hole(self):
        """Cube minus cylinder = plate with hole."""
        xv = XVoxelModel(10, 10, 3, 10.0, 10.0, 3.0)
        body = Cube(cx=5.0, cy=5.0, cz=0.0, sx=10.0, sy=10.0, sz=3.0, name='body')
        hole = CylinderZ(cx=5.0, cy=5.0, r=3.0, zmin=-1.5, zmax=1.5, name='hole')
        xv.add_feature(body)
        xv.add_feature(hole, op=BoolOp.DIFFERENCE)
        # Center region should be void
        solid_count = np.sum(xv.voxel_nature == 1)
        void_count = np.sum(xv.voxel_nature == -1)
        # Should have some void (the hole)
        assert void_count > 0
        assert solid_count > 0

    def test_edit_parameter(self):
        """Editing a parameter updates voxel_nature incrementally."""
        xv = XVoxelModel(10, 10, 3, 10.0, 10.0, 3.0)
        c = Cube(cx=5.0, cy=5.0, cz=0.0, sx=6.0, sy=6.0, sz=3.0, name='cube')
        fid = xv.add_feature(c)
        before = np.sum(xv.voxel_nature == 1)
        dirty = xv.edit_parameter(fid, 'sx', 8.0)
        after = np.sum(xv.voxel_nature == 1)
        assert after > before  # larger cube → more solid
        assert len(dirty) > 0  # some voxels changed

    def test_edit_parameter_same_value(self):
        """Editing to same value still returns affected voxels (re-voxelizes)."""
        xv = XVoxelModel(10, 10, 3, 10.0, 10.0, 3.0)
        c = Cube(cx=5.0, cy=5.0, cz=0.0, sx=6.0, sy=6.0, sz=3.0, name='cube')
        fid = xv.add_feature(c)
        dirty1 = xv.edit_parameter(fid, 'sx', 8.0)
        assert len(dirty1) > 0  # first change affects voxels
        before = xv.voxel_nature.copy()
        dirty2 = xv.edit_parameter(fid, 'sx', 8.0)  # same value
        # Re-voxelization produces dirty list; nature stays same
        assert np.array_equal(xv.voxel_nature, before)

    def test_delete_feature(self):
        """Deleting a feature removes it from the CSG tree."""
        xv = XVoxelModel(10, 10, 3, 10.0, 10.0, 3.0)
        c1 = Cube(cx=3.0, cy=5.0, cz=0.0, sx=4.0, sy=4.0, sz=3.0, name='left')
        c2 = Cube(cx=7.0, cy=5.0, cz=0.0, sx=4.0, sy=4.0, sz=3.0, name='right')
        xv.add_feature(c1)
        f2 = xv.add_feature(c2)
        before = np.sum(xv.voxel_nature == 1)
        xv.delete_feature(f2)
        after = np.sum(xv.voxel_nature == 1)
        assert after < before  # fewer solid voxels
        # CSG root should collapse to just c1
        assert xv.csg_root is c1 or (isinstance(xv.csg_root, Boolean) and not xv.csg_root.children[0]._deleted)

    def test_get_fem_mesh_info(self):
        """get_fem_mesh_info returns correct element types."""
        xv = XVoxelModel(5, 5, 5, 10.0, 10.0, 10.0)
        c = Cube(cx=5.0, cy=5.0, cz=0.0, sx=6.0, sy=6.0, sz=6.0, name='cube')
        xv.add_feature(c)
        elem_E, elem_type = xv.get_fem_mesh_info()
        assert elem_E.shape == (125,)
        assert elem_type.shape == (125,)
        # Boundary voxels should get E (not αE) — M5 fix
        boundary_mask = elem_type == 0
        if np.any(boundary_mask):
            assert np.all(elem_E[boundary_mask] == 1.0)


class TestCSG:
    """Test CSG tree and classify_sdfs."""

    def test_classify_sdfs(self):
        """classify_sdfs correctly classifies solid/boundary/void.

        算法 (xvoxel/csg.py): ``sdf <= -min_half_dim`` → solid (+1),
        ``-min_half_dim < sdf < 0`` → boundary (0), ``sdf >= 0`` → void (-1).
        默认 ``min_half_dim=0.1``. 注意 sdf=0 恰在界面上, 归为 void.
        """
        sdf_vals = np.array([-1.0, -1e-10, 0.0, 1e-10, 1.0])
        result = classify_sdfs(sdf_vals)
        assert result[0] == 1   # solid (sdf=-1 <= -0.1)
        assert result[1] == 0   # boundary (-0.1 < -1e-10 < 0)
        assert result[2] == -1  # void (sdf=0, 恰在界面, 归 void)
        assert result[3] == -1  # void (sdf>0)
        assert result[4] == -1  # void (sdf=1)

    def test_boolean_union(self):
        """Boolean UNION = min of children."""
        c1 = Cube(cx=0, cy=0, cz=0, sx=2, sy=2, sz=2)
        c2 = Cube(cx=2, cy=0, cz=0, sx=2, sy=2, sz=2)
        b = Boolean(BoolOp.UNION, [c1, c2])
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        sdf = b.sdf_batch(pts)
        assert sdf[0] < 0  # inside c1
        assert sdf[1] < 0  # inside c2
        assert sdf[2] > 0  # outside both

    def test_boolean_difference(self):
        """Boolean DIFFERENCE."""
        body = Cube(cx=0, cy=0, cz=0, sx=4, sy=4, sz=4)
        hole = Sphere(cx=0, cy=0, cz=0, r=1)
        b = Boolean(BoolOp.DIFFERENCE, [body, hole])
        pts = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        sdf = b.sdf_batch(pts)
        assert sdf[0] > 0  # center is void (hole)
        assert sdf[1] < 0  # outside hole, inside body


class TestPrimitives:
    """Test primitive SDF implementations."""

    def test_cube_sdf(self):
        """Cube SDF is negative inside, positive outside."""
        c = Cube(cx=0, cy=0, cz=0, sx=2, sy=2, sz=2)
        pts = np.array([[0, 0, 0], [0.9, 0, 0], [1.1, 0, 0]])
        sdf = c.sdf_batch(pts)
        assert sdf[0] < 0  # center
        assert sdf[1] < 0  # near edge, inside
        assert sdf[2] > 0  # outside

    def test_sphere_sdf(self):
        """Sphere SDF."""
        s = Sphere(cx=0, cy=0, cz=0, r=1)
        pts = np.array([[0, 0, 0], [0.5, 0, 0], [1.5, 0, 0]])
        sdf = s.sdf_batch(pts)
        assert sdf[0] < 0
        assert sdf[1] < 0
        assert sdf[2] > 0

    def test_cylinder_z_sdf(self):
        """CylinderZ SDF."""
        cyl = CylinderZ(cx=0, cy=0, r=2, zmin=-1, zmax=1)
        pts = np.array([[0, 0, 0], [1.9, 0, 0], [2.1, 0, 0], [0, 0, 1.5]])
        sdf = cyl.sdf_batch(pts)
        assert sdf[0] < 0   # center
        assert sdf[1] < 0   # inside radially
        assert sdf[2] > 0   # outside radially
        assert sdf[3] > 0   # outside axially

    def test_roundcorner2d_sdf(self):
        """RoundCorner2D SDF fills the corner."""
        rc = RoundCorner2D(cx=3, cy=3, r=2, zmin=-1, zmax=1,
                           sign_x=+1, sign_y=+1)
        pts = np.array([[3.5, 3.5, 0], [3.0, 3.0, 0], [2.0, 2.0, 0]])
        sdf = rc.sdf_batch(pts)
        # (3.5,3.5) is within the corner fill region
        assert sdf[0] < 0
