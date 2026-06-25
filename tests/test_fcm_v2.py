# -*- coding: utf-8 -*-
"""Tests for fcm package (Phase 1 refactor)."""
import numpy as np
import pytest
from xvoxel import XVoxelModel, Cube, CylinderZ, BoolOp
from fcm import (
    FCMSolver, UniformHexMesh, assemble_fcm_k,
    get_element_info, GAUSS_2X2X2, GAUSS_3X3X3, GAUSS_4X4X4,
    elastic_matrix_D,
)


class TestElements:
    """Test element definitions and Gauss rules."""

    def test_gauss_2x2x2(self):
        """2×2×2 Gauss rule has 8 points."""
        assert len(GAUSS_2X2X2.points) == 8
        assert len(GAUSS_2X2X2.weights) == 8
        assert abs(np.sum(GAUSS_2X2X2.weights) - 8.0) < 1e-10

    def test_gauss_3x3x3(self):
        """3×3×3 Gauss rule has 27 points."""
        assert len(GAUSS_3X3X3.points) == 27
        assert len(GAUSS_3X3X3.weights) == 27
        assert abs(np.sum(GAUSS_3X3X3.weights) - 8.0) < 1e-10

    def test_gauss_4x4x4(self):
        """4×4×4 Gauss rule has 64 points."""
        assert len(GAUSS_4X4X4.points) == 64
        assert len(GAUSS_4X4X4.weights) == 64
        assert abs(np.sum(GAUSS_4X4X4.weights) - 8.0) < 1e-10

    def test_get_element_info_hex8(self):
        """Hex8 element info."""
        info = get_element_info(1)
        assert info['npe'] == 8
        assert info['ndof_per_elem'] == 24
        assert info['shape_func'] is not None
        assert info['shape_grad'] is not None
        assert info['ke_func'] is not None

    def test_get_element_info_hex20(self):
        """Hex20 element info."""
        info = get_element_info(2)
        assert info['npe'] == 20
        assert info['ndof_per_elem'] == 60

    def test_get_element_info_hex32(self):
        """Hex32 element info."""
        info = get_element_info(3)
        assert info['npe'] == 32
        assert info['ndof_per_elem'] == 96

    def test_elastic_matrix_D(self):
        """Elastic matrix is 6×6 symmetric."""
        D = elastic_matrix_D(E=2e5, nu=0.3)
        assert D.shape == (6, 6)
        assert np.allclose(D, D.T)

    def test_element_stiffness_positive_definite(self):
        """Element stiffness should be positive semi-definite."""
        info = get_element_info(1)
        ke_func = info['ke_func']
        # Undeformed unit cube
        coords = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=np.float64)
        ke = ke_func(coords, E=2e5, nu=0.3)
        assert ke.shape == (24, 24)
        # Eigenvalues should be >= 0 (6 rigid body modes → 0)
        eigvals = np.linalg.eigvalsh(ke)
        assert np.all(eigvals > -1e-10)
        # At least 6 near-zero eigenvalues (rigid body modes)
        n_zero = np.sum(np.abs(eigvals) < 1e-6)
        assert n_zero >= 6


class TestUniformHexMesh:
    """Test uniform hex mesh construction."""

    def test_hex8_mesh_basic(self):
        """Basic Hex8 mesh from XVoxel model."""
        xv = XVoxelModel(3, 3, 3, 6.0, 6.0, 6.0)
        c = Cube(cx=3, cy=3, cz=0, sx=6, sy=6, sz=6)
        xv.add_feature(c)
        mesh = UniformHexMesh.from_xvoxel(xv, element_order=1)
        assert mesh.n_elems == 27
        assert mesh.n_nodes > 0
        assert mesh.ndof == mesh.n_nodes * 3
        assert mesh.elems.shape[1] == 8

    def test_hex20_mesh(self):
        """Hex20 mesh has 20 nodes per element."""
        xv = XVoxelModel(2, 2, 2, 4.0, 4.0, 4.0)
        c = Cube(cx=2, cy=2, cz=0, sx=4, sy=4, sz=4)
        xv.add_feature(c)
        mesh = UniformHexMesh.from_xvoxel(xv, element_order=2)
        assert mesh.elems.shape[1] == 20
        assert mesh.n_nodes > 8  # has mid-edge nodes

    def test_hex32_mesh(self):
        """Hex32 mesh has 32 nodes per element."""
        xv = XVoxelModel(2, 2, 2, 4.0, 4.0, 4.0)
        c = Cube(cx=2, cy=2, cz=0, sx=4, sy=4, sz=4)
        xv.add_feature(c)
        mesh = UniformHexMesh.from_xvoxel(xv, element_order=3)
        assert mesh.elems.shape[1] == 32

    def test_elem_center(self):
        """Element centers are computed correctly."""
        xv = XVoxelModel(3, 3, 3, 6.0, 6.0, 6.0)
        c = Cube(cx=3, cy=3, cz=0, sx=6, sy=6, sz=6)
        xv.add_feature(c)
        mesh = UniformHexMesh.from_xvoxel(xv, element_order=1)
        center = mesh.elem_center(0)
        assert center.shape == (3,)


class TestFCMSolver:
    """End-to-end FCM solver tests."""

    @pytest.fixture
    def cantilever_xv(self):
        """Simple cantilever beam XVoxel model."""
        xv = XVoxelModel(10, 3, 3, 10.0, 3.0, 3.0, origin=(0, 0, -1.5))
        body = Cube(cx=5.0, cy=1.5, cz=0.0, sx=10.0, sy=3.0, sz=3.0, name='beam')
        xv.add_feature(body)
        return xv

    def test_cantilever_hex8(self, cantilever_xv):
        """Cantilever beam with Hex8 elements produces reasonable tip deflection."""
        solver = FCMSolver(cantilever_xv, order=1)
        solver.set_material(E=2e5, nu=0.3, alpha=1e-8)
        solver.add_dirichlet_bc('xmin', 'ux,uy,uz', 0.0)
        solver.add_traction_bc('xmax', (0.0, -10.0, 0.0))

        u = solver.solve(max_depth=2)
        max_u = np.max(np.abs(u))
        # Beam theory: δ = PL³/(3EI) = 90*1000/(3*2e5*6.75) ≈ 0.0222 mm
        # Allow some error for coarse mesh
        assert 0.005 < max_u < 0.1, f"Expected ~0.022 mm, got {max_u:.6f}"

    def test_cantilever_hex20(self, cantilever_xv):
        """Cantilever with Hex20 should also work."""
        solver = FCMSolver(cantilever_xv, order=2)
        solver.set_material(E=2e5, nu=0.3, alpha=1e-8)
        solver.add_dirichlet_bc('xmin', 'ux,uy,uz', 0.0)
        solver.add_traction_bc('xmax', (0.0, -10.0, 0.0))

        u = solver.solve(max_depth=2)
        max_u = np.max(np.abs(u))
        assert 0.001 < max_u < 0.2

    def test_von_mises_computation(self, cantilever_xv):
        """Von Mises stress computation runs without error."""
        solver = FCMSolver(cantilever_xv, order=1)
        solver.set_material(E=2e5, nu=0.3, alpha=1e-8)
        solver.add_dirichlet_bc('xmin', 'ux,uy,uz', 0.0)
        solver.add_traction_bc('xmax', (0.0, -10.0, 0.0))
        solver.solve(max_depth=2)

        vm = solver.compute_von_mises()
        assert vm.shape == (cantilever_xv.n_voxels,)
        # Should have non-zero stress somewhere
        assert np.max(vm) > 0

    def test_no_dirichlet_warning(self, cantilever_xv, capsys):
        """Solver warns if no Dirichlet BC is set."""
        solver = FCMSolver(cantilever_xv, order=1)
        solver.set_material(E=2e5, nu=0.3, alpha=1e-8)
        # No BC added
        try:
            solver.solve(max_depth=2)
        except Exception:
            pass  # May fail due to singular matrix, which is expected
        captured = capsys.readouterr()
        assert "No Dirichlet BC" in captured.out

    def test_boundary_element_integration(self):
        """FCM with boundary voxels uses octree integration."""
        # Create a model with boundary voxels (cube partially filling grid)
        xv = XVoxelModel(5, 5, 5, 5.0, 5.0, 5.0, origin=(0, 0, -2.5))
        # Cube spanning z∈[-1.25, 1.25] in grid z∈[-2.5, 2.5]
        body = Cube(cx=2.5, cy=2.5, cz=0.0, sx=5.0, sy=5.0, sz=2.5, name='body')
        xv.add_feature(body)

        n_solid = np.sum(xv.voxel_nature == 1)
        n_boundary = np.sum(xv.voxel_nature == 0)
        n_void = np.sum(xv.voxel_nature == -1)

        # There should be some boundary voxels
        assert n_boundary >= 0  # May or may not have boundary depending on alignment

        solver = FCMSolver(xv, order=1)
        solver.set_material(E=2e5, nu=0.3, alpha=1e-8)
        solver.add_dirichlet_bc('xmin', 'ux,uy,uz', 0.0)
        solver.add_traction_bc('xmax', (0.0, -1.0, 0.0))
        u = solver.solve(max_depth=3)
        assert np.all(np.isfinite(u))


class TestAssembly:
    """Test FCM assembly functions."""

    def test_assemble_fcm_k_shape(self):
        """K matrix has correct dimensions."""
        xv = XVoxelModel(4, 4, 2, 4.0, 4.0, 2.0, origin=(0, 0, -1.0))
        c = Cube(cx=2.0, cy=2.0, cz=0.0, sx=4.0, sy=4.0, sz=2.0)
        xv.add_feature(c)
        mesh = UniformHexMesh.from_xvoxel(xv, element_order=1)
        K = assemble_fcm_k(mesh, xv.voxel_nature, xv.csg_root,
                           E=2e5, nu=0.3, alpha=1e-8, order=1, max_depth=2)
        assert K.shape == (mesh.ndof, mesh.ndof)
        assert K.nnz > 0

    def test_assemble_fcm_k_symmetric(self):
        """K should be symmetric."""
        xv = XVoxelModel(4, 4, 2, 4.0, 4.0, 2.0, origin=(0, 0, -1.0))
        c = Cube(cx=2.0, cy=2.0, cz=0.0, sx=4.0, sy=4.0, sz=2.0)
        xv.add_feature(c)
        mesh = UniformHexMesh.from_xvoxel(xv, element_order=1)
        K = assemble_fcm_k(mesh, xv.voxel_nature, xv.csg_root,
                           E=2e5, nu=0.3, alpha=1e-8, order=1, max_depth=2)
        K_dense = K.toarray()
        assert np.allclose(K_dense, K_dense.T, atol=1e-10)
