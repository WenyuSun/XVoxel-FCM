# -*- coding: utf-8 -*-
"""Tests for the base FEM solver — simple cube tension and slender bar."""
import numpy as np
import pytest

from src.fem_base import HexMesh, FEMSolver


class TestCubeTension:
    """Validate FEM solver against analytical uniaxial tension."""

    @pytest.fixture
    def solver(self):
        mesh = HexMesh(4, 4, 4, 1.0, 1.0, 1.0)
        return FEMSolver(mesh, E=2e11, nu=0.0)

    @pytest.fixture
    def assembled(self, solver):
        K = solver.assemble_stiffness()
        F = np.zeros(solver.ndof)
        return K, F

    def test_cube_tension_displacement(self, solver, assembled):
        """Top-face displacement under uniform tension should match σ·L/E."""
        K, F = assembled
        nx = solver.mesh.nx
        ny = solver.mesh.ny
        nz = solver.mesh.nz
        lx, ly, lz = solver.mesh.lx, solver.mesh.ly, solver.mesh.lz

        fixed_dofs = []
        for i in range(nx + 1):
            for j in range(ny + 1):
                nid = i + j * (nx + 1)
                fixed_dofs.append(nid * 3 + 2)
        fixed_dofs.append(0)
        fixed_dofs.append(1)

        K, F = solver.apply_dirichlet(K, F, fixed_dofs)

        traction = 100.0
        for j in range(ny + 1):
            for i in range(nx + 1):
                nid = i + j * (nx + 1) + nz * (nx + 1) * (ny + 1)
                w_xi = 0.5 if i == 0 or i == nx else 1.0
                w_eta = 0.5 if j == 0 or j == ny else 1.0
                weight = w_xi * w_eta
                F[nid * 3 + 2] = traction * lx * ly * weight / (nx * ny)

        u = solver.solve(K, F)

        top_nodes = [
            i + j * (nx + 1) + nz * (nx + 1) * (ny + 1)
            for j in range(ny + 1) for i in range(nx + 1)
        ]
        actual_disp = np.mean([u[nid * 3 + 2] for nid in top_nodes])
        expected_disp = lz * traction / solver.E
        error_pct = abs(actual_disp - expected_disp) / expected_disp * 100

        assert error_pct < 1.0, (
            f"Tension error {error_pct:.2f}% — "
            f"expected {expected_disp:.6e}, got {actual_disp:.6e}"
        )


class TestSlenderBar:
    """Validate FEM solver on a slender bar (8×2×2)."""

    @pytest.fixture
    def solver(self):
        mesh = HexMesh(8, 2, 2, 8.0, 1.0, 1.0)
        return FEMSolver(mesh, E=2e11, nu=0.3)

    def test_slender_bar_displacement(self, solver):
        """End displacement of slender bar under tension."""
        nx, ny, nz = solver.mesh.nx, solver.mesh.ny, solver.mesh.nz
        lx, ly, lz = solver.mesh.lx, solver.mesh.ly, solver.mesh.lz

        K = solver.assemble_stiffness()
        F = np.zeros(solver.ndof)

        fixed_dofs = []
        for j in range(ny + 1):
            for k in range(nz + 1):
                nid = j * (nx + 1) + k * (nx + 1) * (ny + 1)
                fixed_dofs.extend([nid * 3, nid * 3 + 1, nid * 3 + 2])

        K, F = solver.apply_dirichlet(K, F, fixed_dofs)

        traction = 100.0
        for k in range(nz + 1):
            for j in range(ny + 1):
                nid = nx + j * (nx + 1) + k * (nx + 1) * (ny + 1)
                w_eta = 0.5 if j == 0 or j == ny else 1.0
                w_ze = 0.5 if k == 0 or k == nz else 1.0
                weight = w_eta * w_ze
                F[nid * 3] = traction * ly * lz * weight / (ny * nz)

        u = solver.solve(K, F)

        right_nodes = [
            nx + j * (nx + 1) + k * (nx + 1) * (ny + 1)
            for j in range(ny + 1) for k in range(nz + 1)
        ]
        actual_disp = np.mean([u[nid * 3] for nid in right_nodes])
        expected_disp = lx * traction / solver.E
        error_pct = abs(actual_disp - expected_disp) / expected_disp * 100

        assert error_pct < 2.0, (
            f"Slender bar error {error_pct:.2f}% — "
            f"expected {expected_disp:.6e}, got {actual_disp:.6e}"
        )
