# -*- coding: utf-8 -*-
"""
pressure.py — 压力修正 Poisson 方程求解器

压力修正方程 (SIMPLE):
    ∇²p' = (ρ/Δt) * ∇·u*

五点(七点)离散:
    (p'_E - 2p'_P + p'_W)/dx² + (p'_N - 2p'_P + p'_S)/dy² + (p'_T - 2p'_P + p'_B)/dz²
        = (ρ/Δt) * [(u*_E - u*_W)/(2dx) + ...]

实际用面心散度 (更精确):
    RHS = (ρ/Δt) * [(u[i+1]-u[i])/dx + (v[j+1]-v[j])/dy + (w[k+1]-w[k])/dz]

求解: Red-Black SOR (向量化). RHS 全向量化构建.
"""
import numpy as np


class PressureSolver:
    """压力 Poisson 求解器 (Red-Black SOR, 向量化).

    Attributes:
        grid: StaggeredGrid 实例
        rho: 流体密度
        dt: 时间步长
        omega: SOR 松弛因子
    """

    def __init__(self, grid, rho: float, dt: float, omega: float = 1.7):
        self.grid = grid
        self.rho = float(rho)
        self.dt = float(dt)
        self.omega = float(omega)

    def build_rhs(self, u: np.ndarray, v: np.ndarray,
                  w: np.ndarray) -> np.ndarray:
        """全向量化构建压力修正方程 RHS.

        RHS = (ρ/Δt) * ∇·u*

        Returns:
            (nx, ny, nz) float64 — RHS.
        """
        div = ((u[1:, :, :] - u[:-1, :, :]) / self.grid.dx +
               (v[:, 1:, :] - v[:, :-1, :]) / self.grid.dy +
               (w[:, :, 1:] - w[:, :, :-1]) / self.grid.dz)
        return (self.rho / self.dt) * div

    def solve(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
              n_iter: int = 50) -> np.ndarray:
        """求解压力修正方程 ∇²p' = RHS (Red-Black SOR).

        Args:
            u, v, w: 星号速度场 (面心).
            n_iter: SOR 迭代次数.

        Returns:
            (nx, ny, nz) float64 — 压力修正量 p'.
        """
        rhs = self.build_rhs(u, v, w)
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz

        # 离散系数
        dxi2 = 1.0 / (dx * dx)
        dyi2 = 1.0 / (dy * dy)
        dzi2 = 1.0 / (dz * dz)
        aP = 2.0 * (dxi2 + dyi2 + dzi2)

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        i_idx = np.arange(nx)
        j_idx = np.arange(ny)
        k_idx = np.arange(nz)
        I, J, K = np.meshgrid(i_idx, j_idx, k_idx, indexing='ij')
        red_mask = ((I + J + K) % 2 == 0)
        black_mask = ~red_mask

        p_corr = np.zeros((nx, ny, nz), dtype=np.float64)

        for _ in range(n_iter):
            # 邻居 (边界用 0 — Neumann BC for p')
            p_E = np.zeros_like(p_corr)
            p_E[:-1, :, :] = p_corr[1:, :, :]
            p_W = np.zeros_like(p_corr)
            p_W[1:, :, :] = p_corr[:-1, :, :]
            p_N = np.zeros_like(p_corr)
            p_N[:, :-1, :] = p_corr[:, 1:, :]
            p_S = np.zeros_like(p_corr)
            p_S[:, 1:, :] = p_corr[:, :-1, :]
            p_T = np.zeros_like(p_corr)
            p_T[:, :, :-1] = p_corr[:, :, 1:]
            p_B = np.zeros_like(p_corr)
            p_B[:, :, 1:] = p_corr[:, :, :-1]

            resid = (dxi2 * (p_E + p_W) + dyi2 * (p_N + p_S) +
                     dzi2 * (p_T + p_B) - aP * p_corr - rhs)

            # Red
            p_corr = np.where(red_mask,
                              p_corr + self.omega * resid / aP,
                              p_corr)

            # 重新取邻居
            p_E = np.zeros_like(p_corr)
            p_E[:-1, :, :] = p_corr[1:, :, :]
            p_W = np.zeros_like(p_corr)
            p_W[1:, :, :] = p_corr[:-1, :, :]
            p_N = np.zeros_like(p_corr)
            p_N[:, :-1, :] = p_corr[:, 1:, :]
            p_S = np.zeros_like(p_corr)
            p_S[:, 1:, :] = p_corr[:, :-1, :]
            p_T = np.zeros_like(p_corr)
            p_T[:, :, :-1] = p_corr[:, :, 1:]
            p_B = np.zeros_like(p_corr)
            p_B[:, :, 1:] = p_corr[:, :, :-1]
            resid = (dxi2 * (p_E + p_W) + dyi2 * (p_N + p_S) +
                     dzi2 * (p_T + p_B) - aP * p_corr - rhs)

            # Black
            p_corr = np.where(black_mask,
                              p_corr + self.omega * resid / aP,
                              p_corr)

        return p_corr

    def residual_norm(self, p_corr: np.ndarray, rhs: np.ndarray) -> float:
        """计算 Poisson 方程残差范数 (收敛诊断)."""
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        dxi2 = 1.0 / (dx * dx)
        dyi2 = 1.0 / (dy * dy)
        dzi2 = 1.0 / (dz * dz)
        aP = 2.0 * (dxi2 + dyi2 + dzi2)

        p_E = np.zeros_like(p_corr)
        p_E[:-1, :, :] = p_corr[1:, :, :]
        p_W = np.zeros_like(p_corr)
        p_W[1:, :, :] = p_corr[:-1, :, :]
        p_N = np.zeros_like(p_corr)
        p_N[:, :-1, :] = p_corr[:, 1:, :]
        p_S = np.zeros_like(p_corr)
        p_S[:, 1:, :] = p_corr[:, :-1, :]
        p_T = np.zeros_like(p_corr)
        p_T[:, :, :-1] = p_corr[:, :, 1:]
        p_B = np.zeros_like(p_corr)
        p_B[:, :, 1:] = p_corr[:, :, :-1]

        resid = (dxi2 * (p_E + p_W) + dyi2 * (p_N + p_S) +
                 dzi2 * (p_T + p_B) - aP * p_corr - rhs)
        return float(np.max(np.abs(resid)))
