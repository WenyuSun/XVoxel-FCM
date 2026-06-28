# -*- coding: utf-8 -*-
"""
simple_solver.py — SIMPLE 算法主循环

伪瞬态 SIMPLE (稳态求解):
    初始化 u, v, w, p
    for iter = 1 to max_iter:
        1. 求解 u-动量 → u*
        2. 求解 v-动量 → v*
        3. 求解 w-动量 → w*
        4. 求解压力修正 ∇²p' = (ρ/Δt)∇·u* → p'
        5. 速度修正: u ← u* - (Δt/ρ)∇p'
        6. 压力修正: p ← p + α_p·p'
        7. IBM 力施加: ibm.apply()
        8. BC 施加: bc.apply()
        9. 收敛判断: max|∇·u| < ε

调度器本身无内循环 (仅最外层 for iter).
"""
import numpy as np
from typing import Tuple
from .momentum import MomentumSolver
from .pressure import PressureSolver


class SIMPLESolver:
    """SIMPLE 算法主循环.

    Attributes:
        grid: StaggeredGrid 实例
        rho, nu: 流体物性
        dt: 伪瞬态时间步长
        alpha_p: 压力欠松弛因子
        alpha_u: 动量欠松弛因子
        ibm: 边界处理器 (IBMForce 源项 或 SharpIBMHandler 尖锐界面, 鸭子类型)
        bc: FluidBC 实例
    """

    def __init__(self, grid, xv, rho: float, nu: float,
                 ibm, bc, dt: float = 0.01,
                 alpha_p: float = 0.3, alpha_u: float = 0.7):
        self.grid = grid
        self.xv = xv
        self.rho = float(rho)
        self.nu = float(nu)
        self.dt = float(dt)
        self.alpha_p = float(alpha_p)
        self.alpha_u = float(alpha_u)
        self.ibm = ibm
        self.bc = bc

        # 同步 IBM 的 dt 和 rho
        self.ibm.dt = self.dt
        self.ibm.rho = self.rho

        self._momentum = MomentumSolver(grid, rho, nu, dt, alpha_u)
        self._pressure = PressureSolver(grid, rho, dt)

        # 收敛历史
        self.div_history = []
        self.residual_history = []

    def solve_steady(self, max_iter: int = 5000,
                     tol: float = 1e-6,
                     momentum_iter: int = 20,
                     pressure_iter: int = 50,
                     verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """求解稳态流场.

        Args:
            max_iter: 最大 SIMPLE 外迭代次数.
            tol: 散度收敛容差 max|∇·u|.
            momentum_iter: 每步动量方程内迭代次数.
            pressure_iter: 每步压力修正内迭代次数.
            verbose: 打印收敛历史.

        Returns:
            (u, v, p) 收敛的速度场与压力场.
        """
        grid = self.grid

        # 初始 BC 施加
        self.bc.apply(grid)

        for it in range(max_iter):
            # 1. 计算 IBM 体积力 (基于当前速度场, 作为动量源项)
            # f_ibm = PENALTY*ρ*(α/Δt)*(0 - u_cell), 在固体区强力驱动 u→0
            self.ibm.compute_force(grid)

            # 2-4. 动量方程求解 (含 IBM 源项) → u*, v*, w*
            u_star = self._momentum.solve_u(grid.u, grid.v, grid.w,
                                            grid.p, self.ibm.f_ibm[0],
                                            n_iter=momentum_iter)
            v_star = self._momentum.solve_v(u_star, grid.v, grid.w,
                                            grid.p, self.ibm.f_ibm[1],
                                            n_iter=momentum_iter)
            w_star = self._momentum.solve_w(u_star, v_star, grid.w,
                                            grid.p, self.ibm.f_ibm[2],
                                            n_iter=momentum_iter)

            # 施加 BC 到星号场
            grid.u, grid.v, grid.w = u_star, v_star, w_star
            self.bc.apply(grid)

            # 5. 压力修正 (基于星号速度场)
            p_corr = self._pressure.solve(grid.u, grid.v, grid.w,
                                          n_iter=pressure_iter)

            # 6. 速度修正: u ← u* - (Δt/ρ)∇p'
            self._correct_velocities(grid.u, grid.v, grid.w, p_corr)

            # 7. 压力修正: p ← p + α_p·p'
            grid.p = grid.p + self.alpha_p * p_corr

            # 8. BC 施加 (IBM 力已在动量源中处理)
            self.bc.apply(grid)

            # 9. 收敛判断
            div = grid.divergence()
            div_max = float(np.max(np.abs(div)))
            self.div_history.append(div_max)

            if verbose and (it % 50 == 0 or it < 5 or div_max < tol):
                print(f"  [SIMPLE iter {it:4d}] max|div| = {div_max:.6e}")

            if div_max < tol and it > 10:
                if verbose:
                    print(f"  [SIMPLE] converged at iter {it}, max|div| = {div_max:.6e}")
                break

        # 收敛后: 重新计算 IBM 力 (基于最终速度场, 用于阻力积分)
        self.ibm.compute_force(grid)

        return grid.u, grid.v, grid.p

    def _correct_velocities(self, u_star: np.ndarray, v_star: np.ndarray,
                            w_star: np.ndarray, p_corr: np.ndarray,
                            u_face_mask: np.ndarray = None,
                            v_face_mask: np.ndarray = None,
                            w_face_mask: np.ndarray = None) -> None:
        """速度修正 (全向量化, IBM 面不修正).

        u ← u* - (Δt/ρ) * (p'[i] - p'[i-1]) / dx
        v ← v* - (Δt/ρ) * (p'[j] - p'[j-1]) / dy
        w ← w* - (Δt/ρ) * (p'[k] - p'[k-1]) / dz

        固体面 (mask=True) 保持 0, 不参与压力修正.
        """
        coef = self.dt / self.rho
        # u 修正 (内部面)
        u_corr = u_star[1:-1, :, :] - coef * (
            p_corr[1:, :, :] - p_corr[:-1, :, :]) / self.grid.dx
        if u_face_mask is not None:
            u_corr = np.where(u_face_mask, 0.0, u_corr)
        self.grid.u[1:-1, :, :] = u_corr
        # v 修正 (内部面)
        v_corr = v_star[:, 1:-1, :] - coef * (
            p_corr[:, 1:, :] - p_corr[:, :-1, :]) / self.grid.dy
        if v_face_mask is not None:
            v_corr = np.where(v_face_mask, 0.0, v_corr)
        self.grid.v[:, 1:-1, :] = v_corr
        # w 修正 (内部面)
        w_corr = w_star[:, :, 1:-1] - coef * (
            p_corr[:, :, 1:] - p_corr[:, :, :-1]) / self.grid.dz
        if w_face_mask is not None:
            w_corr = np.where(w_face_mask, 0.0, w_corr)
        self.grid.w[:, :, 1:-1] = w_corr
