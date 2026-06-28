# -*- coding: utf-8 -*-
"""
ibm.py — 全向量化浸入边界法 (Immersed Boundary Method) 力施加

直接力法 + 基于 SDF 的连续力权重:
    α(x) = 1                       若 Φ(x) ≤ 0      (固体内部)
    α(x) = 1 - |Φ(x)| / Δ          若 0 < Φ(x) < Δ  (边界区)
    α(x) = 0                       若 Φ(x) ≥ Δ      (流体区)
其中 Δ = min(dx, dy, dz).

IBM 体积力: f_ibm = (α / Δt) * (u_target - u*)
无滑移壁面: u_target = 0.

设计借鉴 fcm/assembly.py 的批量处理模式:
    - _classify_sdfs: 向量化分类
    - _get_elem_dofs_batch: 批量索引预计算
    - _scatter_batch: 批量散列

对 xvoxel 的所有访问均为只读. 零 Python for 循环.
"""
import numpy as np
from typing import Optional


class IBMForce:
    """全向量化 IBM 力施加 — 零 Python for 循环.

    对 xvoxel 的所有访问均为只读消费.

    Attributes:
        voxel_nature: (n_voxels,) int8 — 独立副本
        dx_min: min(dx, dy, dz)
        nx, ny, nz: 网格维度
        _ibm_ids: (M,) int — IBM 体素扁索引 (nature != -1)
        _n_ibm: IBM 体素数
        f_ibm: (3, nx, ny, nz) float64 — 最近一次施加的 IBM 体积力 (用于阻力积分)
    """

    def __init__(self, xv, dt: float = 0.01, rho: float = 1.0):
        """初始化 IBM 力, 预计算所有面索引.

        Args:
            xv: XVoxelModel 实例 (只读消费).
            dt: 伪瞬态时间步长 (用于直接力公式).
            rho: 流体密度 (用于直接力公式 f = ρ*α/Δt*(u_target-u)).
        """
        # 只读消费 xvoxel 数据
        self.voxel_nature = xv.voxel_nature.copy()       # 独立副本
        self._centers = xv._voxel_centers                # 引用 (不修改)
        self._csg_root = xv.csg_root                     # 引用 (只调用 sdf_batch)
        self.dx_min = float(min(xv.dx, xv.dy, xv.dz))
        self.nx, self.ny, self.nz = int(xv.nx), int(xv.ny), int(xv.nz)
        self.dt = float(dt)
        self.rho = float(rho)

        # 预计算 IBM 体素索引 — 一次计算, 永久复用
        # nature != -1 即固体内部 (+1) 或边界 (0)
        self._ibm_ids = np.where(self.voxel_nature != -1)[0].astype(np.int64)
        self._n_ibm = int(len(self._ibm_ids))

        # IBM 体积力存储 (体素中心, 用于阻力积分)
        self.f_ibm = np.zeros((3, self.nx, self.ny, self.nz), dtype=np.float64)

        # 预计算 IBM 体素的 (i,j,k) 坐标
        if self._n_ibm > 0:
            nx_ny = self.nx * self.ny
            self._i_ibm = (self._ibm_ids % self.nx).astype(np.int64)
            self._j_ibm = ((self._ibm_ids // self.nx) % self.ny).astype(np.int64)
            self._k_ibm = (self._ibm_ids // nx_ny).astype(np.int64)
            self._precompute_face_indices()

    # ------------------------------------------------------------------
    # 预计算面索引 (一次性, 全向量化)
    # ------------------------------------------------------------------
    def _precompute_face_indices(self) -> None:
        """预计算所有 IBM 体素的关联面心扁索引 — 一次性完成.

        参考 _get_elem_dofs_batch 的向量化思想:
        从 3D (i,j,k) 坐标到扁索引的批量转换.

        每个体素关联 6 个面心:
            u 面: u[i,j,k] (西) 和 u[i+1,j,k] (东)
            v 面: v[i,j,k] (南) 和 v[i,j+1,k] (北)
            w 面: w[i,j,k] (下) 和 w[i,j,k+1] (上)
        """
        i = self._i_ibm
        j = self._j_ibm
        k = self._k_ibm
        nx, ny, nz = self.nx, self.ny, self.nz

        # u-面扁索引: u[i, j, k] 和 u[i+1, j, k]
        #   u 形状 (nx+1, ny, nz), 扁索引 = k*(nx+1)*ny + j*(nx+1) + i
        u_stride_j = (nx + 1)
        u_stride_k = (nx + 1) * ny
        self._u_face1 = (k * u_stride_k + j * u_stride_j + i).astype(np.int64)
        self._u_face2 = (k * u_stride_k + j * u_stride_j + i + 1).astype(np.int64)

        # v-面扁索引: v[i, j, k] 和 v[i, j+1, k]
        #   v 形状 (nx, ny+1, nz), 扁索引 = k*nx*(ny+1) + j*nx + i
        v_stride_j = nx
        v_stride_k = nx * (ny + 1)
        self._v_face1 = (k * v_stride_k + j * v_stride_j + i).astype(np.int64)
        self._v_face2 = (k * v_stride_k + (j + 1) * v_stride_j + i).astype(np.int64)

        # w-面扁索引: w[i, j, k] 和 w[i, j, k+1]
        #   w 形状 (nx, ny, nz+1), 扁索引 = k*nx*ny + j*nx + i
        w_stride_j = nx
        w_stride_k = nx * ny
        self._w_face1 = (k * w_stride_k + j * w_stride_j + i).astype(np.int64)
        self._w_face2 = ((k + 1) * w_stride_k + j * w_stride_j + i).astype(np.int64)

    # ------------------------------------------------------------------
    # 权重计算 (全向量化)
    # ------------------------------------------------------------------
    def compute_weights(self) -> np.ndarray:
        """全向量化 IBM 权重计算.

        Returns:
            (M,) float64 — 每个 IBM 体素的权重 α.
        """
        if self._n_ibm == 0:
            return np.zeros(0, dtype=np.float64)

        # Step 1: 批量 SDF 求值 — 一次调用, C 级循环
        centers = self._centers[self._ibm_ids]          # (M, 3)
        sdf_vals = self._csg_root.sdf_batch(centers)    # (M,)

        # Step 2: 全向量化权重 — np.where 替代 if/else
        # sdf <= 0 (固体内部) → 1.0
        # 0 < sdf < dx_min (边界区) → 1 - sdf/dx_min
        # sdf >= dx_min (流体) → 0 (由 np.maximum 保证)
        weights = np.where(
            sdf_vals <= 0,
            1.0,
            np.maximum(0.0, 1.0 - sdf_vals / self.dx_min),
        )
        return weights

    # ------------------------------------------------------------------
    # IBM 力计算 (全向量化) — 仅记录力, 不修改速度
    # ------------------------------------------------------------------
    def compute_force(self, grid) -> None:
        """全向量化计算并记录 IBM 体积力 (直接力法), 不修改速度场.

        f_ibm = ρ * (α / Δt) * (u_target - u_cell)
        无滑移: u_target = 0 → f_ibm = -ρ*(α/Δt) * u_cell

        该力作为源项进入动量方程, 抵消伪瞬态项 aP0*u_old = ρ*ΔV/Δt*u_old,
        使固体区速度趋向 0. 力存储在 self.f_ibm (3, nx, ny, nz).

        为确保固体区速度严格为 0, 使用增强系数 (PENALTY * ρ*α/Δt),
        PENALTY >> 1 使 IBM 力主导动量方程.

        Args:
            grid: StaggeredGrid 实例 (只读, 不修改).
        """
        # 重置力存储
        self.f_ibm.fill(0.0)

        if self._n_ibm == 0:
            return

        # Step 1+2: 批量 SDF + 权重
        weights = self.compute_weights()                # (M,)

        # Step 3: 记录 IBM 体积力 (体素中心)
        # f_ibm = ρ * (α / Δt) * (0 - u_cell)
        # 抵消伪瞬态项 aP0*u_old, 使固体区速度趋向 0
        i = self._i_ibm
        j = self._j_ibm
        k = self._k_ibm
        # 体素中心速度近似: u = 0.5*(u[i]+u[i+1]), v = 0.5*(v[j]+v[j+1]), w = 0.5*(w[k]+w[k+1])
        u_cell = 0.5 * (grid.u[i, j, k] + grid.u[i + 1, j, k])
        v_cell = 0.5 * (grid.v[i, j, k] + grid.v[i, j + 1, k])
        w_cell = 0.5 * (grid.w[i, j, k] + grid.w[i, j, k + 1])

        coef = self.rho * weights / self.dt             # (M,) ρ*α/Δt
        self.f_ibm[0, i, j, k] = coef * (0.0 - u_cell)
        self.f_ibm[1, i, j, k] = coef * (0.0 - v_cell)
        self.f_ibm[2, i, j, k] = coef * (0.0 - w_cell)

    # ------------------------------------------------------------------
    # IBM 速度强制 (全向量化) — 直接抑制, 用于收敛后清理
    # ------------------------------------------------------------------
    def apply(self, grid) -> None:
        """全向量化 IBM 速度强制 — 直接力法 (速度阻尼).

        u* *= (1 - α)  在 IBM 体素关联面心.
        用于 SIMPLE 收敛后强制无滑移, 或作为简化 IBM 模式.

        Args:
            grid: StaggeredGrid 实例 (原地修改 u, v, w).
        """
        if self._n_ibm == 0:
            return

        # 批量 SDF + 权重
        weights = self.compute_weights()                # (M,)
        factors = 1.0 - weights                          # (M,) 速度保留因子

        # 批量速度修正 — numpy 高级索引 (无 Python 循环)
        u_flat = grid.u.ravel()
        v_flat = grid.v.ravel()
        w_flat = grid.w.ravel()

        u_flat[self._u_face1] *= factors
        u_flat[self._u_face2] *= factors
        v_flat[self._v_face1] *= factors
        v_flat[self._v_face2] *= factors
        w_flat[self._w_face1] *= factors
        w_flat[self._w_face2] *= factors

    # ------------------------------------------------------------------
    # 阻力 / 升力积分 (全向量化)
    # ------------------------------------------------------------------
    def total_force(self) -> np.ndarray:
        """积分 IBM 体积力得总力 (3,) — 物体所受力.

        F = -Σ f_ibm * ΔV  (牛顿第三定律: f_ibm 作用在流体上, 物体受力反向).

        Returns:
            (3,) float64 — (Fx, Fy, Fz) 物体所受总力.
        """
        dx_min = self.dx_min
        dV = dx_min * dx_min * dx_min                    # 近似体素体积
        return -self.f_ibm.sum(axis=(1, 2, 3)) * dV

    def drag_force(self) -> float:
        """阻力 (物体所受 x 方向力)."""
        return float(self.total_force()[0])

    def lift_force(self) -> float:
        """升力 (物体所受 y 方向力)."""
        return float(self.total_force()[1])
