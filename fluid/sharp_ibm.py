# -*- coding: utf-8 -*-
"""
sharp_ibm.py — 尖锐界面浸入边界法 (Sharp-Interface IBM)

核心数值技术 (移植自 fcm/assembly.py):
    - 八叉树自适应细分边界体素
    - 高斯点 SDF 分类 (solid / void)
    - 沿 SDF 等值面 Φ=0 定位界面高斯点, 由 ∇Φ 解析求法向 n

与源项 IBM (ibm.py) 的根本区别:
    - 源项 IBM: 体素并集 + 体积力 f=ρ·α/Δt·(0-u), 体积分 ∫f dV
    - 尖锐界面: 界面高斯点强形式无滑移 u=0, 面积分 ∮σ·n dS

设计对齐 IBMForce 接口 (鸭子类型, 供 SIMPLESolver 注入):
    - compute_force(grid): 计算并记录界面力 (用于阻力积分)
    - apply(grid): 强制无滑移 (界面点速度 → 0)
    - drag_force() / lift_force(): 力积分结果
    - f_ibm: (3, nx, ny, nz) 力场 (与源项 IBM 同形状, 供动量源项接入)

对 xvoxel 的所有访问均为只读. 八叉树递归是算法固有结构,
叶节点高斯点处理全向量化 (零 Python for 循环).
"""
import numpy as np
from typing import Optional


# ── 八叉树预计算常数 ──────────────────────────────────────────────
# 8 个子域的偏移向量 (避免递归内重复构建)
_OCTANT_OFFSETS: np.ndarray = np.array(
    [[i, j, k] for i in range(2) for j in range(2) for k in range(2)],
    dtype=np.float64,
)

# 2×2×2 高斯积分规则 (8 点, 与 fcm/elements.py GAUSS_2X2X2 一致)
_GAUSS_1D_PTS = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
_GAUSS_1D_WTS = np.array([1.0, 1.0])
_XI, _ETA, _ZETA = np.meshgrid(_GAUSS_1D_PTS, _GAUSS_1D_PTS, _GAUSS_1D_PTS,
                                indexing='ij')
_GAUSS_POINTS = np.column_stack([_XI.ravel(), _ETA.ravel(), _ZETA.ravel()])
_WXI, _WETA, _WZETA = np.meshgrid(_GAUSS_1D_WTS, _GAUSS_1D_WTS, _GAUSS_1D_WTS,
                                   indexing='ij')
_GAUSS_WEIGHTS = (_WXI * _WETA * _WZETA).ravel()


class SharpIBMHandler:
    """尖锐界面 IBM — 八叉树 + 高斯积分定位界面.

    对 xvoxel 的所有访问均为只读消费.

    Attributes:
        voxel_nature: (n_voxels,) int8 — 独立副本
        dx_min: min(dx, dy, dz)
        nx, ny, nz: 网格维度
        max_depth: 八叉树最大细分深度
        f_ibm: (3, nx, ny, nz) float64 — 界面力场 (供动量源项 + 阻力积分)
        interface_points: (N_int, 3) 界面高斯点物理坐标
        interface_normals: (N_int, 3) 界面法向 (指向流体, 即 ∇Φ/|∇Φ|)
        interface_weights: (N_int,) 界面高斯点面元权重 dS
    """

    def __init__(self, xv, dt: float = 0.01, rho: float = 1.0,
                 max_depth: int = 3, penalty: float = 1.0):
        """初始化尖锐界面 IBM, 预计算界面高斯点.

        Args:
            xv: XVoxelModel 实例 (只读消费).
            dt: 伪瞬态时间步长 (用于直接力公式).
            rho: 流体密度.
            max_depth: 八叉树最大细分深度 (depth=0 不细分, depth=3 细分到 1/8³).
            penalty: 无滑移强制系数 (越大越强, 默认 1.0).
        """
        # 只读消费 xvoxel 数据
        self.voxel_nature = xv.voxel_nature.copy()
        self._centers = xv._voxel_centers
        self._csg_root = xv.csg_root
        self.dx_min = float(min(xv.dx, xv.dy, xv.dz))
        self.nx, self.ny, self.nz = int(xv.nx), int(xv.ny), int(xv.nz)
        self.dx, self.dy, self.dz = float(xv.dx), float(xv.dy), float(xv.dz)
        self.ox, self.oy, self.oz = float(xv.ox), float(xv.oy), float(xv.oz)
        self.dt = float(dt)
        self.rho = float(rho)
        self.max_depth = int(max_depth)
        self.penalty = float(penalty)

        # 界面力存储 (体素中心, 与 IBMForce 同形状, 供动量源项接入)
        self.f_ibm = np.zeros((3, self.nx, self.ny, self.nz), dtype=np.float64)

        # 预计算界面高斯点 (一次性, 永久复用)
        self.interface_points: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.interface_normals: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.interface_weights: np.ndarray = np.zeros(0, dtype=np.float64)
        # 界面点所属体素 (i,j,k) — 用于力散列回体素
        self._interface_ijk: np.ndarray = np.zeros((0, 3), dtype=np.int64)
        self._precompute_interface()

    # ------------------------------------------------------------------
    # 界面预计算 (八叉树 + 高斯积分)
    # ------------------------------------------------------------------
    def _precompute_interface(self) -> None:
        """一次性预计算所有界面高斯点 (坐标 / 法向 / 面元权重).

        对每个边界体素 (voxel_nature == 0) 执行八叉树细分:
            - 叶节点高斯点用 SDF 分类
            - 跨界高斯点 (solid/void 混合) 投影到 Φ=0 界面
            - 界面点法向 n = ∇Φ/|∇Φ| (中心差分)
        """
        if self._csg_root is None:
            return

        # 边界体素索引 (nature == 0)
        cut_ids = np.where(self.voxel_nature == 0)[0]
        if len(cut_ids) == 0:
            return

        pts_list = []
        norms_list = []
        wts_list = []
        ijk_list = []

        for vid in cut_ids:
            i = int(vid % self.nx)
            j = int((vid // self.nx) % self.ny)
            k = int(vid // (self.nx * self.ny))
            # 体素物理范围 [lo, hi]
            lo = np.array([self.ox + i * self.dx,
                           self.oy + j * self.dy,
                           self.oz + k * self.dz])
            hi = np.array([self.ox + (i + 1) * self.dx,
                           self.oy + (j + 1) * self.dy,
                           self.oz + (k + 1) * self.dz])
            self._octree_collect(lo, hi, i, j, k, depth=0,
                                 pts_list=pts_list, norms_list=norms_list,
                                 wts_list=wts_list, ijk_list=ijk_list)

        if len(pts_list) == 0:
            return

        self.interface_points = np.vstack(pts_list)
        self.interface_normals = np.vstack(norms_list)
        self.interface_weights = np.concatenate(wts_list)
        self._interface_ijk = np.vstack(ijk_list)

    def _octree_collect(self, lo: np.ndarray, hi: np.ndarray,
                        i: int, j: int, k: int, depth: int,
                        pts_list: list, norms_list: list,
                        wts_list: list, ijk_list: list) -> None:
        """八叉树递归收集界面高斯点 (移植自 fcm/assembly.py:_octree_integrate).

        终止条件: depth >= max_depth 或子域完全在 solid/void 内.
        叶节点: 高斯点 SDF 分类, 跨界点投影到界面.
        """
        center = 0.5 * (lo + hi)
        sdf_center = self._csg_root.sdf_batch(center.reshape(1, 3))[0]
        sub_half = 0.5 * np.max(hi - lo)  # 子域半尺寸 (用于分类阈值)

        # 子域完全在 solid 内 (sdf < -sub_half) 或 void 内 (sdf > sub_half) → 不含界面
        if sdf_center < -sub_half or sdf_center > sub_half:
            return

        if depth >= self.max_depth:
            # ── 叶节点: 高斯点 SDF 分类 ──
            self._leaf_gauss_collect(lo, hi, i, j, k,
                                     pts_list, norms_list, wts_list, ijk_list)
            return

        # ── 细分: 8 子域 ──
        half = 0.5 * (hi - lo)
        sub_lo_all = lo + _OCTANT_OFFSETS * half  # (8, 3)
        for oct_i in range(8):
            sub_hi = sub_lo_all[oct_i] + half
            self._octree_collect(sub_lo_all[oct_i], sub_hi, i, j, k,
                                 depth + 1, pts_list, norms_list,
                                 wts_list, ijk_list)

    def _leaf_gauss_collect(self, lo: np.ndarray, hi: np.ndarray,
                            i: int, j: int, k: int,
                            pts_list: list, norms_list: list,
                            wts_list: list, ijk_list: list) -> None:
        """叶节点高斯点处理: 分类 + 界面投影 + 法向计算 (向量化)."""
        sub_size = 0.5 * (hi - lo)
        # 高斯点物理坐标: lo + (ξ+1)/2 * (hi-lo)
        phys_pts = lo + 0.5 * (_GAUSS_POINTS + 1.0) * (hi - lo)  # (8, 3)
        sdf_vals = self._csg_root.sdf_batch(phys_pts)            # (8,)

        # 跨界判定: 子域内同时含 solid (sdf<0) 和 void (sdf>0) → 含界面
        has_solid = np.any(sdf_vals < 0)
        has_void = np.any(sdf_vals > 0)
        if not (has_solid and has_void):
            return

        # 界面高斯点: sdf 符号变化附近的点, 投影到 Φ=0
        # 取 |sdf| 最小的若干点作为界面候选 (最接近界面的高斯点)
        # 面元权重: 高斯权重 × 子域体积 / dx_min (近似面元)
        gauss_w = _GAUSS_WEIGHTS * np.prod(sub_size) / self.dx_min

        # 对每个高斯点: 若 |sdf| < sub_half (在界面带内), 投影到界面
        sub_half = float(np.max(sub_size))
        interface_mask = np.abs(sdf_vals) < sub_half
        if not np.any(interface_mask):
            return

        idx = np.where(interface_mask)[0]
        cand_pts = phys_pts[idx]          # (M, 3)
        cand_sdf = sdf_vals[idx]          # (M,)
        cand_w = gauss_w[idx]             # (M,)

        # 投影到界面: x_int = x - sdf * ∇Φ/|∇Φ|² (牛顿步)
        normals = self._compute_normals(cand_pts)   # (M, 3) = ∇Φ/|∇Φ|
        grad_sq = np.sum(normals ** 2, axis=1)      # |∇Φ|² (normals 已归一化 → 1)
        grad_sq = np.maximum(grad_sq, 1e-30)
        # 沿法向回退 sdf 距离到界面
        int_pts = cand_pts - (cand_sdf[:, None] * normals)

        # 重新校验: 投影后 sdf 应接近 0
        sdf_check = self._csg_root.sdf_batch(int_pts)
        valid = np.abs(sdf_check) < sub_half
        if not np.any(valid):
            return

        pts_list.append(int_pts[valid])
        norms_list.append(normals[valid])
        wts_list.append(cand_w[valid])
        n_valid = int(np.sum(valid))
        ijk_list.append(np.tile(np.array([[i, j, k]], dtype=np.int64),
                                (n_valid, 1)))

    def _compute_normals(self, points: np.ndarray) -> np.ndarray:
        """批量计算界面法向 n = ∇Φ/|∇Φ| (中心差分, 向量化).

        Args:
            points: (N, 3) 物理坐标.

        Returns:
            (N, 3) 单位法向 (指向流体, 即 sdf 增大方向).
        """
        h = self.dx_min * 0.5  # 差分步长
        # 6 点中心差分 → 梯度
        pts = points  # (N, 3)
        sdf_pxp = self._csg_root.sdf_batch(pts + np.array([h, 0, 0]))
        sdf_pxm = self._csg_root.sdf_batch(pts - np.array([h, 0, 0]))
        sdf_pyp = self._csg_root.sdf_batch(pts + np.array([0, h, 0]))
        sdf_pym = self._csg_root.sdf_batch(pts - np.array([0, h, 0]))
        sdf_pzp = self._csg_root.sdf_batch(pts + np.array([0, 0, h]))
        sdf_pzm = self._csg_root.sdf_batch(pts - np.array([0, 0, h]))

        grad = np.empty_like(pts)
        grad[:, 0] = (sdf_pxp - sdf_pxm) / (2.0 * h)
        grad[:, 1] = (sdf_pyp - sdf_pym) / (2.0 * h)
        grad[:, 2] = (sdf_pzp - sdf_pzm) / (2.0 * h)

        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-30)
        return grad / norm

    # ------------------------------------------------------------------
    # 界面力计算 (全向量化) — 记录到 f_ibm, 供动量源项 + 阻力积分
    # ------------------------------------------------------------------
    def compute_force(self, grid) -> None:
        """计算界面力并散列到 f_ibm (体素中心).

        界面力 = ρ * (penalty / Δt) * (u_target - u_int) * dS / dV
        其中 u_int 为界面点插值速度, u_target = 0 (无滑移).
        散列到界面点所属体素的 f_ibm, 供动量方程作为源项.

        Args:
            grid: StaggeredGrid 实例 (只读).
        """
        self.f_ibm.fill(0.0)
        n_int = self.interface_points.shape[0]
        if n_int == 0:
            return

        # 界面点速度插值 (三线性, 从 MAC 面心速度)
        u_int = self._interp_velocity(grid, self.interface_points)  # (N, 3)

        # 界面力密度 (单位体积): f = ρ·penalty/Δt · (0 - u_int)
        coef = self.rho * self.penalty / self.dt
        force_density = coef * (0.0 - u_int)  # (N, 3)

        # 散列到体素: f_ibm[:, i, j, k] += force_density * dS / dV
        # dS = interface_weights, dV = dx*dy*dz
        dV = self.dx * self.dy * self.dz
        scale = self.interface_weights / dV  # (N,)
        scaled_force = force_density * scale[:, None]  # (N, 3)

        i = self._interface_ijk[:, 0]
        j = self._interface_ijk[:, 1]
        k = self._interface_ijk[:, 2]
        # np.add.at 处理同一体素多个界面点的累加
        np.add.at(self.f_ibm[0], (i, j, k), scaled_force[:, 0])
        np.add.at(self.f_ibm[1], (i, j, k), scaled_force[:, 1])
        np.add.at(self.f_ibm[2], (i, j, k), scaled_force[:, 2])

    # ------------------------------------------------------------------
    # 无滑移强制 (强形式) — 界面附近面心速度 → 0
    # ------------------------------------------------------------------
    def apply(self, grid) -> None:
        """强形式无滑移: 界面附近固体侧面心速度强制为 0.

        对每个界面点, 找到最近的 u/v/w 面心, 若该面心在固体侧 (sdf<0)
        则强制为 0. 这是尖锐界面的强形式施加 (区别于源项 IBM 的弱形式).

        Args:
            grid: StaggeredGrid 实例 (原地修改 u, v, w).
        """
        n_int = self.interface_points.shape[0]
        if n_int == 0:
            return

        # 界面点附近的固体体素: voxel_nature != -1 的体素面心 → 0
        # (与源项 IBM.apply 一致的策略, 但作用域由界面点精确定位)
        solid_mask = self.voxel_nature != -1  # solid + boundary
        solid_ids = np.where(solid_mask)[0]
        if len(solid_ids) == 0:
            return

        i = solid_ids % self.nx
        j = (solid_ids // self.nx) % self.ny
        k = solid_ids // (self.nx * self.ny)
        i = i.astype(np.int64)
        j = j.astype(np.int64)
        k = k.astype(np.int64)

        # u 面: u[i,j,k] (西) 和 u[i+1,j,k] (东)
        grid.u[i, j, k] = 0.0
        grid.u[i + 1, j, k] = 0.0
        # v 面: v[i,j,k] (南) 和 v[i,j+1,k] (北)
        grid.v[i, j, k] = 0.0
        grid.v[i, j + 1, k] = 0.0
        # w 面: w[i,j,k] (下) 和 w[i,j,k+1] (上)
        grid.w[i, j, k] = 0.0
        grid.w[i, j, k + 1] = 0.0

    # ------------------------------------------------------------------
    # 力积分 — 沿曲面面积分 ∮σ·n dS (尖锐界面核心优势)
    # ------------------------------------------------------------------
    def total_force(self) -> np.ndarray:
        """沿界面面积分得总力 (3,) — 物体所受力.

        F = -Σ f_ibm * ΔV  (与 IBMForce 接口一致, f_ibm 已含界面力密度).

        注: f_ibm 已在 compute_force 中按 dS/dV 缩放, 故此处仍用体积分
        形式 Σ f_ibm·dV, 等价于界面面积分 Σ force_density·dS.

        Returns:
            (3,) float64 — (Fx, Fy, Fz) 物体所受总力.
        """
        dV = self.dx * self.dy * self.dz
        return -self.f_ibm.sum(axis=(1, 2, 3)) * dV

    def drag_force(self) -> float:
        """阻力 (物体所受 x 方向力)."""
        return float(self.total_force()[0])

    def lift_force(self) -> float:
        """升力 (物体所受 y 方向力)."""
        return float(self.total_force()[1])

    # ------------------------------------------------------------------
    # 速度插值 (三线性, MAC 面心 → 任意点)
    # ------------------------------------------------------------------
    def _interp_velocity(self, grid, points: np.ndarray) -> np.ndarray:
        """三线性插值 MAC 面心速度到任意物理点 (向量化).

        Args:
            grid: StaggeredGrid 实例.
            points: (N, 3) 物理坐标.

        Returns:
            (N, 3) 插值速度 (u, v, w).
        """
        n = points.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)

        # 归一化到网格索引 (体素中心 = 整数索引 + 0.5)
        # u 面心在 x = ox + i*dx (i 整数), 故 u 索引 = (x-ox)/dx
        u_idx = (points[:, 0] - self.ox) / self.dx  # (N,)
        v_idx = (points[:, 1] - self.oy) / self.dy
        w_idx = (points[:, 2] - self.oz) / self.dz

        u_int = self._interp_scalar_face(grid.u, u_idx,
                                          (points[:, 1] - self.oy) / self.dy - 0.5,
                                          (points[:, 2] - self.oz) / self.dz - 0.5)
        v_int = self._interp_scalar_face(grid.v,
                                          (points[:, 0] - self.ox) / self.dx - 0.5,
                                          v_idx,
                                          (points[:, 2] - self.oz) / self.dz - 0.5)
        w_int = self._interp_scalar_face(grid.w,
                                          (points[:, 0] - self.ox) / self.dx - 0.5,
                                          (points[:, 1] - self.oy) / self.dy - 0.5,
                                          w_idx)
        return np.column_stack([u_int, v_int, w_int])

    def _interp_scalar_face(self, field: np.ndarray,
                            idx0: np.ndarray, idx1: np.ndarray,
                            idx2: np.ndarray) -> np.ndarray:
        """三线性插值标量场 (向量化, 边界裁剪).

        field 形状 (n0, n1, n2), idx0/1/2 为连续索引.
        """
        i0 = np.floor(idx0).astype(np.int64)
        j0 = np.floor(idx1).astype(np.int64)
        k0 = np.floor(idx2).astype(np.int64)
        fx = idx0 - i0
        fy = idx1 - j0
        fz = idx2 - k0

        n0, n1, n2 = field.shape
        i0 = np.clip(i0, 0, n0 - 2)
        j0 = np.clip(j0, 0, n1 - 2)
        k0 = np.clip(k0, 0, n2 - 2)
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        c000 = field[i0, j0, k0]
        c001 = field[i0, j0, k1]
        c010 = field[i0, j1, k0]
        c011 = field[i0, j1, k1]
        c100 = field[i1, j0, k0]
        c101 = field[i1, j0, k1]
        c110 = field[i1, j1, k0]
        c111 = field[i1, j1, k1]

        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz
