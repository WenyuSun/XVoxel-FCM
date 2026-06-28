# -*- coding: utf-8 -*-
"""
postprocess.py — 流体后处理 (全向量化)

提供 MVP 验证所需的纯函数接口:
    - 阻力/升力系数 (IBM 力积分)
    - 涡量场 ω_z
    - 尾迹中心线速度分布
    - 横向速度剖面
    - 回流区长度 L_r
    - 分离角 θ_s
    - 速度幅值云图
"""
import numpy as np


# ======================================================================
# 力系数
# ======================================================================
def compute_drag_coefficient(ibm, rho: float, U_inf: float,
                             D: float = 1.0, W: float = 1.0,
                             dx_min: float = None) -> float:
    """由 IBM 体素上的 f_ibm,x 积分计算 C_d.

    C_d = F_drag / (0.5 * ρ * U_inf² * D * W)
    F_drag = Σ f_ibm,x * ΔV

    Args:
        ibm: IBMForce 实例 (含 f_ibm 力场).
        rho: 密度.
        U_inf: 来流速度.
        D: 特征长度 (圆柱直径).
        W: 展宽 (z 方向长度).
        dx_min: 体素最小尺寸 (用于 ΔV).

    Returns:
        阻力系数 C_d.
    """
    if dx_min is None:
        dx_min = ibm.dx_min
    dV = dx_min * dx_min * dx_min
    # f_ibm 是施加在流体上的力 (将流体减速到 0).
    # 阻力 = 流体施加在物体上的力 = -f_ibm (牛顿第三定律).
    F_drag = -float(ibm.f_ibm[0].sum() * dV)
    denom = 0.5 * rho * U_inf * U_inf * D * W
    if denom == 0:
        return 0.0
    return F_drag / denom


def compute_lift_coefficient(ibm, rho: float, U_inf: float,
                             D: float = 1.0, W: float = 1.0,
                             dx_min: float = None) -> float:
    """由 IBM 体素上的 f_ibm,y 积分计算 C_l (用于确认对称性)."""
    if dx_min is None:
        dx_min = ibm.dx_min
    dV = dx_min * dx_min * dx_min
    # 升力 = 流体施加在物体上的力 = -f_ibm,y
    F_lift = -float(ibm.f_ibm[1].sum() * dV)
    denom = 0.5 * rho * U_inf * U_inf * D * W
    if denom == 0:
        return 0.0
    return F_lift / denom


# ======================================================================
# 涡量场
# ======================================================================
def compute_vorticity_z(u: np.ndarray, v: np.ndarray,
                        dx: float, dy: float) -> np.ndarray:
    """全向量化中心差分涡量 ω_z = ∂v/∂x - ∂u/∂y.

    在体素角点 (i, j) 处计算 (面心速度差分到角点).

    Args:
        u: (nx+1, ny, nz) x-面心速度.
        v: (nx, ny+1, nz) y-面心速度.
        dx, dy: 网格间距.

    Returns:
        (nx+1, ny+1, nz) float64 — 角点涡量.
    """
    # v 差分到角点: ∂v/∂x 在角点 (i,j) = (v[i,j] - v[i-1,j]) / dx
    # v 形状 (nx, ny+1, nz). 角点 (i,j) i∈[0,nx], j∈[0,ny]
    # v[i,j] 在 (i+0.5, j). 角点 (i,j) 的 ∂v/∂x = (v[i,j]-v[i-1,j])/dx
    dvdx = np.zeros((u.shape[0], v.shape[1], u.shape[2]), dtype=np.float64)
    dvdx[1:-1, :, :] = (v[1:, :, :] - v[:-1, :, :]) / dx

    # u 差分到角点: ∂u/∂y 在角点 (i,j) = (u[i,j] - u[i,j-1]) / dy
    dudy = np.zeros((u.shape[0], v.shape[1], u.shape[2]), dtype=np.float64)
    dudy[:, 1:-1, :] = (u[:, 1:, :] - u[:, :-1, :]) / dy

    omega_z = dvdx - dudy
    return omega_z


def compute_vorticity_z_cell(u: np.ndarray, v: np.ndarray,
                             dx: float, dy: float) -> np.ndarray:
    """体素中心涡量 (用于云图). 形状 (nx, ny, nz).

    ω_z(i,j) = (v[i+1,j] - v[i,j])/dx - (u[i,j+1] - u[i,j])/dy
    其中 v, u 取体素界面值.
    """
    nx = u.shape[0] - 1
    ny = v.shape[1] - 1
    nz = u.shape[2]
    # 体素中心 ω_z: 用体素界面速度差分
    # ∂v/∂x 在体素中心 = (v[i+1,j] - v[i,j])/dx, v 在 y-面心, 需插值到体素中心 y
    # v[i,j,k] 在 (i+0.5, j, k+0.5). 体素中心 (i+0.5, j+0.5, k+0.5)
    # v 在体素中心 = 0.5*(v[i,j]+v[i,j+1])
    v_cell = 0.5 * (v[:, :-1, :] + v[:, 1:, :])   # (nx, ny, nz)
    dvdx = np.zeros((nx, ny, nz), dtype=np.float64)
    # ∂v/∂x 在体素中心 (i,j) = (v_cell[i+1,j] - v_cell[i,j])/dx
    # v_cell 形状 (nx,ny,nz); 内部点 i∈[1,nx-2] 用 v_cell[i+1]-v_cell[i]
    dvdx[1:-1, :, :] = (v_cell[2:, :, :] - v_cell[1:-1, :, :]) / dx

    u_cell = 0.5 * (u[:-1, :, :] + u[1:, :, :])   # (nx, ny, nz)
    dudy = np.zeros((nx, ny, nz), dtype=np.float64)
    dudy[:, 1:-1, :] = (u_cell[:, 2:, :] - u_cell[:, 1:-1, :]) / dy

    return dvdx - dudy


# ======================================================================
# 尾迹提取
# ======================================================================
def extract_wake_centerline(u: np.ndarray, y_mid_idx: int, nz: int,
                            x_start_idx: int = 0) -> np.ndarray:
    """提取尾迹中心线 u 分布 — 全切片操作.

    沿 y = y_mid_idx, 对 z 方向平均.

    Args:
        u: (nx+1, ny, nz) x-面心速度.
        y_mid_idx: 中心线 y 索引.
        nz: z 方向体素数.
        x_start_idx: 起始 x 索引.

    Returns:
        (nx+1 - x_start_idx,) float64 — 中心线 u/U_inf (未归一化).
    """
    return u[x_start_idx:, y_mid_idx, :].mean(axis=1)


def extract_transverse_profile(u: np.ndarray, x_idx: int, nz: int) -> np.ndarray:
    """提取给定 x 截面的横向速度剖面 — 全切片操作.

    Args:
        u: (nx+1, ny, nz) x-面心速度.
        x_idx: x 截面索引.
        nz: z 方向体素数.

    Returns:
        (ny,) float64 — 横向 u 分布 (z 平均).
    """
    return u[x_idx, :, :].mean(axis=1)


# ======================================================================
# 尾迹拓扑
# ======================================================================
def compute_wake_length(u: np.ndarray, y_mid_idx: int, nz: int,
                        x_start: float, cx: float, R: float,
                        dx: float) -> float:
    """确定回流区长度 L_r.

    从圆柱后缘 (x = cx + R) 沿中心线向后搜索首个 u > 0 的网格面.

    Args:
        u: (nx+1, ny, nz) x-面心速度.
        y_mid_idx: 中心线 y 索引.
        nz: z 方向体素数.
        x_start: 域起始 x 坐标.
        cx: 圆柱中心 x.
        R: 圆柱半径.
        dx: x 方向网格间距.

    Returns:
        L_r (长度单位). 若无回流返回 0.
    """
    # 中心线 u (z 平均)
    u_cl = u[:, y_mid_idx, :].mean(axis=1)
    # 圆柱后缘索引
    x_back = cx + R
    i_back = int((x_back - x_start) / dx)
    if i_back >= len(u_cl) - 1:
        return 0.0

    # 从后缘向后搜索首个 u > 0
    tail = u_cl[i_back:]
    positive = np.where(tail > 0)[0]
    if len(positive) == 0:
        # 整个尾迹都为负 → 回流区超出域
        return float(len(tail) - 1) * dx
    first_pos = positive[0]
    L_r = float(first_pos) * dx
    return L_r


def compute_separation_angle(u: np.ndarray, v: np.ndarray,
                             cx: float, cy: float, R: float,
                             grid, n_theta: int = 360) -> float:
    """通过近壁体素速度方向确定分离角 θ_s.

    在圆柱表面采样 n_theta 个角度, 提取最近流体体素的速度,
    检测切向速度方向反转点 (分离点).

    Args:
        u, v: 速度场.
        cx, cy: 圆柱中心.
        R: 圆柱半径.
        grid: StaggeredGrid.
        n_theta: 角度采样数.

    Returns:
        分离角 (度, 从前驻点 0° 顺时针).
    """
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    # 表面采样点 (略大于 R 以落在流体区)
    r_sample = R + 0.6 * min(grid.dx, grid.dy)
    xs = cx + r_sample * np.cos(thetas)
    ys = cy + r_sample * np.sin(thetas)

    # 转换为网格索引
    i_idx = ((xs - grid.ox) / grid.dx).astype(int)
    j_idx = ((ys - grid.oy) / grid.dy).astype(int)
    i_idx = np.clip(i_idx, 1, grid.nx - 2)
    j_idx = np.clip(j_idx, 0, grid.ny - 1)
    k_idx = grid.nz // 2

    # 体素中心速度 (近似)
    u_cell = 0.5 * (u[i_idx, j_idx, k_idx] + u[i_idx + 1, j_idx, k_idx])
    v_cell = 0.5 * (v[i_idx, j_idx, k_idx] + v[i_idx, j_idx + 1, k_idx])

    # 切向速度 (沿 θ 方向): u_t = -u*sin(θ) + v*cos(θ)
    u_tan = -u_cell * np.sin(thetas) + v_cell * np.cos(thetas)

    # 分离点: 上半圆 (θ∈[0,π]) 切向速度由正变负
    # 前驻点 θ=0. 分离角 = θ_s
    half = n_theta // 2
    u_tan_half = u_tan[:half]
    # 找首个由正变负的过零点
    sign_change = np.where(np.diff(np.sign(u_tan_half)) < 0)[0]
    if len(sign_change) == 0:
        return 90.0  # 默认
    theta_s = thetas[sign_change[0]]
    return float(np.degrees(theta_s))


# ======================================================================
# 涡心位置
# ======================================================================
def compute_vortex_center(omega_z: np.ndarray, grid, cx: float, cy: float,
                          R: float, upper: bool = True) -> tuple:
    """提取尾迹涡心位置 (上涡或下涡).

    涡心 = |ω_z| 极大值点 (在圆柱下游 0.5R ~ 5R 范围内).

    Args:
        omega_z: (nx, ny, nz) 体素中心涡量.
        grid: StaggeredGrid.
        cx, cy: 圆柱中心.
        R: 圆柱半径.
        upper: True=上涡 (ω_z < 0), False=下涡 (ω_z > 0).

    Returns:
        (x_vortex, y_vortex) 涡心坐标.
    """
    # 搜索区域: x ∈ [cx+0.5R, cx+6R], y 偏离中心
    x_min = cx + 0.5 * R
    x_max = cx + 6.0 * R
    i_min = max(0, int((x_min - grid.ox) / grid.dx))
    i_max = min(grid.nx, int((x_max - grid.ox) / grid.dx))
    j_mid = int((cy - grid.oy) / grid.dy)

    if upper:
        j_range = slice(j_mid, min(grid.ny, j_mid + int(3 * R / grid.dy)))
        sub = omega_z[i_min:i_max, j_range, :]
    else:
        j_range = slice(max(0, j_mid - int(3 * R / grid.dy)), j_mid)
        sub = omega_z[i_min:i_max, j_range, :]

    sub_mean = sub.mean(axis=2)  # z 平均
    if upper:
        # 上涡 ω_z < 0, 找最小值
        idx = np.unravel_index(np.argmin(sub_mean), sub_mean.shape)
    else:
        # 下涡 ω_z > 0, 找最大值
        idx = np.unravel_index(np.argmax(sub_mean), sub_mean.shape)

    i_v = i_min + idx[0]
    j_v = j_range.start + idx[1]
    x_v = grid.ox + (i_v + 0.5) * grid.dx
    y_v = grid.oy + (j_v + 0.5) * grid.dy
    return (float(x_v), float(y_v))


# ======================================================================
# 对称性度量
# ======================================================================
def compute_asymmetry(field: np.ndarray, j_mid: int) -> float:
    """计算场关于 y=j_mid 的最大不对称度.

    不对称度 = max|field(y) - field(2*j_mid - y)| / max|field|

    Args:
        field: (nx, ny, nz) 或 (nx, ny) 场.
        j_mid: 中心线 y 索引.

    Returns:
        不对称度 (0 = 完全对称).
    """
    if field.ndim == 3:
        f = field.mean(axis=2)
    else:
        f = field
    ny = f.shape[1]
    upper = f[:, j_mid:ny]
    lower = f[:, j_mid::-1][:, :ny - j_mid]
    n = min(upper.shape[1], lower.shape[1])
    diff = upper[:, :n] - lower[:, :n]
    max_diff = float(np.max(np.abs(diff)))
    max_field = float(np.max(np.abs(f)))
    if max_field == 0:
        return 0.0
    return max_diff / max_field
