# -*- coding: utf-8 -*-
"""
cylinder_flow_analysis.py — 圆柱绕流 MVP 详细分析 (论文级)

计算方案 v3.1 第十二章定义的全部五类验证指标:
    A. 全局力系数 (Cd, Cl)
    B. 尾迹拓扑 (回流区长度 L_r/D, 分离角 θ_s)
    C. 尾迹中心线速度分布 (u/U_inf vs x/D)
    D. 横向速度剖面 (u/U_inf vs y/D @ x/D = 1, 2, 4)
    E. 涡量场云图 (ω_z 定性结构)

输出:
    figures/paper_*.png           — 论文级图件
    output/cylinder_analysis.json — 全部数值结果 (供报告引用)
"""
import os
import sys
import json
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from xvoxel.xvoxel import XVoxelModel
from xvoxel.primitives import CylinderZ
from xvoxel.csg import BoolOp

from fluid.solver import FluidSolver
from fluid.postprocess import (
    compute_vorticity_z_cell,
    compute_wake_length,
    compute_separation_angle,
    compute_vortex_center,
    compute_asymmetry,
)

FIG_DIR = os.path.join(_REPO_ROOT, 'figures')
OUT_DIR = os.path.join(_REPO_ROOT, 'output')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 论文级绘图风格
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


# ======================================================================
#  文献基准数据
# ======================================================================

# 阻力系数 Cd 文献值
LITERATURE_CD = {
    20: {
        'Tritton_1959_exp': 2.01,      # Tritton 实验 (±0.05)
        'Fornberg_1980': 2.00,         # Fornberg 谱方法
        'Dennis_Chang_1970': 2.04,     # Dennis & Chang 差分
        'range': (1.96, 2.10),         # 方案目标区间 (±5%)
    },
    40: {
        'Tritton_1959_exp': 1.48,      # Tritton 实验 (±0.05)
        'Fornberg_1980': 1.50,         # Fornberg 谱方法
        'Dennis_Chang_1970': 1.52,     # Dennis & Chang 差分
        'range': (1.43, 1.60),         # 方案目标区间 (±5%)
    },
}

# 回流区长度 L_r/D 文献值
LITERATURE_LR = {
    20: {
        'Coutanceau_Bouard_1977_exp': 0.90,   # ±0.05
        'Dennis_Chang_1970': 0.93,
        'range': (0.85, 0.98),
    },
    40: {
        'Coutanceau_Bouard_1977_exp': 2.13,   # ±0.05
        'Dennis_Chang_1970': 2.13,
        'Fornberg_1980': 2.24,
        'range': (2.00, 2.35),
    },
}

# 分离角 θ_s (度) 文献值
LITERATURE_THETA = {
    20: {
        'Coutanceau_Bouard_1977_exp': 44.5,   # ±1
        'Dennis_Chang_1970': 43.7,
        'range': (41.0, 46.0),
    },
    40: {
        'Coutanceau_Bouard_1977_exp': 53.5,   # ±1
        'Dennis_Chang_1970': 53.6,
        'Fornberg_1980': 52.9,
        'range': (51.0, 55.0),
    },
}

# 中心线速度文献数据 (Fornberg 1980, Coutanceau & Bouard 1977)
# 关键特征值: min(u/U_inf) 及其位置 x/D
LITERATURE_CENTERLINE = {
    20: {
        'min_u_over_U': -0.02,
        'min_u_x_over_D': 1.5,
        'range_min_u': (-0.03, -0.01),
        'range_x': (1.3, 1.7),
    },
    40: {
        'min_u_over_U': -0.10,
        'min_u_x_over_D': 1.2,
        'range_min_u': (-0.13, -0.07),
        'range_x': (1.0, 1.4),
    },
}

# 涡心位置文献值 (Coutanceau & Bouard 1977, Fig. 5c/9c)
LITERATURE_VORTEX_CENTER = {
    20: {'x_over_D': 1.8, 'tol_pct': 15.0},
    40: {'x_over_D': 3.0, 'tol_pct': 15.0},
}

# 横向剖面文献数据 (Coutanceau & Bouard 1977)
# 关键特征值: 中心线速度 u(0)/U_inf
LITERATURE_TRANSVERSE = {
    20: {
        'x/D=1.0': {'u_cl': -0.02, 'half_width': 0.8},
        'x/D=2.0': {'u_cl': 0.05, 'half_width': 1.1},
    },
    40: {
        'x/D=1.0': {'u_cl': -0.09},
        'x/D=2.0': {'u_cl': -0.04},
    },
}


# ======================================================================
#  求解
# ======================================================================

def solve_cylinder(Re: float, nx: int = 120, ny: int = 80,
                   max_iter: int = 500, verbose: bool = True,
                   method: str = 'ibm'):
    """求解圆柱绕流并返回完整结果对象.

    Args:
        Re: 雷诺数.
        nx, ny: 网格分辨率.
        max_iter: 最大迭代.
        verbose: 打印收敛历史.
        method: 边界处理方法 — 'ibm' (源项体积力, 默认) 或
                'sharp' (尖锐界面, 八叉树+高斯积分).

    Returns:
        dict 含 solver, u, v, p, grid, div_history, 耗时.
    """
    print(f"\n[求解] Re={Re}, 网格 {nx}x{ny}x4, max_iter={max_iter}, "
          f"method={method}")
    # 域 6×4×0.4, 圆柱 D=0.8 (R=0.4) 中心 (1.5, 2.0)
    xv = XVoxelModel(nx, ny, 4, 6.0, 4.0, 0.4)
    cyl = CylinderZ(1.5, 2.0, 0.4, -1.0, 2.0)
    xv.add_feature(cyl, op=BoolOp.UNION)

    solver = FluidSolver(xv, Re=Re, U_inf=1.0, boundary_method=method)
    solver.add_inlet_bc('xmin', (1.0, 0.0, 0.0))
    solver.add_outlet_bc('xmax', p=0.0)
    solver.add_slip_wall('ymin')
    solver.add_slip_wall('ymax')
    solver.add_slip_wall('zmin')
    solver.add_slip_wall('zmax')
    solver.set_solver_params(dt=0.05, alpha_p=0.2, alpha_u=0.7)

    t0 = time.time()
    u, v, p = solver.solve(max_iter=max_iter, tol=1e-4,
                           momentum_iter=20, pressure_iter=100,
                           verbose=verbose)
    dt = time.time() - t0

    return {
        'Re': Re, 'solver': solver, 'u': u, 'v': v, 'p': p,
        'grid': solver.grid, 'xv': xv,
        'div_history': solver._simple.div_history,
        'time': dt,
        'cx': 1.5, 'cy': 2.0, 'R': 0.4, 'D': 0.8,
    }


# ======================================================================
#  指标计算
# ======================================================================

def compute_all_metrics(res: dict) -> dict:
    """计算全部五类指标.

    Args:
        res: solve_cylinder 返回的结果字典.

    Returns:
        metrics 字典.
    """
    Re = res['Re']
    u, v, p = res['u'], res['v'], res['p']
    grid = res['grid']
    solver = res['solver']
    cx, cy, R, D = res['cx'], res['cy'], res['R'], res['D']

    m = {'Re': Re}

    # ---- A. 全局力系数 ----
    m['Cd'] = float(solver.compute_drag_coefficient(D=D))
    m['Cl'] = float(solver.compute_lift_coefficient(D=D))
    m['div_max'] = float(np.max(np.abs(grid.divergence())))

    # ---- B. 尾迹拓扑 ----
    y_mid_idx = grid.ny // 2
    # 中心线 u (z 平均)
    u_cl = u[:, y_mid_idx, :].mean(axis=1)
    x_faces = grid.ox + np.arange(grid.nx + 1) * grid.dx
    # 圆柱后缘 x = cx + R
    x_back = cx + R
    # 回流区: 从后缘向后, u<0 的区域; L_r = (u 恢复为正的 x) - x_back
    # 找后缘之后首个 u > 0 的点 (跳过紧邻固体的近零区)
    mask_after = x_faces > x_back + 0.05 * D
    x_after = x_faces[mask_after]
    u_after = u_cl[mask_after]
    # 回流区 = u < 0 的连续段; L_r = 最后一个 u<0 点到后缘
    neg_mask = u_after < -1e-4
    if np.any(neg_mask):
        last_neg_idx = np.where(neg_mask)[0][-1]
        L_r = float(x_after[last_neg_idx] - x_back)
    else:
        L_r = 0.0
    m['L_r'] = L_r
    m['L_r_over_D'] = L_r / D
    m['theta_s'] = float(compute_separation_angle(
        u, v, cx, cy, R, grid, n_theta=360))

    # ---- C. 中心线速度分布 ----
    u_cl = u[:, y_mid_idx, :].mean(axis=1)  # (nx+1,)
    x_faces = grid.ox + np.arange(grid.nx + 1) * grid.dx
    m['centerline'] = {
        'x_over_D': ((x_faces - cx) / D).tolist(),
        'u_over_Uinf': (u_cl / 1.0).tolist(),
    }

    # ---- D. 横向速度剖面 @ x/D = 1, 2, 4 ----
    # x/D 相对圆柱中心; x_idx 对应 u 面索引
    profiles = {}
    for xd_target in [1.0, 2.0, 4.0]:
        x_phys = cx + xd_target * D
        x_idx = int(round((x_phys - grid.ox) / grid.dx))
        x_idx = max(0, min(x_idx, grid.nx))
        u_prof = u[x_idx, :, :].mean(axis=1)  # (ny,)
        y_faces = grid.oy + np.arange(grid.ny) * grid.dy
        profiles[f'x/D={xd_target}'] = {
            'y_over_D': ((y_faces - cy) / D).tolist(),
            'u_over_Uinf': (u_prof / 1.0).tolist(),
        }
    m['transverse'] = profiles

    # ---- E. 涡量场 ----
    omega_z = compute_vorticity_z_cell(u, v, grid.dx, grid.dy)
    m['omega_z_max'] = float(np.max(np.abs(omega_z)))
    # 涡心位置
    try:
        xv_up = compute_vortex_center(omega_z, grid, cx, cy, R, upper=True)
        xv_lo = compute_vortex_center(omega_z, grid, cx, cy, R, upper=False)
        m['vortex_upper'] = xv_up
        m['vortex_lower'] = xv_lo
    except Exception:
        m['vortex_upper'] = None
        m['vortex_lower'] = None

    # 对称性度量
    m['asymmetry_u'] = float(compute_asymmetry(u, y_mid_idx))
    m['asymmetry_omega'] = float(compute_asymmetry(omega_z, y_mid_idx))

    # 收敛历史 (供对比图件使用)
    m['div_history'] = [float(x) for x in res['div_history']]

    return m


# ======================================================================
#  论文级图件
# ======================================================================

def plot_convergence(res: dict, ax=None):
    """图: 收敛历史 (散度 vs 迭代)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    div_hist = res['div_history']
    iters = np.arange(len(div_hist))
    ax.semilogy(iters, div_hist, 'b-', linewidth=1.5, label=f"Re={res['Re']}")
    ax.set_xlabel('SIMPLE iteration')
    ax.set_ylabel(r'$\max|\nabla \cdot \mathbf{u}|$')
    ax.set_title(f'Convergence history (Re={res["Re"]})')
    ax.legend()
    return ax


def plot_flow_field(res: dict, metrics: dict, ax=None):
    """图: 速度幅值 + 流线 + 圆柱."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    u, v = res['u'], res['v']
    grid = res['grid']
    cx, cy, R = res['cx'], res['cy'], res['R']

    # cell 中心速度
    u_c = 0.5 * (u[:-1, :, 0] + u[1:, :, 0])
    v_c = 0.5 * (v[:, :-1, 0] + v[:, 1:, 0])
    speed = np.sqrt(u_c ** 2 + v_c ** 2)

    x = grid.ox + (np.arange(grid.nx) + 0.5) * grid.dx
    y = grid.oy + (np.arange(grid.ny) + 0.5) * grid.dy

    im = ax.pcolormesh(x, y, speed.T, shading='auto',
                       cmap='viridis', vmin=0, vmax=1.3)
    ax.streamplot(x, y, u_c.T, v_c.T, color='white',
                  density=1.2, linewidth=0.8, arrowsize=1.0)
    ax.add_patch(Circle((cx, cy), R, color='black', zorder=5))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Flow field (Re={res["Re"]}, '
                 f'Cd={metrics["Cd"]:.3f})')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=r'$|\mathbf{u}|/U_\infty$')
    return ax


def plot_pressure_field(res: dict, metrics: dict, ax=None):
    """图: 压力场."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    p = res['p']
    grid = res['grid']
    cx, cy, R = res['cx'], res['cy'], res['R']

    x = grid.ox + (np.arange(grid.nx) + 0.5) * grid.dx
    y = grid.oy + (np.arange(grid.ny) + 0.5) * grid.dy
    im = ax.pcolormesh(x, y, p[:, :, 0].T, shading='auto', cmap='coolwarm')
    ax.add_patch(Circle((cx, cy), R, color='black', zorder=5))
    # 标注驻点高压区
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Pressure field (Re={res["Re"]})')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='p')
    return ax


def plot_vorticity(res: dict, metrics: dict, ax=None):
    """图: 涡量场 ω_z."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    u, v = res['u'], res['v']
    grid = res['grid']
    cx, cy, R = res['cx'], res['cy'], res['R']

    omega_z = compute_vorticity_z_cell(u, v, grid.dx, grid.dy)[:, :, 0]
    wmax = metrics['omega_z_max']
    x = grid.ox + (np.arange(grid.nx) + 0.5) * grid.dx
    y = grid.oy + (np.arange(grid.ny) + 0.5) * grid.dy
    im = ax.pcolormesh(x, y, omega_z.T, shading='auto', cmap='RdBu_r',
                       vmin=-wmax, vmax=wmax)
    ax.add_patch(Circle((cx, cy), R, color='black', zorder=5))
    # 标注涡心
    if metrics.get('vortex_upper'):
        xv_u = metrics['vortex_upper']
        ax.plot(xv_u[0], xv_u[1], 'g^', markersize=8, zorder=6)
    if metrics.get('vortex_lower'):
        xv_l = metrics['vortex_lower']
        ax.plot(xv_l[0], xv_l[1], 'gv', markersize=8, zorder=6)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Vorticity $\\omega_z$ (Re={res["Re"]})')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=r'$\omega_z$')
    return ax


def plot_centerline(res: dict, metrics: dict, ax=None):
    """图: 中心线速度 u/U_inf vs x/D + 文献对照."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    cl = metrics['centerline']
    xD = np.array(cl['x_over_D'])
    uU = np.array(cl['u_over_Uinf'])
    ax.plot(xD, uU, 'b-', linewidth=2, label=f'Present (Re={res["Re"]})')

    # 文献关键特征值对照 (min u/U_inf 及其位置)
    Re = res['Re']
    if Re in LITERATURE_CENTERLINE:
        lit = LITERATURE_CENTERLINE[Re]
        ax.plot(lit['min_u_x_over_D'], lit['min_u_over_U'], 'r*',
                markersize=14, markeredgecolor='k',
                label=f"Lit. min $u/U_\\infty$={lit['min_u_over_U']:.2f} "
                      f"@ $x/D$={lit['min_u_x_over_D']:.1f}")
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8,
               label='cylinder trailing edge')
    ax.set_xlabel(r'$x/D$ (from cylinder center)')
    ax.set_ylabel(r'$u/U_\infty$')
    ax.set_title(f'Wake centerline velocity (Re={res["Re"]})')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.5, 6)
    return ax


def plot_transverse(res: dict, metrics: dict, ax=None):
    """图: 横向速度剖面 @ x/D=1,2,4."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'x/D=1.0': 'b', 'x/D=2.0': 'r', 'x/D=4.0': 'g'}
    for key, prof in metrics['transverse'].items():
        yD = np.array(prof['y_over_D'])
        uU = np.array(prof['u_over_Uinf'])
        ax.plot(yD, uU, '-', color=colors.get(key, 'k'),
                linewidth=2, label=f'Present {key}')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(-0.5, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel(r'$y/D$ (from cylinder center)')
    ax.set_ylabel(r'$u/U_\infty$')
    ax.set_title(f'Transverse velocity profiles (Re={res["Re"]})')
    ax.legend()
    return ax


def plot_wake_topology(res: dict, metrics: dict, ax=None):
    """图: 尾迹拓扑 (回流区标注)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    u = res['u']
    grid = res['grid']
    cx, cy, R, D = res['cx'], res['cy'], res['R'], res['D']

    y_mid = grid.ny // 2
    u_c = 0.5 * (u[:-1, y_mid, 0] + u[1:, y_mid, 0])
    x_c = grid.ox + (np.arange(grid.nx) + 0.5) * grid.dx

    ax.plot((x_c - cx) / D, u_c, 'b-', linewidth=2, label='u centerline')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', label='trailing edge')
    # 标注回流区
    Lr = metrics['L_r_over_D']
    ax.axvspan(0.5, 0.5 + Lr, alpha=0.2, color='red',
               label=f'recirculation $L_r/D$={Lr:.2f}')
    ax.set_xlabel(r'$x/D$')
    ax.set_ylabel(r'$u/U_\infty$')
    ax.set_title(f'Wake topology (Re={res["Re"]}, '
                 r'$L_r/D$='f'{Lr:.2f}, '
                 r'$\theta_s$='f'{metrics["theta_s"]:.1f}°)')
    ax.legend()
    ax.set_xlim(-0.5, 5)
    return ax


# ======================================================================
#  主流程
# ======================================================================

def main(method: str = 'ibm'):
    """运行圆柱绕流分析.

    Args:
        method: 边界处理方法 — 'ibm' (源项体积力) 或
                'sharp' (尖锐界面, 八叉树+高斯积分).
    """
    print("=" * 72)
    print("圆柱绕流 MVP 详细分析 (论文级)")
    print("方案 v3.1 第十二章 — 五类验证指标")
    print(f"边界方法: {method} "
          f"({'源项体积力 IBM' if method == 'ibm' else '尖锐界面 IBM (八叉树+高斯积分)'})")
    print("=" * 72)

    all_metrics = {}

    for Re in [40, 20]:
        res = solve_cylinder(Re=Re, nx=120, ny=80, max_iter=500,
                             verbose=True, method=method)
        metrics = compute_all_metrics(res)
        all_metrics[f'Re{Re}'] = metrics

        # 打印指标汇总
        print(f"\n[Re={Re} 指标汇总]")
        print(f"  A. Cd = {metrics['Cd']:.4f}, Cl = {metrics['Cl']:.4f}")
        print(f"  A. max|div| = {metrics['div_max']:.4f}")
        print(f"  B. L_r/D = {metrics['L_r_over_D']:.4f}")
        print(f"  B. theta_s = {metrics['theta_s']:.2f} deg")
        print(f"  E. |omega_z|_max = {metrics['omega_z_max']:.4f}")
        print(f"  E. asymmetry(u) = {metrics['asymmetry_u']:.4f}")
        if metrics.get('vortex_upper'):
            print(f"  E. vortex_upper = {metrics['vortex_upper']}")
            print(f"  E. vortex_lower = {metrics['vortex_lower']}")

        # 生成图件
        print(f"\n[Re={Re} 生成图件]")

        # 图1: 流场总览 (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        plot_flow_field(res, metrics, ax=axes[0, 0])
        plot_pressure_field(res, metrics, ax=axes[0, 1])
        plot_vorticity(res, metrics, ax=axes[1, 0])
        plot_wake_topology(res, metrics, ax=axes[1, 1])
        fig.suptitle(f'Cylinder cross-flow analysis (Re={Re}, {method})',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR,
                    f'paper_flowfield_Re{Re}_{method}.png'))
        plt.close(fig)
        print(f"    -> paper_flowfield_Re{Re}_{method}.png")

        # 图2: 速度剖面 (中心线 + 横向)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        plot_centerline(res, metrics, ax=axes[0])
        plot_transverse(res, metrics, ax=axes[1])
        fig.suptitle(f'Velocity profiles (Re={Re}, {method})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR,
                    f'paper_profiles_Re{Re}_{method}.png'))
        plt.close(fig)
        print(f"    -> paper_profiles_Re{Re}_{method}.png")

        # 图3: 收敛历史
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_convergence(res, ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR,
                    f'paper_convergence_Re{Re}_{method}.png'))
        plt.close(fig)
        print(f"    -> paper_convergence_Re{Re}_{method}.png")

    # Cd 对比图 (两个 Re + 文献)
    fig, ax = plt.subplots(figsize=(8, 5))
    Re_vals = [20, 40]
    cd_present = [all_metrics[f'Re{r}']['Cd'] for r in Re_vals]
    ax.plot(Re_vals, cd_present, 'bs-', markersize=10, linewidth=2,
            label=f'Present ({method})')
    # 文献范围
    for r in Re_vals:
        lo, hi = LITERATURE_CD[r]['range']
        ax.plot([r, r], [lo, hi], 'r-', linewidth=3, alpha=0.5,
                label='Literature range' if r == 20 else '')
        # 文献点
        for name, val in LITERATURE_CD[r].items():
            if name == 'range':
                continue
            ax.plot(r, val, 'r^', markersize=7, alpha=0.8)
    ax.set_xlabel('Reynolds number Re')
    ax.set_ylabel(r'Drag coefficient $C_d$')
    ax.set_title(f'Drag coefficient vs Reynolds number ({method})')
    ax.legend()
    ax.set_xticks(Re_vals)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'paper_cd_comparison_{method}.png'))
    plt.close(fig)
    print(f"\n[全局] -> paper_cd_comparison_{method}.png")

    # 保存全部指标到 JSON
    out_path = os.path.join(OUT_DIR, f'cylinder_analysis_{method}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[全局] 指标已保存: {out_path}")

    # 汇总表
    print("\n" + "=" * 72)
    print("五类指标汇总")
    print("=" * 72)
    print(f"{'指标':<28} {'Re=20':<18} {'Re=40':<18} {'文献(Re=40)':<18}")
    print("-" * 82)
    m20 = all_metrics['Re20']
    m40 = all_metrics['Re40']
    print(f"{'A. Cd':<28} {m20['Cd']:<18.4f} {m40['Cd']:<18.4f} "
          f"{LITERATURE_CD[40]['range']}")
    print(f"{'A. Cl':<28} {m20['Cl']:<18.4f} {m40['Cl']:<18.4f} {'~0':<18}")
    print(f"{'A. max|div|':<28} {m20['div_max']:<18.4f} {m40['div_max']:<18.4f} "
          f"{'< 1e-3':<18}")
    print(f"{'B. L_r/D':<28} {m20['L_r_over_D']:<18.4f} "
          f"{m40['L_r_over_D']:<18.4f} {LITERATURE_LR[40]['range']}")
    print(f"{'B. theta_s (deg)':<28} {m20['theta_s']:<18.2f} "
          f"{m40['theta_s']:<18.2f} {LITERATURE_THETA[40]['range']}")
    print(f"{'E. |omega_z|_max':<28} {m20['omega_z_max']:<18.4f} "
          f"{m40['omega_z_max']:<18.4f} {'-':<18}")
    print(f"{'E. asymmetry(u)':<28} {m20['asymmetry_u']:<18.4f} "
          f"{m40['asymmetry_u']:<18.4f} {'~0':<18}")
    print("=" * 72)

    return all_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='圆柱绕流 MVP 详细分析 (论文级)')
    parser.add_argument(
        '--method', choices=['ibm', 'sharp'], default='ibm',
        help='边界处理方法: ibm (源项体积力, 默认) 或 sharp (尖锐界面, '
             '八叉树+高斯积分)')
    args = parser.parse_args()
    main(method=args.method)
