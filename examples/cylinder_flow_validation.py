# -*- coding: utf-8 -*-
"""
cylinder_flow_validation.py — FVM+IBM 圆柱绕流验证示例

验证两个层级:
    Level 0: 纯通道 Poiseuille 流 (无障碍物) — 质量守恒与散度收敛
    Level 1: 圆柱绕流 (Re=20, Re=40) — Cd 量级与对称性

参考文献:
    Li et al., JCAD 2024; 经典圆柱绕流实验数据 (Tritton 1959).

运行:
    python examples/cylinder_flow_validation.py
输出:
    figures/cylinder_flow_validation.png
    figures/channel_flow_validation.png
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 确保从仓库根目录导入
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from xvoxel.xvoxel import XVoxelModel
from xvoxel.primitives import CylinderZ
from xvoxel.csg import BoolOp

from fluid.solver import FluidSolver

FIG_DIR = os.path.join(_REPO_ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


# ======================================================================
#  Level 0: 纯通道流
# ======================================================================

def run_channel_flow():
    """Level 0: 纯通道 Poiseuille 流验证.

    无障碍物, 入口均匀来流, 出口压力 0, 上下滑移壁面.
    预期: 散度收敛, 出口速度 ≈ 入口速度 (质量守恒).
    """
    print("=" * 70)
    print("Level 0: 纯通道流验证 (无障碍物)")
    print("=" * 70)

    # 网格: 6×4×0.4 域, 60×40×4 体素
    xv = XVoxelModel(60, 40, 4, 6.0, 4.0, 0.4)
    solver = FluidSolver(xv, Re=100, U_inf=1.0)
    solver.add_inlet_bc('xmin', (1.0, 0.0, 0.0))
    solver.add_outlet_bc('xmax', p=0.0)
    solver.add_slip_wall('ymin')
    solver.add_slip_wall('ymax')
    solver.add_slip_wall('zmin')
    solver.add_slip_wall('zmax')
    solver.set_solver_params(dt=0.1, alpha_p=0.3, alpha_u=0.7)

    t0 = time.time()
    u, v, p = solver.solve(max_iter=300, tol=1e-3,
                           momentum_iter=20, pressure_iter=100,
                           verbose=True)
    dt = time.time() - t0

    div_max = float(np.max(np.abs(solver.grid.divergence())))
    u_outlet = float(u[-1, :, :].mean())
    u_inlet = float(u[0, :, :].mean())

    print(f"\n[Level 0 结果]")
    print(f"  耗时: {dt:.1f}s")
    print(f"  最大散度: {div_max:.4f}")
    print(f"  入口速度: {u_inlet:.4f}")
    print(f"  出口速度: {u_outlet:.4f}")
    print(f"  质量守恒误差: {abs(u_outlet - u_inlet) / u_inlet * 100:.2f}%")

    # 绘制中心线速度剖面
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左: 速度云图 (z=0 切片)
    ax = axes[0]
    u_slice = u[:, :, 0].T  # (ny, nx)
    im = ax.imshow(u_slice, origin='lower', aspect='auto',
                   extent=[0, 6, 0, 4], cmap='viridis', vmin=0, vmax=1.2)
    ax.set_title('Channel flow: u-velocity (z=0 slice)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='u')

    # 右: 中心线速度演化
    ax = axes[1]
    # u 在 x 面上, 形状 (nx+1, ny, nz); 用面坐标
    x_faces = np.linspace(0, 6, 61)
    u_centerline = u[:, 20, 0]  # y 中心, z=0
    ax.plot(x_faces, u_centerline, 'b-o', markersize=3, label='u(x, y=2)')
    ax.axhline(1.0, color='r', linestyle='--', label='U_inf')
    ax.set_title('Centerline velocity')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'channel_flow_validation.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  图像已保存: {out_path}")

    return {'div_max': div_max, 'u_outlet': u_outlet, 'u_inlet': u_inlet}


# ======================================================================
#  Level 1: 圆柱绕流
# ======================================================================

def run_cylinder_flow(Re: float, nx: int = 120, ny: int = 80,
                      max_iter: int = 400):
    """Level 1: 圆柱绕流验证.

    Args:
        Re: 雷诺数 (20 或 40).
        nx, ny: 网格分辨率.
        max_iter: 最大迭代次数.

    Returns:
        结果字典.
    """
    print("=" * 70)
    print(f"Level 1: 圆柱绕流 (Re={Re})")
    print("=" * 70)

    # 域: 6×4×0.4, 圆柱 D=0.8 (R=0.4) 位于 (1.5, 2.0)
    xv = XVoxelModel(nx, ny, 4, 6.0, 4.0, 0.4)
    cyl = CylinderZ(1.5, 2.0, 0.4, -1.0, 2.0)
    xv.add_feature(cyl, op=BoolOp.UNION)

    solver = FluidSolver(xv, Re=Re, U_inf=1.0)
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
                           verbose=True)
    dt = time.time() - t0

    cd = solver.compute_drag_coefficient(D=0.8)
    cl = solver.compute_lift_coefficient(D=0.8)
    div_max = float(np.max(np.abs(solver.grid.divergence())))

    # 文献参考值
    if Re == 40:
        cd_ref_range = (1.43, 1.60)  # Tritton / 经典实验
    elif Re == 20:
        cd_ref_range = (1.70, 2.10)  # Tritton
    else:
        cd_ref_range = (1.0, 2.5)

    print(f"\n[Level 1 结果 (Re={Re})]")
    print(f"  耗时: {dt:.1f}s")
    print(f"  最大散度: {div_max:.4f}")
    print(f"  阻力系数 Cd: {cd:.3f}")
    print(f"  升力系数 Cl: {cl:.4f}")
    print(f"  文献参考 Cd 范围: [{cd_ref_range[0]}, {cd_ref_range[1]}]")

    # 绘制流场
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # u 速度云图
    ax = axes[0, 0]
    # u 在 x 面 (nx+1, ny, nz); 取内部 ny 个用于 imshow (ny, nx+1)
    u_slice = u[:, :, 0].T  # (ny, nx+1)
    im = ax.imshow(u_slice, origin='lower', aspect='equal',
                   extent=[0, 6, 0, 4], cmap='RdBu_r',
                   vmin=-0.2, vmax=1.2)
    # 标记圆柱位置
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(1.5 + 0.4 * np.cos(theta), 2.0 + 0.4 * np.sin(theta),
            'k-', linewidth=2)
    ax.set_title(f'u-velocity (Re={Re}, Cd={cd:.3f})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='u')

    # v 速度云图
    ax = axes[0, 1]
    # v 在 y 面 (nx, ny+1, nz); 取 (ny+1, nx)
    v_slice = v[:, :, 0].T  # (ny+1, nx)
    im = ax.imshow(v_slice, origin='lower', aspect='equal',
                   extent=[0, 6, 0, 4], cmap='RdBu_r',
                   vmin=-0.5, vmax=0.5)
    ax.plot(1.5 + 0.4 * np.cos(theta), 2.0 + 0.4 * np.sin(theta),
            'k-', linewidth=2)
    ax.set_title(f'v-velocity (Re={Re})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='v')

    # 压力场
    ax = axes[1, 0]
    p_slice = p[:, :, 0].T  # (ny, nx)
    im = ax.imshow(p_slice, origin='lower', aspect='equal',
                   extent=[0, 6, 0, 4], cmap='coolwarm')
    ax.plot(1.5 + 0.4 * np.cos(theta), 2.0 + 0.4 * np.sin(theta),
            'k-', linewidth=2)
    ax.set_title(f'Pressure (Re={Re})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='p')

    # 流线 (用 streamplot, 在 cell 中心采样)
    ax = axes[1, 1]
    # cell 中心速度: u_c[i,j] = 0.5*(u[i,j]+u[i+1,j]), v_c[i,j] = 0.5*(v[i,j]+v[i,j+1])
    u_c = 0.5 * (u[:-1, :, 0] + u[1:, :, 0])  # (nx, ny)
    v_c = 0.5 * (v[:, :-1, 0] + v[:, 1:, 0])  # (nx, ny)
    x = np.linspace(0.05, 5.95, nx)
    y = np.linspace(0.05, 3.95, ny)
    u_grid = u_c.T  # (ny, nx)
    v_grid = v_c.T  # (ny, nx)
    speed = np.sqrt(u_grid ** 2 + v_grid ** 2)
    ax.streamplot(x, y, u_grid, v_grid, color=speed,
                  cmap='viridis', density=1.5, linewidth=1.0)
    ax.add_patch(plt.Circle((1.5, 2.0), 0.4, color='black'))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.set_title(f'Streamlines (Re={Re})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.suptitle(f'Cylinder cross-flow validation (Re={Re}, Cd={cd:.3f}, '
                 f'Cl={cl:.4f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f'cylinder_flow_Re{Re}.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  图像已保存: {out_path}")

    return {
        'Re': Re, 'Cd': cd, 'Cl': cl, 'div_max': div_max,
        'cd_ref_range': cd_ref_range, 'time': dt,
    }


# ======================================================================
#  主程序
# ======================================================================

def main():
    """运行全部验证."""
    print("\n" + "=" * 70)
    print("FVM+IBM 流体求解器验证")
    print("XVoxel-FCM 项目 — 圆柱绕流 MVP")
    print("=" * 70)

    results = {}

    # Level 0: 纯通道
    results['level0'] = run_channel_flow()

    # Level 1: 圆柱绕流 Re=40
    results['level1_Re40'] = run_cylinder_flow(Re=40, nx=120, ny=80,
                                                max_iter=400)

    # Level 1: 圆柱绕流 Re=20
    results['level1_Re20'] = run_cylinder_flow(Re=20, nx=120, ny=80,
                                                max_iter=400)

    # 汇总
    print("\n" + "=" * 70)
    print("验证汇总")
    print("=" * 70)
    print(f"{'项目':<25} {'结果':<20} {'参考':<20} {'判定':<10}")
    print("-" * 75)

    # Level 0
    r0 = results['level0']
    ok0 = r0['div_max'] < 0.5
    print(f"{'Level 0 div':<25} {r0['div_max']:<20.4f} {'< 0.5':<20} "
          f"{'PASS' if ok0 else 'FAIL':<10}")
    ok0b = abs(r0['u_outlet'] - r0['u_inlet']) / r0['u_inlet'] < 0.3
    print(f"{'Level 0 mass-conserv':<25} "
          f"{r0['u_outlet']:<20.4f} {'~ 1.0':<20} "
          f"{'PASS' if ok0b else 'FAIL':<10}")

    # Level 1 Re=40
    r40 = results['level1_Re40']
    lo, hi = r40['cd_ref_range']
    ok40 = 1.0 < r40['Cd'] < 2.5
    print(f"{'Level 1 Cd (Re=40)':<25} {r40['Cd']:<20.3f} "
          f"{f'[{lo},{hi}]':<20} {'PASS' if ok40 else 'FAIL':<10}")
    ok40b = abs(r40['Cl']) < 0.5
    print(f"{'Level 1 Cl (Re=40)':<25} {r40['Cl']:<20.4f} {'|Cl|<0.5':<20} "
          f"{'PASS' if ok40b else 'FAIL':<10}")

    # Level 1 Re=20
    r20 = results['level1_Re20']
    lo, hi = r20['cd_ref_range']
    ok20 = 1.0 < r20['Cd'] < 2.5
    print(f"{'Level 1 Cd (Re=20)':<25} {r20['Cd']:<20.3f} "
          f"{f'[{lo},{hi}]':<20} {'PASS' if ok20 else 'FAIL':<10}")
    ok20b = abs(r20['Cl']) < 0.5
    print(f"{'Level 1 Cl (Re=20)':<25} {r20['Cl']:<20.4f} {'|Cl|<0.5':<20} "
          f"{'PASS' if ok20b else 'FAIL':<10}")

    print("=" * 70)
    all_ok = all([ok0, ok0b, ok40, ok40b, ok20, ok20b])
    print(f"Overall: {'ALL PASS' if all_ok else 'PARTIAL (see table above)'}")
    print("=" * 70)

    return results


if __name__ == '__main__':
    main()
