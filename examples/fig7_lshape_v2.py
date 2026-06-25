# -*- coding: utf-8 -*-
"""
examples/fig7_lshape_v2.py — L型支架 (Phase 1 重构版)

使用 xvoxel + fcm 包, 复现论文 Fig 7-9.
几何: 两 Cube + RoundCorner2D 圆角, 5 步半径编辑 6→2 mm.
BC: 顶面固定, 右侧面向下牵引力 100 N/mm².
"""
import time
import numpy as np
from xvoxel import XVoxelModel, Cube, RoundCorner2D
from fcm import FCMSolver

# ============================================================
# 模型参数 (论文 Fig 7)
# ============================================================
NX, NY, NZ = 15, 15, 3
LX, LY, LZ = 15.0, 15.0, 3.0

E = 2e5       # MPa
NU = 0.3
ALPHA = 1e-8
ORDER = 1     # Hex8

TRACTION_Y = -100.0  # N/mm² 向下

# L 型几何
# 竖直臂: x∈[0,3], y∈[0,15]
# 水平臂: x∈[0,15], y∈[0,3]
# 内角: (3, 3), 圆角半径 R

def build_lshape_model(r=6.0):
    """构建 L 型 XVoxel 模型."""
    origin = (0.0, 0.0, -LZ/2)
    xv = XVoxelModel(NX, NY, NZ, LX, LY, LZ, origin=origin)

    # 竖直臂
    vert = Cube(cx=1.5, cy=7.5, cz=0.0, sx=3.0, sy=15.0, sz=3.0, name="vertical_arm")
    xv.add_feature(vert)

    # 水平臂
    horiz = Cube(cx=7.5, cy=1.5, cz=0.0, sx=15.0, sy=3.0, sz=3.0, name="horizontal_arm")
    xv.add_feature(horiz)

    # 内角圆角 (加材料填充)
    corner = RoundCorner2D(cx=3.0, cy=3.0, r=r,
                           zmin=-LZ/2, zmax=LZ/2, sign_x=+1, sign_y=+1,
                           name="round_corner")
    corner_fid = xv.add_feature(corner)

    solid = np.sum(xv.voxel_nature == 1)
    bnd = np.sum(xv.voxel_nature == 0)
    void = np.sum(xv.voxel_nature == -1)
    print(f"  Voxels: solid={solid}, boundary={bnd}, void={void}")

    return xv, corner_fid


def run_step(xv, step_label, solver_order=1, max_depth=3):
    """运行一个半径步骤的 FCM 求解."""
    print(f"\n{'='*60}")
    print(f"  {step_label}")
    print(f"{'='*60}")

    t0 = time.time()
    solver = FCMSolver(xv, order=solver_order)
    solver.set_material(E, NU, ALPHA)

    # BC: 顶面 (ymax) 固定
    solver.add_dirichlet_bc('ymax', 'ux,uy,uz', 0.0)
    # 载荷: 右侧面 (xmax) 向下牵引力
    solver.add_traction_bc('xmax', (0.0, TRACTION_Y, 0.0))

    u = solver.solve(max_depth=max_depth)
    vm = solver.compute_von_mises()
    elapsed = time.time() - t0

    max_u = np.max(np.abs(u))
    max_vm = np.max(vm[vm > 0]) if np.any(vm > 0) else 0.0
    print(f"  Max displacement: {max_u:.6f}")
    print(f"  Max von Mises: {max_vm:.4f}")
    print(f"  Time: {elapsed:.2f}s")

    return solver, u, vm


def main():
    print("=" * 60)
    print("  Fig 7: L-Shape Bracket — Phase 1 Refactor")
    print("=" * 60)

    # 构建初始模型 (R=6)
    print("\nBuilding initial model (R=6)...")
    xv, corner_fid = build_lshape_model(r=6.0)

    # 步 1-5: 半径编辑 6→5→4→3→2
    radii = [6.0, 5.0, 4.0, 3.0, 2.0]
    results = []

    for i, r in enumerate(radii):
        if i == 0:
            # 初始状态已构建, 直接求解
            step_label = f"Step {i+1}: R={r} mm"
        else:
            step_label = f"Step {i+1}: R={r} mm"
            t0 = time.time()
            xv.edit_parameter(corner_fid, 'r', r)
            print(f"  Edit R {radii[i-1]}→{r}: {time.time()-t0:.3f}s")

        solver, u, vm = run_step(xv, step_label, solver_order=ORDER, max_depth=3)
        results.append({
            'radius': r,
            'max_u': np.max(np.abs(u)),
            'max_vm': np.max(vm[vm > 0]) if np.any(vm > 0) else 0.0,
        })

    # 输出汇总
    print(f"\n{'='*60}")
    print("  Results Summary")
    print(f"{'='*60}")
    print(f"  {'R (mm)':>8s}  {'Max U':>12s}  {'Max σ_vm':>12s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}")
    for r in results:
        print(f"  {r['radius']:8.1f}  {r['max_u']:12.6f}  {r['max_vm']:12.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
