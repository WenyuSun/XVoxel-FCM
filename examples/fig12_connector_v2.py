# -*- coding: utf-8 -*-
"""
examples/fig12_connector_v2.py — 机械连杆 (Phase 1 重构版)

使用 xvoxel + fcm 包, 复现论文 Fig 10-12.
几何: Cube 主体 + 两 CylinderZ 外圆柱 + 两 CylinderZ 内孔 + 两 CylinderZ 沟槽.
BC: 左端孔面固定, 右端孔面轴承载荷.
"""
import time
import numpy as np
from xvoxel import XVoxelModel, Cube, CylinderZ, BoolOp
from fcm import FCMSolver

# ============================================================
# 模型参数 (论文 Fig 10)
# ============================================================
NX, NY, NZ = 55, 16, 9
LX, LY, LZ = 55.0, 16.0, 9.0

E = 2e5       # MPa
NU = 0.3
ALPHA = 1e-8
ORDER = 1     # Hex8 (大型模型用低阶)

# 载荷 (缩放以匹配论文位移量级 ~0.2mm)
TRACTION_X = 100.0 / 30.0   # ~3.33 N/mm²
TRACTION_Y = 200.0 / 30.0   # ~6.67 N/mm²

# 几何参数
LEFT_CYL_CX = 12.0
LEFT_CYL_R_OUTER = 8.0
LEFT_CYL_R_INNER = 5.0

RIGHT_CYL_CX = 43.0
RIGHT_CYL_R_OUTER = 8.0
RIGHT_CYL_R_INNER = 5.0

GROOVE1_CX = 20.0
GROOVE2_CX = 35.0
GROOVE_R = 2.0


def build_connector_model():
    """构建连杆 XVoxel 模型."""
    origin = (0.0, -LY/2, -LZ/2)
    xv = XVoxelModel(NX, NY, NZ, LX, LY, LZ, origin=origin)

    # 1. 主体方块 (加材料)
    body = Cube(cx=LX/2, cy=0, cz=0, sx=LX, sy=LY, sz=LZ, name="body")
    xv.add_feature(body)

    # 2. 左端外圆柱 (加材料)
    left_outer = CylinderZ(cx=LEFT_CYL_CX, cy=0, r=LEFT_CYL_R_OUTER,
                           zmin=-LZ/2, zmax=LZ/2, name="left_outer")
    xv.add_feature(left_outer)

    # 3. 右端外圆柱 (加材料)
    right_outer = CylinderZ(cx=RIGHT_CYL_CX, cy=0, r=RIGHT_CYL_R_OUTER,
                            zmin=-LZ/2, zmax=LZ/2, name="right_outer")
    xv.add_feature(right_outer)

    # 4. 左端内孔 (减材料)
    left_inner = CylinderZ(cx=LEFT_CYL_CX, cy=0, r=LEFT_CYL_R_INNER,
                           zmin=-LZ/2, zmax=LZ/2, name="left_inner")
    xv.add_feature(left_inner, op=BoolOp.DIFFERENCE)

    # 5. 右端内孔 (减材料)
    right_inner = CylinderZ(cx=RIGHT_CYL_CX, cy=0, r=RIGHT_CYL_R_INNER,
                            zmin=-LZ/2, zmax=LZ/2, name="right_inner")
    xv.add_feature(right_inner, op=BoolOp.DIFFERENCE)

    # 6. 沟槽 1 (减材料)
    groove1 = CylinderZ(cx=GROOVE1_CX, cy=0, r=GROOVE_R,
                        zmin=-LZ/2, zmax=LZ/2, name="groove_1")
    g1_fid = xv.add_feature(groove1, op=BoolOp.DIFFERENCE)

    # 7. 沟槽 2 (减材料)
    groove2 = CylinderZ(cx=GROOVE2_CX, cy=0, r=GROOVE_R,
                        zmin=-LZ/2, zmax=LZ/2, name="groove_2")
    g2_fid = xv.add_feature(groove2, op=BoolOp.DIFFERENCE)

    solid = np.sum(xv.voxel_nature == 1)
    bnd = np.sum(xv.voxel_nature == 0)
    void = np.sum(xv.voxel_nature == -1)
    print(f"  Voxels: solid={solid}, boundary={bnd}, void={void}")

    fids = {'groove1': g1_fid, 'groove2': g2_fid}
    return xv, fids


def run_connector(xv, step_label, max_depth=3):
    """运行 FCM 求解."""
    print(f"\n{'='*60}")
    print(f"  {step_label}")
    print(f"{'='*60}")

    t0 = time.time()
    solver = FCMSolver(xv, order=ORDER)
    solver.set_material(E, NU, ALPHA)

    # BC: 左端孔面 (内圆柱面) 固定
    # 简化: 将 xmin 面所有节点固定
    solver.add_dirichlet_bc('xmin', 'ux,uy,uz', 0.0)

    # 载荷: 右端孔面轴承载荷 (简化为 xmax 面牵引力)
    solver.add_traction_bc('xmax', (TRACTION_X, TRACTION_Y, 0.0))

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
    print("  Fig 12: Connector Rod — Phase 1 Refactor")
    print("=" * 60)

    # 构建初始模型
    print("\nBuilding connector model...")
    xv, fids = build_connector_model()

    # 步 1: 初始状态
    solver, u, vm = run_connector(xv, "Step 1: Initial (groove R=2.0)")

    # 步 2: 编辑沟槽半径 2.0→1.5
    print("\n  Editing groove radius 2.0→1.5...")
    xv.edit_parameter(fids['groove1'], 'r', 1.5)
    xv.edit_parameter(fids['groove2'], 'r', 1.5)
    solver, u, vm = run_connector(xv, "Step 2: Groove R=1.5")

    # 步 3: 编辑沟槽半径 1.5→1.0
    print("\n  Editing groove radius 1.5→1.0...")
    xv.edit_parameter(fids['groove1'], 'r', 1.0)
    xv.edit_parameter(fids['groove2'], 'r', 1.0)
    solver, u, vm = run_connector(xv, "Step 3: Groove R=1.0")

    # 步 4: 编辑沟槽圆心
    print("\n  Editing groove center x offsets...")
    xv.edit_parameter(fids['groove1'], 'cx', 22.0)
    xv.edit_parameter(fids['groove2'], 'cx', 33.0)
    solver, u, vm = run_connector(xv, "Step 4: Groove cx shifted")

    print("\nDone.")


if __name__ == '__main__':
    main()
