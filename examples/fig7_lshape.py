# -*- coding: utf-8 -*-
"""
fig7_lshape.py — L型模型复现 (论文 Example #1, Fig 7-9)

几何构造: 两个立方体 + 一个圆角 (CSG union)
  - 竖直臂 Cube(x∈[0,3], y∈[0,15]) + 水平臂 Cube(x∈[0,15], y∈[0,3])
  - 内角圆角: corner-fill region 在 (3,3), 象限 x>3, y>3 — 加材料填充
  - 论文参数: 每条臂宽3mm, 长15mm, 深3mm
编辑序列: 5步修改圆角半径 R: 6→5→4→3→2 mm

边界条件:
  - 固定: 上表面 (upper face, y = ymax)
  - 载荷: 右侧面 (right face, x = xmax) 向下 τ = 100 N/mm²

网格: 3×15×15 = 675 体素 (薄板平面应力)
材料: E = 2e5 N/mm² (MPa), ν = 0.3
"""
import sys, os
import time
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.primitives import Cube, RoundCorner2D
from src.xvoxel import XVoxelModel
from src.fem_xvoxel import XVoxelFEMSolver

# ============================================================
# 模型参数 (论文 Fig 7)
# ============================================================
# 体素网格: 论文 Fig 7e: 3×15×15 = 768 体素
# 注意: 论文Table 1中"675"应为"3 15 15"(768体素)的排版错误
NX, NY, NZ = 15, 15, 3
LX, LY, LZ = 15.0, 15.0, 3.0  # 几何尺寸 (mm), 论文: 15×15×3
# 厚度方向只有3层，模拟薄板平面应力

E = 2e5     # 杨氏模量 (N/mm² = MPa), 论文: 2e11 Pa = 2e5 N/mm²
NU = 0.3    # 泊松比
ALPHA = 1e-8  # FCM 虚拟域系数

# 载荷 (论文 Fig 7c)
# 论文原文: "subject to a downward traction of τ = 100 N/mm²"
# 单位: N/mm² = MPa
TRACTION_Y = -100.0  # 向下 (负y方向), 单位: N/mm² = MPa

# L型几何参数 (论文 Fig 7: 每条臂宽3mm, 长15mm, 深3mm)
#
# 坐标系: origin = (0, 0, -1.5)
#   x: 0 到 15, y: 0 到 15, z: -1.5 到 1.5
#
# L型构造:
#   竖直臂 (Cube1): 左侧，宽3，高15
#     范围: x=[0,3], y=[0,15], 中心: (1.5, 7.5, 0)
#   水平臂 (Cube2): 底部，宽15，高3
#     范围: x=[0,15], y=[0,3], 中心: (7.5, 1.5, 0)
#   内角顶点: (3, 3)
#   圆角: corner-fill (Box - Cylinder) at (3,3), radius R

# 竖直臂参数
VERT_SX, VERT_SY, VERT_SZ = 3.0, 15.0, 3.0
VERT_CX, VERT_CY, VERT_CZ = 1.5, 7.5, 0.0

# 水平臂参数
HORIZ_SX, HORIZ_SY, HORIZ_SZ = 15.0, 3.0, 3.0
HORIZ_CX, HORIZ_CY, HORIZ_CZ = 7.5, 1.5, 0.0

# 内角顶点 (x=3, y=3)
CORNER_CX, CORNER_CY = 3.0, 3.0
CORNER_R_INIT = 6.0  # 初始圆角半径


def build_lshape_model():
    """
    构建L型 XVoxel 模型（初始状态，Step 1）
    CSG 构造:
      1. Cube1(竖直臂) — 加材料
      2. Cube2(水平臂) — 加材料 (union)
      3. RoundCorner(圆角) — 加材料 (union, 填充内角)
    """
    origin = (0.0, 0.0, -LZ/2)  # x从0到15, y从0到15, z从-1.5到1.5
    xv = XVoxelModel(NX, NY, NZ, LX, LY, LZ, origin=origin)

    # 1. 竖直臂 (加材料)
    vert = Cube(cx=VERT_CX, cy=VERT_CY, cz=VERT_CZ,
                sx=VERT_SX, sy=VERT_SY, sz=VERT_SZ)
    fid_vert = xv.add_feature(vert, nature=1, name="vertical_arm")
    print(f"  Vertical arm: fid={fid_vert}")

    # 2. 水平臂 (加材料)
    horiz = Cube(cx=HORIZ_CX, cy=HORIZ_CY, cz=HORIZ_CZ,
                 sx=HORIZ_SX, sy=HORIZ_SY, sz=HORIZ_SZ)
    fid_horiz = xv.add_feature(horiz, nature=1, name="horizontal_arm")
    print(f"  Horizontal arm: fid={fid_horiz}")

    # 3. 凹圆角 (加材料!) — 在L型内角填充Box-Cylinder，形成凹入圆角
    # RoundCorner2D: corner fill at (3,3), radius R, sign_x=+1/sign_y=+1
    # nature=+1 加材料填充内角区域
    corner = RoundCorner2D(cx=CORNER_CX, cy=CORNER_CY, r=CORNER_R_INIT,
                           zmin=-LZ/2, zmax=LZ/2, sign_x=+1, sign_y=+1)
    fid_corner = xv.add_feature(corner, nature=+1, name="round_corner")
    print(f"  Inner fillet (R={CORNER_R_INIT}): fid={fid_corner}")

    solid = np.sum(xv.voxel_nature == 1)
    void = np.sum(xv.voxel_nature == -1)
    bnd  = np.sum(xv.voxel_nature == 0)
    print(f"  Voxels: solid={solid}, void={void}, boundary={bnd}, total={solid+void+bnd}")

    fids = {
        'vertical': fid_vert,
        'horizontal': fid_horiz,
        'corner': fid_corner,
    }
    return xv, fids


def build_step_sequence(xv, fids):
    """
    5 步编辑序列 (论文 Fig 7, Table 1 Example #1)
    修改圆角半径 R: 6→5→4→3→2 mm
    """
    steps = []

    # Step 1: 初始状态 (R=6)
    steps.append(("Step 1: R=6 mm", []))

    # Step 2: R=6→5
    steps.append(("Step 2: R=5 mm",
                  [(fids['corner'], 'r', 5.0)]))

    # Step 3: R=5→4
    steps.append(("Step 3: R=4 mm",
                  [(fids['corner'], 'r', 4.0)]))

    # Step 4: R=4→3
    steps.append(("Step 4: R=3 mm",
                  [(fids['corner'], 'r', 3.0)]))

    # Step 5: R=3→2
    steps.append(("Step 5: R=2 mm",
                  [(fids['corner'], 'r', 2.0)]))

    return steps


def get_face_fixed_dofs(solver, face_type):
    """获取面上所有节点的固定自由度 (3方向全部固定)"""
    xv = solver.xvoxel
    fixed_dofs = []
    for nid in range(solver.n_nodes):
        x, y, z = solver.nodes[nid]
        tol = 1e-10
        if face_type == 'xmin' and abs(x - xv.ox) < tol:
            fixed_dofs.extend([nid*3, nid*3+1, nid*3+2])
        elif face_type == 'xmax' and abs(x - (xv.ox+xv.lx)) < tol:
            fixed_dofs.extend([nid*3, nid*3+1, nid*3+2])
        elif face_type == 'ymin' and abs(y - xv.oy) < tol:
            fixed_dofs.extend([nid*3, nid*3+1, nid*3+2])
        elif face_type == 'ymax' and abs(y - (xv.oy+xv.ly)) < tol:
            fixed_dofs.extend([nid*3, nid*3+1, nid*3+2])
        elif face_type == 'zmin' and abs(z - xv.oz) < tol:
            fixed_dofs.extend([nid*3, nid*3+1, nid*3+2])
        elif face_type == 'zmax' and abs(z - (xv.oz+xv.lz)) < tol:
            fixed_dofs.extend([nid*3, nid*3+1, nid*3+2])
    return list(set(fixed_dofs))


def get_face_traction_nodal_forces(solver, face_type, traction):
    """
    使用正确的 Gauss 积分计算面载荷的一致节点力
    支持 Hex8 (4节点面), Hex20 (8节点面), Hex32 (12节点面)
    只对物理固体区域（PMC 结果为 +1 或 0）施加载荷
    """
    xv = solver.xvoxel
    F = np.zeros(solver.ndof)
    gauss_2_pts = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
    gauss_3_pts = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    gauss_3_wts = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    order = solver.element_order

    if order == 3:
        # Hex32 faces: 12 nodes per face (4 corners + 8 edge nodes)
        face_nodes_map = {
            'xmin': [0, 3, 7, 4, 14, 15, 30, 31, 22, 23, 24, 25],
            'xmax': [1, 2, 6, 5, 10, 11, 28, 29, 18, 19, 26, 27],
            'ymin': [0, 1, 5, 4, 8, 9, 26, 27, 16, 17, 24, 25],
            'ymax': [3, 2, 6, 7, 12, 13, 28, 29, 20, 21, 30, 31],
            'zmin': [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15],
            'zmax': [4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23],
        }
    elif order == 2:
        face_nodes_map = {
            'xmin': [0, 3, 7, 4, 11, 15, 19, 16],
            'xmax': [1, 2, 6, 5, 9, 13, 18, 17],
            'ymin': [0, 1, 5, 4, 8, 12, 17, 16],
            'ymax': [2, 3, 7, 6, 10, 15, 19, 18],
            'zmin': [0, 1, 2, 3, 8, 9, 10, 11],
            'zmax': [4, 5, 6, 7, 12, 13, 14, 15],
        }
    else:
        face_nodes_map = {
            'xmin': [0, 3, 7, 4],
            'xmax': [1, 2, 6, 5],
            'ymin': [0, 1, 5, 4],
            'ymax': [2, 3, 7, 6],
            'zmin': [0, 1, 2, 3],
            'zmax': [4, 5, 6, 7],
        }

    for eid in range(xv.n_voxels):
        coords = solver._get_elem_coords(eid)
        tol = 1e-10

        is_face = False
        if face_type == 'xmin' and abs(coords[0,0] - xv.ox) < tol:
            is_face = True
        elif face_type == 'xmax' and abs(coords[1,0] - (xv.ox+xv.lx)) < tol:
            is_face = True
        elif face_type == 'ymin' and abs(coords[0,1] - xv.oy) < tol:
            is_face = True
        elif face_type == 'ymax' and abs(coords[3,1] - (xv.oy+xv.ly)) < tol:
            is_face = True
        elif face_type == 'zmin' and abs(coords[0,2] - xv.oz) < tol:
            is_face = True
        elif face_type == 'zmax' and abs(coords[4,2] - (xv.oz+xv.lz)) < tol:
            is_face = True

        if not is_face:
            continue

        fn = face_nodes_map[face_type]
        elem_nodes = solver.elems[eid]
        face_coords = coords[fn]
        n_face_nodes = len(fn)

        if n_face_nodes == 12:
            gauss_iter = [(xi, eta, gauss_3_wts[i]*gauss_3_wts[j])
                          for i, xi in enumerate(gauss_3_pts)
                          for j, eta in enumerate(gauss_3_pts)]
        else:
            gauss_iter = [(xi, eta, 1.0)
                          for xi in gauss_2_pts
                          for eta in gauss_2_pts]

        for xi, eta, w in gauss_iter:

            if order == 3:
                # Hex32 face: 12-node cubic Serendipity quadrilateral
                from src.fem_base import hex32_face_shape_12
                N_face, dN_xi, dN_eta = hex32_face_shape_12(face_type, xi, eta)
            elif order == 2:
                N_face = np.array([
                    0.25*(1-xi)*(1-eta)*(-xi-eta-1),
                    0.25*(1+xi)*(1-eta)*( xi-eta-1),
                    0.25*(1+xi)*(1+eta)*( xi+eta-1),
                    0.25*(1-xi)*(1+eta)*(-xi+eta-1),
                    0.5*(1-xi*xi)*(1-eta),
                    0.5*(1+xi)*(1-eta*eta),
                    0.5*(1-xi*xi)*(1+eta),
                    0.5*(1-xi)*(1-eta*eta),
                ])
                dN_xi = np.array([
                    0.25*(1-eta)*(2*xi+eta),
                    0.25*(1-eta)*(2*xi-eta),
                    0.25*(1+eta)*(2*xi+eta),
                    0.25*(1+eta)*(2*xi-eta),
                    -xi*(1-eta),
                    0.5*(1-eta*eta),
                    -xi*(1+eta),
                    -0.5*(1-eta*eta),
                ])
                dN_eta = np.array([
                    0.25*(1-xi)*(xi+2*eta),
                    0.25*(1+xi)*(-xi+2*eta),
                    0.25*(1+xi)*(xi+2*eta),
                    0.25*(1-xi)*(-xi+2*eta),
                    -0.5*(1-xi*xi),
                    -(1+xi)*eta,
                    0.5*(1-xi*xi),
                    -(1-xi)*eta,
                ])
            else:
                N_face = np.array([
                    0.25*(1-xi)*(1-eta),
                    0.25*(1+xi)*(1-eta),
                    0.25*(1+xi)*(1+eta),
                    0.25*(1-xi)*(1+eta),
                ])
                dN_xi = np.array([
                    -0.25*(1-eta),  0.25*(1-eta),
                     0.25*(1+eta), -0.25*(1+eta),
                ])
                dN_eta = np.array([
                    -0.25*(1-xi), -0.25*(1+xi),
                     0.25*(1+xi),  0.25*(1-xi),
                ])

            gp_xyz = N_face @ face_coords

            # PMC check
            from src.pmc import pmc_point_3d
            gp_status = pmc_point_3d(
                gp_xyz[0], gp_xyz[1], gp_xyz[2],
                xv.voxel_attrs[eid], xv.features
            )
            if gp_status == -1:
                continue

            if face_type in ['xmin', 'xmax']:
                dy_dxi  = dN_xi @ face_coords[:, 1]
                dy_deta = dN_eta @ face_coords[:, 1]
                dz_dxi  = dN_xi @ face_coords[:, 2]
                dz_deta = dN_eta @ face_coords[:, 2]
                J = np.array([[dy_dxi, dz_dxi], [dy_deta, dz_deta]])
            elif face_type in ['ymin', 'ymax']:
                dx_dxi  = dN_xi @ face_coords[:, 0]
                dx_deta = dN_eta @ face_coords[:, 0]
                dz_dxi  = dN_xi @ face_coords[:, 2]
                dz_deta = dN_eta @ face_coords[:, 2]
                J = np.array([[dx_dxi, dz_dxi], [dx_deta, dz_deta]])
            else:
                dx_dxi  = dN_xi @ face_coords[:, 0]
                dx_deta = dN_eta @ face_coords[:, 0]
                dy_dxi  = dN_xi @ face_coords[:, 1]
                dy_deta = dN_eta @ face_coords[:, 1]
                J = np.array([[dx_dxi, dy_dxi], [dx_deta, dy_deta]])

            detJ = abs(np.linalg.det(J))

            for a, ni in enumerate(fn):
                nid = elem_nodes[ni]
                force = w * N_face[a] * detJ
                F[nid*3]     += force * traction[0]
                F[nid*3 + 1] += force * traction[1]
                F[nid*3 + 2] += force * traction[2]

    return F


def compute_nodal_fields(solver, u, von_mises_elem):
    """
    将单元级结果插值到节点级，得到光滑的场分布
    位移: 直接从 u 向量提取
    应力: 用体积加权平均将单元应力映射到节点
    """
    n_nodes = solver.n_nodes
    nodal_disp = np.zeros(n_nodes)
    nodal_stress = np.zeros(n_nodes)
    nodal_weight = np.zeros(n_nodes)
    
    # 位移范数 → 节点
    for nid in range(n_nodes):
        ux = u[nid*3]
        uy = u[nid*3+1]
        uz = u[nid*3+2]
        nodal_disp[nid] = np.sqrt(ux**2 + uy**2 + uz**2)
    
    # von Mises 应力 → 节点 (体积加权平均)
    for eid in range(solver.xvoxel.n_voxels):
        vm = von_mises_elem[eid]
        for nid in solver.elems[eid]:
            nodal_stress[nid] += vm
            nodal_weight[nid] += 1.0
    
    # 避免除零
    mask = nodal_weight > 0
    nodal_stress[mask] /= nodal_weight[mask]
    
    return nodal_disp, nodal_stress


def nodal_field_to_grid(solver, nodal_values, nx, ny, nz, k_slice):
    """
    将节点级场值映射到结构化网格 (nx × ny)
    用于 imshow 可视化
    使用反距离加权插值
    """
    from scipy.interpolate import griddata
    
    # 获取所有节点的 x, y 坐标
    x_coords = solver.nodes[:, 0]
    y_coords = solver.nodes[:, 1]
    
    # 创建目标网格
    ox, oy = solver.xvoxel.ox, solver.xvoxel.oy
    dx, dy = solver.xvoxel.dx, solver.xvoxel.dy
    
    xi = np.linspace(ox + dx/2, ox + nx*dx - dx/2, nx)
    yi = np.linspace(oy + dy/2, oy + ny*dy - dy/2, ny)
    XI, YI = np.meshgrid(xi, yi)
    
    # 只使用 z ≈ z_slice 的节点
    z_target = oz_mid = solver.xvoxel.oz + (k_slice + 0.5) * solver.xvoxel.dz
    z_tol = solver.xvoxel.dz
    z_mask = np.abs(solver.nodes[:, 2] - z_target) < z_tol
    
    if np.sum(z_mask) < 4:
        return np.zeros((ny, nx))
    
    points = np.column_stack([x_coords[z_mask], y_coords[z_mask]])
    values = nodal_values[z_mask]
    
    # 反距离加权插值
    grid_values = griddata(points, values, (XI, YI), method='cubic', fill_value=0)
    
    return grid_values


def build_voxel_geometry(xv, corner_R):
    """
    构建 L-shape 的体素几何信息用于3D渲染
    L-shape = two rectangles PLUS quarter-cylinder fillet
    Cylinder center at (10+R, 10+R), tangent to both faces
    """
    solid_voxels = []
    for vi in range(xv.n_voxels):
        i, j, k = xv._ijk(vi)
        cx = xv.ox + (i + 0.5) * xv.dx
        cy = xv.oy + (j + 0.5) * xv.dy
        in_vert = (0 <= cx <= 3) and (0 <= cy <= 15)
        in_horiz = (0 <= cx <= 15) and (0 <= cy <= 3)
        # Fillet: box minus quarter-cylinder (region between arc and L-shape)
        ocx, ocy = 3.0 + corner_R, 3.0 + corner_R
        in_fillet = (cx > 3) and (cy > 3) and (cx <= 3+corner_R) and (cy <= 3+corner_R) and ((cx - ocx)**2 + (cy - ocy)**2) > corner_R**2
        if in_vert or in_horiz or in_fillet:
            solid_voxels.append((i, j, k))
    return solid_voxels


def plot_results(results, output_dir):
    """生成 Fig 9 风格云图 (2D + 3D) — 仅展示 steps 1/3/5"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.interpolate import griddata

    # 只展示 step 1, 3, 5 (与论文 Fig 9 一致)
    paper_steps = [0, 2, 4]  # 0-indexed: step 1, 3, 5
    step_labels = ['Step 1 (R=6)', 'Step 3 (R=4)', 'Step 5 (R=2)']
    step_radii = [results['corner_radii'][i] for i in paper_steps]
    n_show = 3
    k_mid = NZ // 2

    # 高分辨率网格用于 SDF 掩码
    HRES = 200
    xv = results['solver'].xvoxel
    solver = results['solver']
    xi_hr = np.linspace(xv.ox, xv.ox + xv.lx, HRES)
    yi_hr = np.linspace(xv.oy, xv.oy + xv.ly, HRES)
    XI_HR, YI_HR = np.meshgrid(xi_hr, yi_hr)

    # Build per-step geometry masks: L-shape PLUS quarter-cylinder fillet
    # Cylinder center = (10+R, 10+R) — tangent to both faces
    geo_masks = []
    for R in step_radii:
        l_shape = (
            ((XI_HR >= 0) & (XI_HR <= 3) & (YI_HR >= 0) & (YI_HR <= 15)) |
            ((XI_HR >= 0) & (XI_HR <= 15) & (YI_HR >= 0) & (YI_HR <= 3))
        )
        # Fillet: box minus quarter-cylinder (region between arc and L-shape)
        # 3 < x ≤ 3+R, 3 < y ≤ 3+R, outside the cylinder
        fillet_add = (XI_HR > 3) & (YI_HR > 3) & (XI_HR <= 3+R) & (YI_HR <= 3+R) & (((XI_HR - (3+R))**2 + (YI_HR - (3+R))**2) > R**2)
        solid = l_shape | fillet_add
        geo_masks.append(~solid)

    # ---- 图1: 2D 合成云图 ----
    fig2d, axes = plt.subplots(2, n_show, figsize=(5.5*n_show, 9))

    vmin_d, vmax_d = 1e10, -1e10
    vmin_s, vmax_s = 1e10, -1e10
    for idx, si in enumerate(paper_steps):
        if results['nodal_disp'][si] is not None:
            d = results['nodal_disp'][si]
            valid = d[d > 0]
            if len(valid) > 0:
                vmin_d = min(vmin_d, 0)
                vmax_d = max(vmax_d, valid.max())
        if results['nodal_stress'][si] is not None:
            s = results['nodal_stress'][si]
            valid = s[s > 0]
            if len(valid) > 0:
                vmin_s = min(vmin_s, 0)
                vmax_s = max(vmax_s, valid.max())

    for idx, si in enumerate(paper_steps):
        ax_d = axes[0, idx]
        ax_s = axes[1, idx]

        if results['nodal_disp'][si] is not None:
            x_coords = solver.nodes[:, 0]
            y_coords = solver.nodes[:, 1]
            z_target = xv.oz + xv.lz / 2
            z_tol = xv.dz
            z_mask = np.abs(solver.nodes[:, 2] - z_target) < z_tol
            if np.sum(z_mask) >= 4:
                points = np.column_stack([x_coords[z_mask], y_coords[z_mask]])
                values_d = results['nodal_disp'][si][z_mask]
                values_s = results['nodal_stress'][si][z_mask]

                d_grid = griddata(points, values_d, (XI_HR, YI_HR), method='cubic', fill_value=np.nan)
                s_grid = griddata(points, values_s, (XI_HR, YI_HR), method='cubic', fill_value=np.nan)

                d_plot = np.ma.masked_where(geo_masks[idx], d_grid)
                s_plot = np.ma.masked_where(geo_masks[idx], s_grid)

                im_d = ax_d.pcolormesh(XI_HR, YI_HR, d_plot, cmap='jet',
                                       norm=Normalize(vmin=vmin_d, vmax=vmax_d),
                                       shading='auto', rasterized=True)
                divider = make_axes_locatable(ax_d)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im_d, cax=cax, label='|u| (mm)')

                im_s = ax_s.pcolormesh(XI_HR, YI_HR, s_plot, cmap='jet',
                                       norm=Normalize(vmin=vmin_s, vmax=vmax_s),
                                       shading='auto', rasterized=True)
                divider = make_axes_locatable(ax_s)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im_s, cax=cax, label='σ_vM (MPa)')

        ax_d.set_title(f'{step_labels[idx]}\n|u|_max={results["max_disp"][si]:.3f} mm', fontsize=10)
        ax_d.set_aspect('equal')
        ax_d.set_xticks([])
        ax_d.set_yticks([])

        ax_s.set_title(f'σ_max={results["max_stress"][si]:.0f} MPa', fontsize=10)
        ax_s.set_aspect('equal')
        ax_s.set_xticks([])
        ax_s.set_yticks([])

    axes[0,0].set_ylabel('Displacement norm |u|', fontsize=11, fontweight='bold')
    axes[1,0].set_ylabel('von Mises stress σ_vM', fontsize=11, fontweight='bold')

    fig2d.suptitle('Fig 9: L-Shaped Model — XVoxel FCM (p=3, d=3, Nitsche BC)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig2d.savefig(os.path.join(output_dir, 'fig9_composite.png'), dpi=200)
    plt.close(fig2d)
    print(f"  Saved fig9_composite.png (2D)")

    # ---- 图2: 3D 可视化 ----
    fig3d = plt.figure(figsize=(5.5*n_show, 8))
    
    # Gather voxel field data for each step
    for idx, si in enumerate(paper_steps):
        R = step_radii[idx]
        solid_voxels = build_voxel_geometry(xv, R)
        
        # Build voxel faces and colors for displacement
        ax_d3d = fig3d.add_subplot(2, n_show, idx + 1, projection='3d')
        _plot_voxel_3d(ax_d3d, xv, solid_voxels, results['nodal_disp'][si],
                       solver, vmin_d, vmax_d, '|u| (mm)', R,
                       f'{step_labels[idx]}\n|u|_max={results["max_disp"][si]:.3f} mm')
        
        # Build voxel faces and colors for stress
        ax_s3d = fig3d.add_subplot(2, n_show, idx + 1 + n_show, projection='3d')
        _plot_voxel_3d(ax_s3d, xv, solid_voxels, results['nodal_stress'][si],
                       solver, vmin_s, vmax_s, 'σ_vM (MPa)', R,
                       f'σ_max={results["max_stress"][si]:.0f} MPa', arc_only=True)

    fig3d.suptitle('Fig 9 (3D): L-Shaped Model — XVoxel FCM (p=3, d=3)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig3d.savefig(os.path.join(output_dir, 'fig9_3d.png'), dpi=200)
    plt.close(fig3d)
    print(f"  Saved fig9_3d.png (3D)")


def _plot_voxel_3d(ax, xv, solid_voxels, nodal_field, solver, vmin, vmax, clabel, corner_R, title, arc_only=False):
    """绘制所有外表面，按场值着色（云图）。

    显示坐标使用 (x, z, y)，使物理 y 方向在图中竖直显示，接近论文视角。
    arc_only=True 时，前后主面 + 侧壁显示灰色，仅圆角弧面显示场值云图。
    """
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    GRAY = np.array([0.65, 0.65, 0.65, 1.0])  # 灰色表面

    ox, oy, oz = xv.ox, xv.oy, xv.oz
    lz = xv.lz
    norm = Normalize(vmin=vmin, vmax=vmax)

    hres = 180
    xi = np.linspace(ox, ox + xv.lx, hres)
    yi = np.linspace(oy, oy + xv.ly, hres)
    XI, YI = np.meshgrid(xi, yi)

    l_shape = (
        ((XI >= 0) & (XI <= 3) & (YI >= 0) & (YI <= 15)) |
        ((XI >= 0) & (XI <= 15) & (YI >= 0) & (YI <= 3))
    )
    fillet = (
        (XI > 3) & (YI > 3) & (XI <= 3 + corner_R) & (YI <= 3 + corner_R) &
        (((XI - (3 + corner_R))**2 + (YI - (3 + corner_R))**2) > corner_R**2)
    )
    solid = l_shape | fillet

    # ---- 节点场 → 高分辨率网格插值 ----
    x_coords = solver.nodes[:, 0]
    y_coords = solver.nodes[:, 1]
    z_target = oz + lz / 2
    z_mask = np.abs(solver.nodes[:, 2] - z_target) < xv.dz
    points = np.column_stack([x_coords[z_mask], y_coords[z_mask]])
    values = nodal_field[z_mask] if nodal_field is not None else np.zeros(np.sum(z_mask))

    # field_grid: 仅固体内有值（用于前后主面）
    field_grid = griddata(points, values, (XI, YI), method='cubic', fill_value=0)
    field_grid = np.where(solid, field_grid, np.nan)

    # field_clean: 全域可查询（用于侧壁采样）
    field_clean = griddata(points, values, (XI, YI), method='cubic', fill_value=0)

    # ---- 前后主面 (z=zmax, z=zmin) ----
    # In arc_only mode, draw only the rounded cylindrical surface.
    face_solid = np.zeros_like(solid, dtype=bool) if arc_only else solid
    X_plot = np.where(face_solid, XI, np.nan)
    Y_front = np.full_like(XI, oz + lz)
    Y_back = np.full_like(XI, oz)
    Z_plot = np.where(face_solid, YI, np.nan)

    if arc_only:
        face_colors = np.tile(GRAY, (hres, hres, 1))
        face_colors[..., -1] = np.where(face_solid, 1.0, 0.0)
        back_colors = face_colors.copy()
        back_colors[..., :3] = back_colors[..., :3] * 0.85
    else:
        face_colors = plt.cm.jet(norm(field_grid))
        face_colors[..., -1] = np.where(solid, 1.0, 0.0)
        back_colors = face_colors.copy()
        back_colors[..., :3] = back_colors[..., :3] * 0.72 + 0.18
        back_colors[..., -1] = np.where(solid, 1.0, 0.0)

    front_surf = ax.plot_surface(X_plot, Y_front, Z_plot, facecolors=face_colors,
                                 linewidth=0, edgecolor='none', antialiased=False, shade=False)
    back_surf = ax.plot_surface(X_plot, Y_back, Z_plot, facecolors=back_colors,
                                linewidth=0, edgecolor='none', antialiased=False, shade=False)
    if arc_only:
        front_surf.set_zsort('min')
        back_surf.set_zsort('min')
        front_surf.set_sort_zpos(1e9)
        back_surf.set_sort_zpos(1e9)

    # ---- 侧壁：双线性采样场值并着色 ----
    def _sample_field(x_pts, y_pts):
        """在 field_clean 上双线性插值采样"""
        ix = np.clip(np.searchsorted(xi, x_pts) - 1, 0, hres - 2)
        iy = np.clip(np.searchsorted(yi, y_pts) - 1, 0, hres - 2)
        fx = np.clip((x_pts - xi[ix]) / (xi[ix + 1] - xi[ix]), 0, 1)
        fy = np.clip((y_pts - yi[iy]) / (yi[iy + 1] - yi[iy]), 0, 1)
        f00 = field_clean[iy, ix]
        f10 = field_clean[iy, ix + 1]
        f01 = field_clean[iy + 1, ix]
        f11 = field_clean[iy + 1, ix + 1]
        return (1 - fx) * (1 - fy) * f00 + fx * (1 - fy) * f10 + (1 - fx) * fy * f01 + fx * fy * f11

    def _add_x_wall(x_const, y0, y1):
        """x=常数 侧壁"""
        yy = np.linspace(y0, y1, hres)
        zz = np.array([oz, oz + lz])
        YY, ZZ = np.meshgrid(yy, zz)
        XX = np.full_like(YY, x_const)
        if arc_only:
            wall_colors = np.tile(GRAY[:3], (2, len(yy), 1))
            wall_colors = np.dstack([wall_colors, np.ones((2, len(yy)))])
        else:
            f_vals = _sample_field(np.full(hres, x_const), yy)
            F = np.tile(f_vals, (2, 1))
            wall_colors = plt.cm.jet(norm(F))
            wall_colors[..., -1] = 1.0
        ax.plot_surface(XX, ZZ, YY, facecolors=wall_colors,
                        linewidth=0, edgecolor='none', antialiased=False, shade=False)

    def _add_y_wall(y_const, x0, x1):
        """y=常数 侧壁"""
        xx = np.linspace(x0, x1, hres)
        zz = np.array([oz, oz + lz])
        XX, ZZ = np.meshgrid(xx, zz)
        YY = np.full_like(XX, y_const)
        if arc_only:
            wall_colors = np.tile(GRAY[:3], (2, len(xx), 1))
            wall_colors = np.dstack([wall_colors, np.ones((2, len(xx)))])
        else:
            f_vals = _sample_field(xx, np.full(hres, y_const))
            F = np.tile(f_vals, (2, 1))
            wall_colors = plt.cm.jet(norm(F))
            wall_colors[..., -1] = 1.0
        ax.plot_surface(XX, ZZ, YY, facecolors=wall_colors,
                        linewidth=0, edgecolor='none', antialiased=False, shade=False)

    if not arc_only:
        # x=常数侧壁
        _add_x_wall(0, 0, 15)                  # 竖直臂左面
        _add_x_wall(3, 3 + corner_R, 15)       # 竖直臂右面（圆角以上）
        _add_x_wall(15, 0, 3)                  # 水平臂右面

        # y=常数侧壁
        _add_y_wall(0, 0, 15)                  # 水平臂底面
        _add_y_wall(3, 3 + corner_R, 15)       # 水平臂顶面（圆角以右）
        _add_y_wall(15, 0, 3)                  # 竖直臂顶面

    # ---- 圆弧柱面：圆弧沿厚度方向扫出的真实柱面 ----
    theta = np.linspace(np.pi, 3 * np.pi / 2, hres * 2)
    z_samples = np.linspace(oz, oz + lz, 32 if arc_only else 2)
    THETA, Z_SURF = np.meshgrid(theta, z_samples)
    arc_dx = arc_dy = arc_dz = 0.0
    X_SURF = (3 + corner_R) + corner_R * np.cos(THETA)
    Y_SURF = (3 + corner_R) + corner_R * np.sin(THETA)
    # Sample the field on the true arc and draw it only on the cylinder.
    arc_x = X_SURF[0, :]
    arc_y = Y_SURF[0, :]
    field_arc_x = arc_x - arc_dx
    field_arc_y = arc_y - arc_dy
    f_arc = _sample_field(field_arc_x, field_arc_y)
    F_arc = np.tile(f_arc, (len(z_samples), 1))
    arc_colors = plt.cm.jet(norm(F_arc))
    arc_colors[..., -1] = 1.0

    arc_surf = ax.plot_surface(X_SURF, Z_SURF, Y_SURF, facecolors=arc_colors,
                               linewidth=0, edgecolor='none', antialiased=False, shade=False,
                               rstride=1, cstride=1)
    if arc_only:
        arc_surf.set_zsort('max')
        arc_surf.set_sort_zpos(-1e9)

    if arc_only:
        # 用边界线强调这是一个圆弧柱面，而不是平面补丁。
        edge_color = (0.15, 0.15, 0.15)
        ax.plot(arc_x, np.full_like(arc_x, oz + arc_dz), arc_y,
                color=edge_color, linewidth=0.8)
        ax.plot(arc_x, np.full_like(arc_x, oz + lz + arc_dz), arc_y,
                color=edge_color, linewidth=0.8)
        for theta_end in (np.pi, 3 * np.pi / 2):
            x_end = (3 + corner_R) + corner_R * np.cos(theta_end) + arc_dx
            y_end = (3 + corner_R) + corner_R * np.sin(theta_end) + arc_dy
            edge_z_samples = z_samples + arc_dz
            ax.plot(np.full_like(edge_z_samples, x_end), edge_z_samples,
                    np.full_like(edge_z_samples, y_end),
                    color=edge_color, linewidth=0.8)

    # ---- 坐标轴设置 ----
    ax.set_xlim(ox, ox + xv.lx)
    ax.set_ylim(oz, oz + lz)
    ax.set_zlim(oy, oy + xv.ly)
    ax.set_box_aspect((xv.lx, lz, xv.ly))
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()

    mappable = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    mappable.set_array([])
    plt.colorbar(mappable, ax=ax, shrink=0.55, pad=0.02, label=clabel)

    if arc_only:
        ax.view_init(elev=22, azim=-35)
    else:
        ax.view_init(elev=18, azim=-55)


def run_fig7_simulation(save_results=True):
    """运行 Fig 7 的完整仿真流程"""
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Building L-shaped XVoxel model (CSG)...")
    print("=" * 60)
    xv, fids = build_lshape_model()

    print("\n" + "=" * 60)
    print("Initializing FCM solver (p=3 Hex32)...")
    print("=" * 60)
    solver = XVoxelFEMSolver(xv, E=E, nu=NU, alpha=ALPHA, element_order=3)
    print(f"  Nodes: {solver.n_nodes}, DOFs: {solver.ndof}")
    n_s = np.sum(xv.voxel_nature == 1)
    n_v = np.sum(xv.voxel_nature == -1)
    n_b = np.sum(xv.voxel_nature == 0)
    print(f"  Voxels: {xv.n_voxels} (solid={n_s}, void={n_v}, boundary={n_b})")

    # 边界条件: Nitsche弱BC固定上表面 (y=ymax), 右侧面(x=xmax)向下牵引力
    print("\n  BCs: Nitsche weak BC at upper face (y=max), traction on right face (x=max)")

    # 定义编辑序列
    steps = build_step_sequence(xv, fids)

    # 存储结果
    results = {
        'step_names': [],
        'displacements': [],
        'von_mises': [],
        'nodal_disp': [],
        'nodal_stress': [],
        'active_voxels': [],
        'solve_times': [],
        'solid_counts': [],
        'boundary_counts': [],
        'void_counts': [],
        'voxel_nature_history': [],
        'active_voxel_masks': [],
        'max_disp': [],
        'max_stress': [],
        'solver': solver,  # 保存solver引用用于可视化
    }

    # 运行 5 步
    for step_idx, (step_name, edits) in enumerate(steps):
        print(f"\n{'='*60}")
        print(f"{step_name}")
        print(f"{'='*60}")

        if edits:
            active_set = set()
            for fid, param, val in edits:
                av = xv.edit_parameter(fid, param, val)
                active_set.update(av)
            print(f"  Edits: {edits}")
            print(f"  Affected (active) voxels: {len(active_set)}")
        else:
            active_set = None
            print(f"  No edits (initial state)")

        # 统计
        n_s = np.sum(xv.voxel_nature == 1)
        n_v = np.sum(xv.voxel_nature == -1)
        n_b = np.sum(xv.voxel_nature == 0)
        results['solid_counts'].append(int(n_s))
        results['void_counts'].append(int(n_v))
        results['boundary_counts'].append(int(n_b))
        results['active_voxels'].append(len(active_set) if active_set else 0)
        results['voxel_nature_history'].append(xv.voxel_nature.copy())

        if active_set:
            mask = np.zeros(xv.n_voxels, dtype=bool)
            mask[list(active_set)] = True
            results['active_voxel_masks'].append(mask)
        else:
            results['active_voxel_masks'].append(np.zeros(xv.n_voxels, dtype=bool))

        # 装配和求解
        t0 = time.time()
        print(f"  Assembling stiffness matrix ({xv.n_voxels} elements)...")
        try:
            K, F = solver.assemble_FCM_system()

            # Nitsche 弱 Dirichlet BC (固定上表面 y=ymax)
            K, F = solver.apply_nitsche_dirichlet(K, F, 'ymax')

            # 施加右侧面面载荷 (向下)
            F_traction = get_face_traction_nodal_forces(
                solver, 'xmax', (0, TRACTION_Y, 0)
            )
            F += F_traction
            print(f"  Total traction force magnitude: {np.linalg.norm(F):.2e}")

            # 求解
            print(f"  Solving sparse system ({solver.ndof} DOFs)...")
            u = solver.solve(K, F)

            t1 = time.time()
            solve_time = t1 - t0
            results['solve_times'].append(solve_time)

            # 计算单元结果
            disp_norm, von_mises = solver.compute_element_results(u)
            results['displacements'].append(disp_norm)
            results['von_mises'].append(von_mises)
            
            # 计算节点级场（用于光滑云图）
            nodal_disp, nodal_stress = compute_nodal_fields(solver, u, von_mises)
            results['nodal_disp'].append(nodal_disp)
            results['nodal_stress'].append(nodal_stress)
            
            results['step_names'].append(step_name)
            results['max_disp'].append(float(disp_norm.max()))
            results['max_stress'].append(float(von_mises.max()))
            # Track corner radius for per-step SDF mask
            corner_r = xv.features[fids['corner']].primitive.r
            results.setdefault('corner_radii', []).append(corner_r)

            print(f"  Solved in {solve_time:.2f}s")
            print(f"  max|u| = {disp_norm.max():.6e} mm")
            print(f"  max σ_vM = {von_mises.max():.2e} MPa")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results['displacements'].append(None)
            results['von_mises'].append(None)
            results['nodal_disp'].append(None)
            results['nodal_stress'].append(None)
            results['solve_times'].append(None)
            results['step_names'].append(step_name)
            results['max_disp'].append(0)
            results['max_stress'].append(0)

    # 保存数据
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'fig7_lshape_results.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {data_path}")

    # 生成可视化
    print("\n" + "=" * 60)
    print("Generating Fig 9 style contour plots...")
    print("=" * 60)
    vis_dir = os.path.join(output_dir, 'output')
    os.makedirs(vis_dir, exist_ok=True)
    plot_results(results, vis_dir)
    print(f"Visualizations saved to {vis_dir}")

    return results, vis_dir


if __name__ == "__main__":
    results, vis_dir = run_fig7_simulation(save_results=True)
    print(f"\n{'='*60}")
    print(f"All done! Visualizations in: {vis_dir}")
    print(f"{'='*60}")
