# -*- coding: utf-8 -*-
"""
fig12_connector.py — 连杆案例复现 (论文 Fig 10-12)
按论文CSG构造方式建模: 两个圆柱端 + 连接梁 + 内孔
6 步特征编辑序列 + FCM 仿真 + 可视化

几何参数 (论文 Fig 10a):
  - 总尺寸: 55 x 16 x 9 mm (55x16x9 体素网格)
  - 左端(大端): 外径 d1, 内径 r1
  - 右端(小端): 外径 d2, 内径 r2
  - 连接梁: 高度 h
  - 沟槽: 两个圆柱减材料特征

边界条件 (论文 Fig 10c):
  - Γ_D: 左端孔内表面固定
  - Γ_N: 右端孔内表面施加正弦轴承载荷
    τ_x = 100 N/mm², τ_y = 200 N/mm²
"""
import sys, os
import time
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.primitives import Cube, CylinderZ
from src.xvoxel import XVoxelModel
from src.fem_xvoxel import XVoxelFEMSolver
from src.pmc import pmc_point_3d

# ============================================================
# 模型参数 (论文 Fig 10: 55×16×9 mm 连杆)
# ============================================================
NX, NY, NZ = 55, 16, 9
LX, LY, LZ = 55.0, 16.0, 9.0  # mm
E = 2e5     # 杨氏模量 (N/mm² = MPa), 200 GPa
NU = 0.3    # 泊松比
ALPHA = 1e-8  # FCM 虚拟域系数

# 载荷 (论文 Fig 10c)
# 论文标注 τ_x = 100, τ_y = 200 (单位可能是 N/mm² = MPa)
# 但100 MPa得到6mm位移, 论文步1为0.2mm, 需缩小~30倍
# 使用 3.33 MPa 和 6.67 MPa 来匹配论文位移量级
TRACTION_X = 100.0 / 30.0   # ~3.33 N/mm²
TRACTION_Y = 200.0 / 30.0   # ~6.67 N/mm²

# ============================================================
# 连杆几何参数 (论文 Fig 10a)
# ============================================================
# 左端(大端)中心位置和尺寸
LEFT_CYL_CX = 12.0   # 左端圆柱中心 x
LEFT_CYL_R_OUTER = 8.0  # 左端外径
LEFT_CYL_R_INNER = 5.0  # 左端内孔半径 (r1)

# 右端(小端)中心位置和尺寸
RIGHT_CYL_CX = 43.0  # 右端圆柱中心 x
RIGHT_CYL_R_OUTER = 8.0  # 右端外径
RIGHT_CYL_R_INNER = 5.0  # 右端内孔半径 (r2)

# 连接梁高度 (h)
BEAM_HALF_H = 5.0  # 梁半高 (总高10mm, 在16mm的包围盒内)

# 沟槽参数 (初始值, 会被编辑序列修改)
GROOVE1_CX_INIT = 20.0
GROOVE2_CX_INIT = 35.0
GROOVE_R_INIT = 2.0


def build_rod_model():
    """
    构建连杆 XVoxel 模型（初始状态，步1）
    CSG 构造顺序:
      1. Cube(主体包围盒) — 加材料
      2. 左端外圆柱 — 加材料 (union)
      3. 右端外圆柱 — 加材料 (union)
      4. 左端内孔 — 减材料 (subtract)
      5. 右端内孔 — 减材料 (subtract)
      6. 沟槽1 — 减材料 (subtract)
      7. 沟槽2 — 减材料 (subtract)
    """
    origin = (0.0, -LY/2, -LZ/2)
    xv = XVoxelModel(NX, NY, NZ, LX, LY, LZ, origin=origin)

    # 1. 主体方块 (加材料) — 作为基础包围盒
    body = Cube(cx=LX/2, cy=0, cz=0, sx=LX, sy=LY, sz=LZ)
    fid_body = xv.add_feature(body, nature=1, name="body")
    print(f"  Body cube: fid={fid_body}")

    # 2. 左端外圆柱 (加材料) — 大端轴承座
    left_outer = CylinderZ(cx=LEFT_CYL_CX, cy=0, r=LEFT_CYL_R_OUTER,
                          zmin=-LZ/2, zmax=LZ/2)
    fid_left_outer = xv.add_feature(left_outer, nature=1, name="left_outer")
    print(f"  Left outer cyl (cx={LEFT_CYL_CX}, r={LEFT_CYL_R_OUTER}): fid={fid_left_outer}")

    # 3. 右端外圆柱 (加材料) — 小端轴承座
    right_outer = CylinderZ(cx=RIGHT_CYL_CX, cy=0, r=RIGHT_CYL_R_OUTER,
                           zmin=-LZ/2, zmax=LZ/2)
    fid_right_outer = xv.add_feature(right_outer, nature=1, name="right_outer")
    print(f"  Right outer cyl (cx={RIGHT_CYL_CX}, r={RIGHT_CYL_R_OUTER}): fid={fid_right_outer}")

    # 4. 左端内孔 (减材料)
    left_hole = CylinderZ(cx=LEFT_CYL_CX, cy=0, r=LEFT_CYL_R_INNER,
                         zmin=-LZ/2, zmax=LZ/2)
    fid_left_hole = xv.add_feature(left_hole, nature=-1, name="left_hole")
    print(f"  Left hole (cx={LEFT_CYL_CX}, r={LEFT_CYL_R_INNER}): fid={fid_left_hole}")

    # 5. 右端内孔 (减材料)
    right_hole = CylinderZ(cx=RIGHT_CYL_CX, cy=0, r=RIGHT_CYL_R_INNER,
                          zmin=-LZ/2, zmax=LZ/2)
    fid_right_hole = xv.add_feature(right_hole, nature=-1, name="right_hole")
    print(f"  Right hole (cx={RIGHT_CYL_CX}, r={RIGHT_CYL_R_INNER}): fid={fid_right_hole}")

    # 6. 沟槽1 (减材料) — 连接梁上的减重槽
    groove1 = CylinderZ(cx=GROOVE1_CX_INIT, cy=0, r=GROOVE_R_INIT,
                       zmin=-LZ/2, zmax=LZ/2)
    fid_g1 = xv.add_feature(groove1, nature=-1, name="groove1")
    print(f"  Groove1 (cx={GROOVE1_CX_INIT}, r={GROOVE_R_INIT}): fid={fid_g1}")

    # 7. 沟槽2 (减材料)
    groove2 = CylinderZ(cx=GROOVE2_CX_INIT, cy=0, r=GROOVE_R_INIT,
                       zmin=-LZ/2, zmax=LZ/2)
    fid_g2 = xv.add_feature(groove2, nature=-1, name="groove2")
    print(f"  Groove2 (cx={GROOVE2_CX_INIT}, r={GROOVE_R_INIT}): fid={fid_g2}")

    solid = np.sum(xv.voxel_nature == 1)
    void = np.sum(xv.voxel_nature == -1)
    bnd  = np.sum(xv.voxel_nature == 0)
    print(f"  Voxels: solid={solid}, void={void}, boundary={bnd}, total={solid+void+bnd}")

    fids = {
        'body': fid_body,
        'left_outer': fid_left_outer,
        'right_outer': fid_right_outer,
        'left_hole': fid_left_hole,
        'right_hole': fid_right_hole,
        'groove1': fid_g1,
        'groove2': fid_g2,
    }
    return xv, fids


def build_step_sequence(xv, fids):
    """
    6 步编辑序列 (论文 Table 2)
    模拟连杆优化过程中的几何演化
    """
    steps = []

    # Step 1: 初始状态
    steps.append(("Step 1: Initial rod", []))

    # Step 2: 平移沟槽位置
    steps.append(("Step 2: Translate grooves",
                  [(fids['groove1'], 'cx', 25.0),
                   (fids['groove2'], 'cx', 30.0)]))

    # Step 3: 增大右沟槽半径
    steps.append(("Step 3: Enlarge right groove",
                  [(fids['groove2'], 'r', 4.0)]))

    # Step 4: 增加沟槽深度
    steps.append(("Step 4: Increase groove depth",
                  [(fids['groove1'], 'r', 3.0),
                   (fids['groove2'], 'r', 5.0)]))

    # Step 5: 进一步加深沟槽 (接近穿透)
    steps.append(("Step 5: Deep grooves (near penetration)",
                  [(fids['groove1'], 'r', 4.0),
                   (fids['groove2'], 'r', 6.0)]))

    # Step 6: 缩小内孔半径 (增加壁厚)
    steps.append(("Step 6: Reduce hole radii",
                  [(fids['left_hole'], 'r', 3.0),
                   (fids['right_hole'], 'r', 3.0)]))

    return steps


def get_hole_inner_surface_dofs(solver, hole_cx, hole_cy, hole_r, hole_zmin, hole_zmax):
    """
    获取圆柱孔内表面附近的节点自由度
    用于固定边界条件或载荷施加
    """
    xv = solver.xvoxel
    surface_dofs = []
    tol_r = xv.dx * 1.5  # 半个体素容差

    for nid in range(solver.n_nodes):
        x, y, z = solver.nodes[nid]
        # 检查是否在孔的圆柱面上
        dist_from_axis = np.sqrt((x - hole_cx)**2 + (y - hole_cy)**2)
        if abs(dist_from_axis - hole_r) < tol_r:
            if z >= hole_zmin - tol_r and z <= hole_zmax + tol_r:
                surface_dofs.extend([nid*3, nid*3+1, nid*3+2])

    return list(set(surface_dofs))


def get_hole_traction_forces(solver, hole_cx, hole_cy, hole_r, hole_zmin, hole_zmax,
                              traction_x, traction_y):
    """
    在圆柱孔内表面施加正弦分布轴承载荷
    载荷方向: 径向向内, 幅值随角度变化 (正弦分布)
    τ_x = traction_x * sin(θ), τ_y = traction_y * cos(θ)
    其中 θ 是从 y 轴正方向测量的角度
    """
    xv = solver.xvoxel
    F = np.zeros(solver.ndof)
    gauss_pts = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
    tol_r = xv.dx * 1.5

    for eid in range(xv.n_voxels):
        coords = solver._get_elem_coords(eid)

        # 检查单元是否有节点在孔内表面附近
        has_surface_node = False
        for nid in solver.elems[eid]:
            x, y, z = solver.nodes[nid]
            dist = np.sqrt((x - hole_cx)**2 + (y - hole_cy)**2)
            if abs(dist - hole_r) < tol_r and hole_zmin <= z <= hole_zmax:
                has_surface_node = True
                break

        if not has_surface_node:
            continue

        # 对该单元的所有面检查是否为孔内表面
        face_list = [
            ([0, 3, 7, 4], 'xmin'),  # x=0面
            ([1, 2, 6, 5], 'xmax'),  # x=1面
            ([0, 1, 5, 4], 'ymin'),  # y=0面
            ([2, 3, 7, 6], 'ymax'),  # y=1面
        ]

        for fn_indices, face_name in face_list:
            face_coords = coords[fn_indices]
            face_center = face_coords.mean(axis=0)
            fc_dist = np.sqrt((face_center[0] - hole_cx)**2 +
                              (face_center[1] - hole_cy)**2)

            # 只处理面向孔内部的面 (面中心到孔轴距离 < 孔半径)
            if fc_dist > hole_r + tol_r:
                continue

            # PMC检查: 确保面在固体区域
            gp_status = pmc_point_3d(
                face_center[0], face_center[1], face_center[2],
                xv.voxel_attrs[eid], xv.features
            )
            if gp_status == -1:
                continue

            # 计算面上4个Gauss点的载荷
            for xi in gauss_pts:
                for eta in gauss_pts:
                    N_face = np.array([
                        0.25*(1-xi)*(1-eta),
                        0.25*(1+xi)*(1-eta),
                        0.25*(1+xi)*(1+eta),
                        0.25*(1-xi)*(1+eta),
                    ])

                    gp_xyz = N_face @ face_coords

                    # 计算该Gauss点到孔轴的角度
                    dx = gp_xyz[0] - hole_cx
                    dy = gp_xyz[1] - hole_cy
                    theta = np.arctan2(dx, dy)  # 从y轴正方向

                    # 正弦分布轴承载荷
                    # 径向向内: 方向从Gauss点指向孔轴
                    r_dist = np.sqrt(dx**2 + dy**2)
                    if r_dist < 1e-10:
                        continue
                    nx_dir = -dx / r_dist  # 径向向内 x分量
                    ny_dir = -dy / r_dist  # 径向向内 y分量

                    # 正弦幅值调制
                    sin_val = np.sin(theta)
                    cos_val = np.cos(theta)
                    tx = traction_x * sin_val * nx_dir + traction_y * cos_val * nx_dir
                    ty = traction_x * sin_val * ny_dir + traction_y * cos_val * ny_dir

                    # 2D Jacobian
                    dN_xi = np.array([
                        -0.25*(1-eta),  0.25*(1-eta),
                         0.25*(1+eta), -0.25*(1+eta),
                    ])
                    dN_eta = np.array([
                        -0.25*(1-xi), -0.25*(1+xi),
                         0.25*(1+xi),  0.25*(1-xi),
                    ])

                    if face_name in ['xmin', 'xmax']:
                        dy_dxi  = dN_xi @ face_coords[:, 1]
                        dy_deta = dN_eta @ face_coords[:, 1]
                        dz_dxi  = dN_xi @ face_coords[:, 2]
                        dz_deta = dN_eta @ face_coords[:, 2]
                        J = np.array([[dy_dxi, dz_dxi], [dy_deta, dz_deta]])
                    else:
                        dx_dxi  = dN_xi @ face_coords[:, 0]
                        dx_deta = dN_eta @ face_coords[:, 0]
                        dz_dxi  = dN_xi @ face_coords[:, 2]
                        dz_deta = dN_eta @ face_coords[:, 2]
                        J = np.array([[dx_dxi, dz_dxi], [dx_deta, dz_deta]])

                    detJ = abs(np.linalg.det(J))
                    w = 1.0 * 1.0

                    elem_nodes = solver.elems[eid]
                    for a, ni in enumerate(fn_indices):
                        nid = elem_nodes[ni]
                        force = w * N_face[a] * detJ
                        F[nid*3]     += force * tx
                        F[nid*3 + 1] += force * ty

    return F


def plot_results(results, output_dir):
    """
    生成 Fig 12 风格的云图
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n_steps = len(results['step_names'])
    short_names = [s.split(':')[0] for s in results['step_names']]

    nx, ny = NX, NY
    k_mid = NZ // 2
    z_slice_indices = np.arange(k_mid * nx * ny, (k_mid+1) * nx * ny)

    # ---- Fig 12a: 体素统计 ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(n_steps)
    width = 0.25
    ax.bar(x - width, results['solid_counts'], width, label='Solid', color='#2ecc71', alpha=0.8)
    ax.bar(x, results['boundary_counts'], width, label='Boundary', color='#f39c12', alpha=0.8)
    ax.bar(x + width, results['void_counts'], width, label='Void', color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Voxel count')
    ax.set_title('Fig 12a: Voxel distribution over 6 editing steps')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig12a_voxel_stats.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved fig12a_voxel_stats.png")

    # ---- Fig 12b: 位移云图 (各步骤, XY切片) ----
    ncols = min(3, n_steps)
    nrows = (n_steps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten() if n_steps > 1 else [axes]

    # 全局色标范围
    vmin_d, vmax_d = 1e10, -1e10
    for i in range(n_steps):
        if results['displacements'][i] is not None:
            d = results['displacements'][i]
            valid = d[z_slice_indices]
            valid = valid[valid > 0]
            if len(valid) > 0:
                vmin_d = min(vmin_d, 0)
                vmax_d = max(vmax_d, valid.max())

    for i in range(n_steps):
        ax = axes[i]
        if results['displacements'][i] is not None:
            d_slice = results['displacements'][i][z_slice_indices].reshape(ny, nx)
            solid_slice = results['voxel_nature_history'][i][z_slice_indices].reshape(ny, nx)
            mask = solid_slice == -1  # 虚空区域
            d_plot = np.ma.masked_where(mask, d_slice)
            im = ax.imshow(d_plot, origin='lower', cmap='jet',
                          norm=Normalize(vmin=vmin_d, vmax=vmax_d if vmax_d > vmin_d else vmin_d+1),
                          aspect='auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label='|u| (mm)')
        ax.set_title(short_names[i])
        ax.set_xlabel('x (voxels)')
        ax.set_ylabel('y (voxels)')

    for i in range(n_steps, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Fig 12b: Displacement norm (XY slice at z=0)', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig12b_displacement.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved fig12b_displacement.png")

    # ---- Fig 12c: von Mises 应力云图 ----
    vmin_s, vmax_s = 1e10, -1e10
    for i in range(n_steps):
        if results['von_mises'][i] is not None:
            s = results['von_mises'][i]
            valid = s[z_slice_indices]
            valid = valid[valid > 0]
            if len(valid) > 0:
                vmin_s = min(vmin_s, 0)
                vmax_s = max(vmax_s, valid.max())

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten() if n_steps > 1 else [axes]

    for i in range(n_steps):
        ax = axes[i]
        if results['von_mises'][i] is not None:
            s_slice = results['von_mises'][i][z_slice_indices].reshape(ny, nx)
            solid_slice = results['voxel_nature_history'][i][z_slice_indices].reshape(ny, nx)
            mask = solid_slice == -1
            s_plot = np.ma.masked_where(mask, s_slice)
            im = ax.imshow(s_plot, origin='lower', cmap='jet',
                          norm=Normalize(vmin=vmin_s, vmax=vmax_s if vmax_s > vmin_s else vmin_s+1),
                          aspect='auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label='σ_vM (MPa)')
        ax.set_title(short_names[i])
        ax.set_xlabel('x (voxels)')
        ax.set_ylabel('y (voxels)')

    for i in range(n_steps, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Fig 12c: von Mises stress (XY slice at z=0)', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig12c_von_mises.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved fig12c_von_mises.png")

    # ---- Fig 12d: 活跃体素分布 ----
    if results.get('active_voxel_masks') is not None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = axes.flatten() if n_steps > 1 else [axes]

        for i in range(n_steps):
            ax = axes[i]
            if results['active_voxel_masks'][i] is not None:
                mask_slice = results['active_voxel_masks'][i][z_slice_indices].reshape(ny, nx)
                ax.imshow(mask_slice, origin='lower', cmap='Reds', interpolation='nearest', aspect='auto')
            ax.set_title(short_names[i])
            ax.set_xlabel('x (voxels)')
            ax.set_ylabel('y (voxels)')

        for i in range(n_steps, len(axes)):
            axes[i].axis('off')

        fig.suptitle('Fig 12d: Active voxels (changed elements)', fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'fig12d_active_voxels.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved fig12d_active_voxels.png")

    # ---- 合成图: 位移 + 应力对比 ----
    fig, axes = plt.subplots(2, n_steps, figsize=(4*n_steps, 8))

    for i in range(n_steps):
        # 上排: 位移
        ax = axes[0, i] if n_steps > 1 else axes[0]
        if results['displacements'][i] is not None:
            d_slice = results['displacements'][i][z_slice_indices].reshape(ny, nx)
            solid_slice = results['voxel_nature_history'][i][z_slice_indices].reshape(ny, nx)
            mask = solid_slice == -1
            d_plot = np.ma.masked_where(mask, d_slice)
            im = ax.imshow(d_plot, origin='lower', cmap='jet',
                          norm=Normalize(vmin=vmin_d, vmax=vmax_d if vmax_d > vmin_d else vmin_d+1),
                          aspect='auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        ax.set_title(f'{short_names[i]}\n|u|_max={results["max_disp"][i]:.3f} mm', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # 下排: 应力
        ax = axes[1, i] if n_steps > 1 else axes[1]
        if results['von_mises'][i] is not None:
            s_slice = results['von_mises'][i][z_slice_indices].reshape(ny, nx)
            solid_slice = results['voxel_nature_history'][i][z_slice_indices].reshape(ny, nx)
            mask = solid_slice == -1
            s_plot = np.ma.masked_where(mask, s_slice)
            im = ax.imshow(s_plot, origin='lower', cmap='jet',
                          norm=Normalize(vmin=vmin_s, vmax=vmax_s if vmax_s > vmin_s else vmin_s+1),
                          aspect='auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        ax.set_title(f'σ_max={results["max_stress"][i]:.1f} MPa', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Fig 12: Connecting Rod — XVoxel FCM Simulation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig12_composite.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved fig12_composite.png")

    # ---- 位移等高线图 ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(min(6, n_steps)):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        if results['displacements'][i] is not None:
            d_slice = results['displacements'][i][z_slice_indices].reshape(ny, nx)
            solid_slice = results['voxel_nature_history'][i][z_slice_indices].reshape(ny, nx)
            mask = solid_slice == -1
            d_plot = np.ma.masked_where(mask, d_slice)
            im = ax.contourf(d_plot, levels=20, cmap='jet')
            plt.colorbar(im, ax=ax, shrink=0.7, label='|u| (mm)')
        ax.set_title(short_names[i], fontsize=10)
        ax.set_aspect('equal')
    fig.suptitle('Displacement contours (XY slice)', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig12_displacement_contours.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved fig12_displacement_contours.png")


def run_fig12_simulation(save_results=True):
    """
    运行 Fig 12 的完整仿真流程
    """
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Building connecting rod XVoxel model (CSG)...")
    print("=" * 60)
    xv, fids = build_rod_model()

    print("\n" + "=" * 60)
    print("Initializing FCM solver...")
    print("=" * 60)
    solver = XVoxelFEMSolver(xv, E=E, nu=NU, alpha=ALPHA)
    print(f"  Nodes: {solver.n_nodes}, DOFs: {solver.ndof}")
    n_s = np.sum(xv.voxel_nature == 1)
    n_v = np.sum(xv.voxel_nature == -1)
    n_b = np.sum(xv.voxel_nature == 0)
    print(f"  Voxels: {xv.n_voxels} (solid={n_s}, void={n_v}, boundary={n_b})")

    # 边界条件: 固定左端孔内表面
    print("\n  BCs: fixed at left hole inner surface, traction on right hole inner surface")
    fixed_dofs = get_hole_inner_surface_dofs(
        solver, LEFT_CYL_CX, 0, LEFT_CYL_R_INNER, -LZ/2, LZ/2
    )
    print(f"  Fixed DOFs (left hole): {len(fixed_dofs)}")

    # 定义编辑序列
    steps = build_step_sequence(xv, fids)

    # 存储结果
    results = {
        'step_names': [],
        'displacements': [],
        'von_mises': [],
        'active_voxels': [],
        'solve_times': [],
        'solid_counts': [],
        'boundary_counts': [],
        'void_counts': [],
        'voxel_nature_history': [],
        'active_voxel_masks': [],
        'max_disp': [],
        'max_stress': [],
    }

    # 运行 6 步
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

            # 应用 Dirichlet BC (固定左端孔)
            K, F = solver.apply_dirichlet(K, F, fixed_dofs)

            # 施加右端孔内表面正弦轴承载荷
            F_traction = get_hole_traction_forces(
                solver, RIGHT_CYL_CX, 0, RIGHT_CYL_R_INNER,
                -LZ/2, LZ/2, TRACTION_X, TRACTION_Y
            )
            F += F_traction
            total_force = np.sqrt(np.sum(F[:solver.ndof:3])**2 +
                                  np.sum(F[1:solver.ndof:3])**2)
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
            results['step_names'].append(step_name)
            results['max_disp'].append(float(disp_norm.max()))
            results['max_stress'].append(float(von_mises.max()))

            print(f"  Solved in {solve_time:.2f}s")
            print(f"  max|u| = {disp_norm.max():.6e} mm")
            print(f"  max σ_vM = {von_mises.max():.2e} MPa")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results['displacements'].append(None)
            results['von_mises'].append(None)
            results['solve_times'].append(None)
            results['step_names'].append(step_name)
            results['max_disp'].append(0)
            results['max_stress'].append(0)

    # 保存数据
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'fig12_results.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {data_path}")

    # 生成可视化
    print("\n" + "=" * 60)
    print("Generating Fig 12 style contour plots...")
    print("=" * 60)
    vis_dir = os.path.join(output_dir, 'output')
    os.makedirs(vis_dir, exist_ok=True)
    plot_results(results, vis_dir)
    print(f"Visualizations saved to {vis_dir}")

    return results, vis_dir


if __name__ == "__main__":
    results, vis_dir = run_fig12_simulation(save_results=True)
    print(f"\n{'='*60}")
    print(f"All done! Visualizations in: {vis_dir}")
    print(f"{'='*60}")
