# -*- coding: utf-8 -*-
"""
fcm/boundary.py — 边界条件统一模块

支持:
    - Dirichlet BC (强施加)
    - Traction (面载荷, 带 PMC 过滤)
    - Nitsche 弱 Dirichlet BC (TODO: Phase 2)
"""
import numpy as np
from typing import List, Optional, Callable
from .elements import hex20_shape_func, GAUSS_2X2X2


def apply_dirichlet(K, F: np.ndarray,
                    fixed_dofs: np.ndarray,
                    prescribed_vals: Optional[np.ndarray] = None):
    """强施加 Dirichlet BC (消行消列法). 原地修改 K 和 F.

    Args:
        K: 全局刚度矩阵 (lil_matrix 或 csr_matrix, 被原地修改).
        F: 全局载荷向量 (被修改).
        fixed_dofs: (M,) 固定 DOF 索引.
        prescribed_vals: (M,) 指定位移值, 默认 0.

    Returns:
        (K, F) — 修改后的矩阵和向量 (可能为 lil 格式).
    """
    if prescribed_vals is None:
        prescribed_vals = np.zeros(len(fixed_dofs))

    # 确保是 lil 格式以便原地修改
    if hasattr(K, 'tolil'):
        K = K.tolil()

    for idx, dof in enumerate(fixed_dofs):
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1.0
        F[dof] = prescribed_vals[idx]

    return K, F


def get_face_traction_nodal_forces(
    mesh,
    face_name: str,
    traction: tuple,  # (tx, ty, tz) in N/mm²
    elem_mask: Optional[np.ndarray] = None,
    npe: int = 8,
    csg_root=None,
) -> np.ndarray:
    """计算面载荷对应的等效节点力 (带 PMC 过滤).

    Args:
        mesh: UniformHexMesh 实例
        face_name: 'xmin'/'xmax'/'ymin'/'ymax'/'zmin'/'zmax'
        traction: (tx, ty, tz) 面牵引力
        elem_mask: 可选, 限制到特定单元
        npe: 单元节点数 (8/20/32)
        csg_root: CSG 树根, 用于 PMC 过滤 (可选)

    Returns:
        (ndof,) 全局节点力向量
    """
    tx, ty, tz = traction
    F = np.zeros(mesh.ndof, dtype=np.float64)

    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz

    # 确定面和面单元
    if face_name == 'xmin':
        face_eids = np.arange(0, nx*ny*nz, nx)
        face_xi, face_eta = 0, 1  # η, ζ
        face_normal_sign = -1
    elif face_name == 'xmax':
        face_eids = np.arange(nx-1, nx*ny*nz, nx)
        face_xi, face_eta = 0, 1
        face_normal_sign = 1
    elif face_name == 'ymin':
        face_eids = np.array([k*nx*ny + j*nx + i for k in range(nz) for j in [0] for i in range(nx)])
        face_xi, face_eta = 0, 2  # ξ, ζ
        face_normal_sign = -1
    elif face_name == 'ymax':
        face_eids = np.array([k*nx*ny + j*nx + i for k in range(nz) for j in [ny-1] for i in range(nx)])
        face_xi, face_eta = 0, 2
        face_normal_sign = 1
    elif face_name == 'zmin':
        face_eids = np.array([k*nx*ny + j*nx + i for k in [0] for j in range(ny) for i in range(nx)])
        face_xi, face_eta = 0, 1  # ξ, η
        face_normal_sign = -1
    elif face_name == 'zmax':
        face_eids = np.array([k*nx*ny + j*nx + i for k in [nz-1] for j in range(ny) for i in range(nx)])
        face_xi, face_eta = 0, 1
        face_normal_sign = 1
    else:
        raise ValueError(f"Unknown face: {face_name}")

    if elem_mask is not None:
        face_eids = np.intersect1d(face_eids, np.where(elem_mask)[0])

    # 使用 2×2 Gauss 积分计算面载荷
    gauss_pts = GAUSS_2X2X2.points[:4, :2]  # 4 个 2D Gauss 点
    gauss_wts = GAUSS_2X2X2.weights[:4]

    for eid in face_eids:
        coords = mesh.get_elem_coords(eid)
        elem_nodes = mesh.elems[eid]

        for gp in range(4):
            a, b = gauss_pts[gp]
            w_gauss = gauss_wts[gp]

            # 确定 3D 自然坐标
            if face_name in ('xmin', 'xmax'):
                xi_nat = -1.0 if face_name == 'xmin' else 1.0
                eta_nat, zeta_nat = a, b
            elif face_name in ('ymin', 'ymax'):
                eta_nat = -1.0 if face_name == 'ymin' else 1.0
                xi_nat, zeta_nat = a, b
            else:  # zmin, zmax
                zeta_nat = -1.0 if face_name == 'zmin' else 1.0
                xi_nat, eta_nat = a, b

            # PMC 过滤 (如果提供了 csg_root)
            if csg_root is not None and npe >= 8:
                if npe <= 8:
                    N_face = np.empty(8)
                    for i in range(8):
                        p = [[-1,-1], [1,-1], [1,1], [-1,1]][i % 4]
                        N_face[i] = 0.25 * (1+p[0]*xi_nat)*(1+p[1]*eta_nat)
                phys_pt = coords[:8].T @ N_face[:8] if npe <= 8 else coords[:8].T @ N_face[:8] if npe <= 8 else np.zeros(3)
                # 使用 hex8 插值
                from .elements import hex8_shape_func
                N8 = hex8_shape_func(xi_nat, eta_nat, zeta_nat)
                phys_pt = N8 @ coords[:8]

                sdf_val = csg_root.sdf_batch(phys_pt.reshape(1, 3))[0]
                if sdf_val > 1e-8:  # void — 跳过
                    continue

            # 计算面元素 Jacobian
            if npe <= 8:
                from .elements import hex8_shape_grad
                dN = hex8_shape_grad(xi_nat, eta_nat, zeta_nat)
                J = dN @ coords[:8]
            elif npe <= 20:
                from .elements import hex20_shape_grad
                dN = hex20_shape_grad(xi_nat, eta_nat, zeta_nat)
                J = dN @ coords
            else:
                from .elements import hex32_shape_grad
                dN = hex32_shape_grad(xi_nat, eta_nat, zeta_nat)
                J = dN @ coords

            # 面 Jacobian (dA)
            if face_name in ('xmin', 'xmax'):
                dS = np.linalg.norm(np.cross(J[:, 1], J[:, 2]))
            elif face_name in ('ymin', 'ymax'):
                dS = np.linalg.norm(np.cross(J[:, 0], J[:, 2]))
            else:
                dS = np.linalg.norm(np.cross(J[:, 0], J[:, 1]))

            # 形函数在面 Gauss 点的值
            if npe <= 8:
                from .elements import hex8_shape_func as sf
                N_val = sf(xi_nat, eta_nat, zeta_nat)
            elif npe <= 20:
                N_val = hex20_shape_func(xi_nat, eta_nat, zeta_nat)
            else:
                from .elements import hex32_shape_func
                N_val = hex32_shape_func(xi_nat, eta_nat, zeta_nat)

            force_contrib = w_gauss * dS * face_normal_sign

            for a in range(npe):
                node = elem_nodes[a]
                F[node*3 + 0] += N_val[a] * tx * force_contrib
                F[node*3 + 1] += N_val[a] * ty * force_contrib
                F[node*3 + 2] += N_val[a] * tz * force_contrib

    return F


def get_face_fixed_dofs(mesh, face_name: str) -> np.ndarray:
    """获取面上的固定 DOF 索引.

    Args:
        mesh: UniformHexMesh
        face_name: 'xmin'/'xmax'/'ymin'/'ymax'/'zmin'/'zmax'

    Returns:
        (M,) DOF 索引数组
    """
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz

    all_nodes = set()
    if face_name == 'xmin':
        for k in range(nz+1):
            for j in range(ny+1):
                n = 0 + j*(nx+1) + k*(nx+1)*(ny+1)
                all_nodes.add(n)
    elif face_name == 'xmax':
        for k in range(nz+1):
            for j in range(ny+1):
                n = nx + j*(nx+1) + k*(nx+1)*(ny+1)
                all_nodes.add(n)
    elif face_name == 'ymin':
        for k in range(nz+1):
            for i in range(nx+1):
                n = i + 0*(nx+1) + k*(nx+1)*(ny+1)
                all_nodes.add(n)
    elif face_name == 'ymax':
        for k in range(nz+1):
            for i in range(nx+1):
                n = i + ny*(nx+1) + k*(nx+1)*(ny+1)
                all_nodes.add(n)
    elif face_name == 'zmin':
        for j in range(ny+1):
            for i in range(nx+1):
                n = i + j*(nx+1) + 0*(nx+1)*(ny+1)
                all_nodes.add(n)
    elif face_name == 'zmax':
        for j in range(ny+1):
            for i in range(nx+1):
                n = i + j*(nx+1) + nz*(nx+1)*(ny+1)
                all_nodes.add(n)
    else:
        raise ValueError(f"Unknown face: {face_name}")

    # 扩展到 mesh 的边中点节点 (Hex20/32)
    # 对于高阶单元, 额外添加边中点
    if mesh.element_order >= 2:
        extra_nodes = set()
        # 简单策略: 检查所有节点坐标是否在面上
        eps = 1e-10
        for nid in range(mesh.n_nodes):
            x, y, z = mesh.nodes[nid]
            if face_name == 'xmin' and abs(x - mesh.ox) < eps:
                extra_nodes.add(nid)
            elif face_name == 'xmax' and abs(x - (mesh.ox + mesh.lx)) < eps:
                extra_nodes.add(nid)
            elif face_name == 'ymin' and abs(y - mesh.oy) < eps:
                extra_nodes.add(nid)
            elif face_name == 'ymax' and abs(y - (mesh.oy + mesh.ly)) < eps:
                extra_nodes.add(nid)
            elif face_name == 'zmin' and abs(z - mesh.oz) < eps:
                extra_nodes.add(nid)
            elif face_name == 'zmax' and abs(z - (mesh.oz + mesh.lz)) < eps:
                extra_nodes.add(nid)
        all_nodes.update(extra_nodes)

    dofs = []
    for nid in sorted(all_nodes):
        dofs.extend([nid*3, nid*3+1, nid*3+2])
    return np.array(dofs, dtype=np.int32)
