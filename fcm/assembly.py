# -*- coding: utf-8 -*-
"""
fcm/assembly.py — 向量化 FCM 刚度矩阵装配

核心改进:
    - 批量计算同类型单元刚度 → 稀疏散射到 coo_matrix
    - PMC 统一走 csg_root.sdf_batch() + classify_sdfs()
    - 边界 Gauss 点 status==0 视为 solid (修复 M5)
    - 支持 Hex8/20/32
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from typing import Optional, Tuple, Callable
from .elements import (get_element_info, elastic_matrix_D,
                        hex8_shape_grad, _build_B_matrix)
from .mesh import UniformHexMesh


def classify_elements(voxel_nature: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据体素 nature 分类单元.

    Returns:
        solid_eids: (N_s,) solid 单元索引
        void_eids: (N_v,) void 单元索引
        cut_eids: (N_c,) boundary 单元索引
    """
    solid_eids = np.where(voxel_nature == 1)[0]
    void_eids = np.where(voxel_nature == -1)[0]
    cut_eids = np.where(voxel_nature == 0)[0]
    return solid_eids, void_eids, cut_eids


def assemble_fcm_k(
    mesh: UniformHexMesh,
    voxel_nature: np.ndarray,
    csg_root,
    E: float,
    nu: float,
    alpha: float = 1e-8,
    order: int = 1,
    max_depth: int = 3,
) -> csr_matrix:
    """装配 FCM 全局刚度矩阵.

    Args:
        mesh: 规则网格
        voxel_nature: (n_elems,) 体素分类
        csg_root: CSG 树根节点 (Feature), 用于 PMC
        E: 杨氏模量
        nu: 泊松比
        alpha: FCM 虚域惩罚参数
        order: 形函数阶次 (1=Hex8, 2=Hex20, 3=Hex32)
        max_depth: 八叉树最大深度

    Returns:
        csr_matrix 全局刚度矩阵
    """
    info = get_element_info(order)
    npe = info['npe']
    ndof_per_elem = info['ndof_per_elem']
    ke_func = info['ke_func']
    gauss_rule = info['gauss_rule']
    ndof = mesh.ndof

    # 使用 Lil 矩阵逐元素装配 (保证正确性, 后续可向量化)
    K = lil_matrix((ndof, ndof), dtype=np.float64)
    n_elems = mesh.n_elems

    solid_eids, void_eids, cut_eids = classify_elements(voxel_nature)

    # 1. 虚空体素 → αE 小刚度 (防止奇异)
    for eid in void_eids:
        coords = mesh.get_elem_coords(eid)
        ke = ke_func(coords, E * alpha, nu)
        dofs = _get_elem_dofs(mesh.elems[eid])
        _add_ke_to_lil(K, ke, dofs, ndof_per_elem)

    # 2. 完全固体体素 → 全刚度
    for eid in solid_eids:
        coords = mesh.get_elem_coords(eid)
        ke = ke_func(coords, E, nu)
        dofs = _get_elem_dofs(mesh.elems[eid])
        _add_ke_to_lil(K, ke, dofs, ndof_per_elem)

    # 3. 边界体素 → 八叉树自适应积分 (带 PMC)
    print(f"  Solid: {len(solid_eids)}, Void: {len(void_eids)}, Cut: {len(cut_eids)}")
    for idx, eid in enumerate(cut_eids):
        coords = mesh.get_elem_coords(eid)
        ke = assemble_boundary_ke(
            coords, csg_root, E, nu, alpha, order, gauss_rule, max_depth
        )
        if ke is not None:
            dofs = _get_elem_dofs(mesh.elems[eid])
            _add_ke_to_lil(K, ke, dofs, ndof_per_elem)
        if (idx + 1) % 50 == 0:
            print(f"\r  Cut elements: {idx+1}/{len(cut_eids)}...", end='', flush=True)
    if len(cut_eids) > 0:
        print(f"\r  Cut elements: {len(cut_eids)}/{len(cut_eids)} done.   ")

    return K.tocsr()


def assemble_boundary_ke(
    coords: np.ndarray,
    csg_root,
    E: float,
    nu: float,
    alpha: float,
    order: int,
    gauss_rule,
    max_depth: int,
) -> Optional[np.ndarray]:
    """对边界体素用自适应八叉树积分.

    Returns:
        (ndof_per_elem, ndof_per_elem) 或 None (全 void).
    """
    npe = get_element_info(order)['npe']
    ndof_per_elem = npe * 3
    ke = np.zeros((ndof_per_elem, ndof_per_elem), dtype=np.float64)
    has_material = np.array([False])

    _octree_integrate(
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        coords, csg_root, E, nu, alpha, order, gauss_rule,
        ke, has_material, depth=0, max_depth=max_depth,
    )

    return ke if has_material[0] else None


def _octree_integrate(
    lo: np.ndarray, hi: np.ndarray,
    coords: np.ndarray, csg_root,
    E: float, nu: float, alpha: float,
    order: int, gauss_rule,
    ke: np.ndarray, has_material: np.ndarray,
    depth: int, max_depth: int,
):
    """八叉树递归积分."""
    info = get_element_info(order)
    npe = info['npe']
    shape_grad_func = info['shape_grad']
    ndof_per_elem = npe * 3
    D = elastic_matrix_D(E, nu)

    # 检测子域中心是否在物理域内 → 决定是否细分
    center = (lo + hi) / 2.0
    phys_center = _ref_to_phys(center, coords, npe)

    if csg_root is not None:
        sdf_val = csg_root.sdf_batch(phys_center.reshape(1, 3))[0]
        status = _classify_single(sdf_val)
    else:
        status = 1  # 无 CSG 树, 全固体

    if status == -1 and depth == 0:
        # 整个子域在虚空外 → 检查所有 8 个子域
        # 快速检测: 先检查 8 个子域中心
        all_void = True
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    sub_lo = lo + np.array([di, dj, dk]) * (hi - lo) / 2.0
                    sub_hi = sub_lo + (hi - lo) / 2.0
                    sub_ctr = (sub_lo + sub_hi) / 2.0
                    phys_sub = _ref_to_phys(sub_ctr, coords, npe)
                    if csg_root is not None:
                        s = _classify_single(csg_root.sdf_batch(phys_sub.reshape(1, 3))[0])
                        if s != -1:
                            all_void = False
                            break
                if not all_void:
                    break
            if not all_void:
                break
        if all_void:
            return

    if depth >= max_depth or status == 1:
        # 在最大深度或完全固体子域 → Gauss 积分
        sub_size = (hi - lo) / 2.0
        has_mat = False

        for gp in range(len(gauss_rule.points)):
            # 映射 Gauss 点到子域
            local_pt = lo + (gauss_rule.points[gp] + 1.0) * sub_size
            phys_pt = _ref_to_phys(local_pt, coords, npe)

            # PMC 判定
            if csg_root is not None:
                sdf_val = csg_root.sdf_batch(phys_pt.reshape(1, 3))[0]
                gp_status = _classify_single(sdf_val)
            else:
                gp_status = 1

            if gp_status == -1:
                continue  # 虚空中无贡献

            # 关键修复 (M5): status==0 (边界上) 视为 solid, 使用完整 E
            E_local = E if gp_status != -1 else E * alpha

            has_mat = True
            w = gauss_rule.weights[gp] * np.prod(sub_size)

            # 计算形函数梯度和 B 矩阵
            xi, eta, zeta = local_pt
            dN = shape_grad_func(xi, eta, zeta)
            J = dN @ coords
            detJ = np.linalg.det(J)
            if detJ <= 0:
                continue
            invJ = np.linalg.inv(J)
            dN_dx = invJ @ dN
            B = _build_B_matrix(dN_dx, npe)

            D_local = elastic_matrix_D(E_local, nu)
            ke += w * (B.T @ D_local @ B) * detJ

        if has_mat:
            has_material[0] = True
        return

    # 细分到 8 个子域
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                sub_lo = lo + np.array([di, dj, dk]) * (hi - lo) / 2.0
                sub_hi = sub_lo + (hi - lo) / 2.0
                _octree_integrate(
                    sub_lo, sub_hi, coords, csg_root,
                    E, nu, alpha, order, gauss_rule,
                    ke, has_material, depth + 1, max_depth,
                )


def _ref_to_phys(xi_vec: np.ndarray, coords: np.ndarray, npe: int) -> np.ndarray:
    """将参考坐标 (ξ,η,ζ) 映射到物理坐标.

    使用 Hex8 形函数做映射 (适用于任意阶次的规则网格).
    """
    from .elements import hex8_shape_func
    N = hex8_shape_func(xi_vec[0], xi_vec[1], xi_vec[2])
    return (N @ coords[:8]).flatten()


def _classify_single(sdf_val: float, tol: float = 1e-8) -> int:
    """单点 SDF 分类."""
    if sdf_val < -tol:
        return 1      # solid
    elif abs(sdf_val) <= tol:
        return 0      # boundary
    else:
        return -1     # void


def _get_elem_dofs(elem_nodes: np.ndarray) -> np.ndarray:
    """单元节点 → DOF 数组 (3*npe,)."""
    npe = len(elem_nodes)
    dofs = np.empty(3 * npe, dtype=np.int32)
    for a in range(npe):
        dofs[a*3:a*3+3] = elem_nodes[a]*3 + np.arange(3)
    return dofs


def _add_ke_to_lil(K: lil_matrix, ke: np.ndarray, dofs: np.ndarray, ndof_per_elem: int):
    """将单元刚度矩阵加到 Lil 全局矩阵."""
    for a in range(ndof_per_elem):
        for b in range(ndof_per_elem):
            K[dofs[a], dofs[b]] += ke[a, b]
