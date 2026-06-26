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
from scipy.sparse import csr_matrix, coo_matrix
from typing import Optional, Tuple, Callable
from .elements import (get_element_info, elastic_matrix_D,
                        hex8_shape_grad, hex8_shape_func_batch,
                        _build_B_matrix, HEX8_NODES)
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

    优化策略:
    1. 规则网格上所有单元几何相同 → ke 仅计算一次, 批量散列
    2. DOF 索引向量化计算 (broadcasting, 零 Python 循环)
    3. 固体/虚空类批量 scatter; 边界类仍逐单元八叉树积分
    """
    info = get_element_info(order)
    npe = info['npe']
    ndof_per_elem = info['ndof_per_elem']
    ke_func = info['ke_func']
    gauss_rule = info['gauss_rule']
    ndof = mesh.ndof
    n_entries = ndof_per_elem * ndof_per_elem

    solid_eids, void_eids, cut_eids = classify_elements(voxel_nature)
    n_total = len(solid_eids) + len(void_eids) + len(cut_eids)

    # 预分配 (cut 元素可能 ke=None, 略超配, 不影响正确性)
    rows = np.empty(n_total * n_entries, dtype=np.int32)
    cols = np.empty(n_total * n_entries, dtype=np.int32)
    data = np.empty(n_total * n_entries, dtype=np.float64)
    offset = 0

    # 规则网格上任意单元的 ke 均相同 → 取 eid=0 计算一次
    ke_solid = ke_func(mesh.get_elem_coords(0), E, nu)

    # 1. 固体体素批量散列
    if len(solid_eids) > 0:
        dofs_solid = _get_elem_dofs_batch(mesh.elems[solid_eids])
        offset = _scatter_batch(rows, cols, data, offset, dofs_solid, ke_solid)

    # 2. 虚空体素批量散列 (ke ∝ E, 故 ke_void = ke_solid * alpha)
    if len(void_eids) > 0:
        dofs_void = _get_elem_dofs_batch(mesh.elems[void_eids])
        offset = _scatter_batch(rows, cols, data, offset, dofs_void, ke_solid * alpha)

    # 3. 边界体素 → 八叉树自适应积分 (批量 SDF, 预计算 DOF)
    print(f"  Solid: {len(solid_eids)}, Void: {len(void_eids)}, Cut: {len(cut_eids)}")
    if len(cut_eids) > 0:
        dofs_cut = _get_elem_dofs_batch(mesh.elems[cut_eids])
    for idx, eid in enumerate(cut_eids):
        coords = mesh.get_elem_coords(eid)
        ke = assemble_boundary_ke(
            coords, csg_root, E, nu, alpha, order, gauss_rule, max_depth
        )
        if ke is not None:
            offset = _scatter_triplets(rows, cols, data, offset,
                                       dofs_cut[idx], ke, ndof_per_elem)
        if (idx + 1) % 50 == 0:
            print(f"\r  Cut elements: {idx+1}/{len(cut_eids)}...", end='', flush=True)
    if len(cut_eids) > 0:
        print(f"\r  Cut elements: {len(cut_eids)}/{len(cut_eids)} done.   ")

    # 一次性构建 COO → CSR (scipy 自动 sum 重复项)
    K = coo_matrix((data[:offset], (rows[:offset], cols[:offset])),
                   shape=(ndof, ndof), dtype=np.float64).tocsr()
    return K


# ── 八叉树预计算常数 ──────────────────────────────────────────────
# 8 个子域的偏移向量 (避免递归内重复 np.array 构建)
_OCTANT_OFFSETS: np.ndarray = np.array(
    [[i, j, k] for i in range(2) for j in range(2) for k in range(2)],
    dtype=np.float64,
)


def _scatter_batch(
    rows: np.ndarray, cols: np.ndarray, data: np.ndarray,
    offset: int, all_dofs: np.ndarray, ke: np.ndarray,
) -> int:
    """批量散列: 同一 ke 应用于所有单元, 纯向量化无 Python 循环.

    Args:
        all_dofs: (n_elems, ndof_per_elem) 单元 DOF 索引.
        ke: (ndof_per_elem, ndof_per_elem) 共享的单元刚度矩阵.
    Returns:
        更新后的 offset.
    """
    n_elems = all_dofs.shape[0]
    n = all_dofs.shape[1]
    block = n_elems * n * n
    end = offset + block

    # rows: 每个 dof 重复 n 次 → [d0,...,d0, d1,...,d1, ...]
    #   np.repeat((N,24), 24, axis=1) → (N,576) → ravel
    rows[offset:end] = np.repeat(all_dofs, n, axis=1).ravel()

    # cols: 整个 dof 组 tile n 次 → [d0..d23, d0..d23, ...]
    #   np.tile((N,24), (1,24)) → (N,576) → ravel
    cols[offset:end] = np.tile(all_dofs, (1, n)).ravel()

    # data: ke.ravel() 重复 n_elems 次
    data[offset:end] = np.tile(ke.ravel(), n_elems)
    return end


def _scatter_triplets(
    rows: np.ndarray, cols: np.ndarray, data: np.ndarray,
    offset: int, dofs: np.ndarray, ke: np.ndarray, ndof_per_elem: int,
) -> int:
    """单元素散列 (用于边界体素八叉树积分, 每个 ke 不同)."""
    n = ndof_per_elem
    end = offset + n * n
    rows[offset:end] = np.repeat(dofs, n)
    cols[offset:end] = np.tile(dofs, n)
    data[offset:end] = ke.ravel()
    return end


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

    # D 矩阵仅依赖 E,nu → hoist 到递归外 (每层重算浪费)
    D = elastic_matrix_D(E, nu)

    _octree_integrate(
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        coords, csg_root, D, E, nu, alpha, order, gauss_rule,
        ke, has_material, depth=0, max_depth=max_depth,
    )

    return ke if has_material[0] else None


def _octree_integrate(
    lo: np.ndarray, hi: np.ndarray,
    coords: np.ndarray, csg_root,
    D: np.ndarray,
    E: float, nu: float, alpha: float,
    order: int, gauss_rule,
    ke: np.ndarray, has_material: np.ndarray,
    depth: int, max_depth: int,
):
    """八叉树递归积分 (D 矩阵 hoist, SDF 批量, Hex8 批量 Gauss)."""
    info = get_element_info(order)
    npe = info['npe']
    shape_grad_func = info['shape_grad']

    # ── 子域中心 PMC ──
    center = (lo + hi) / 2.0
    phys_center = _ref_to_phys(center, coords, npe)

    if csg_root is not None:
        sdf_val = csg_root.sdf_batch(phys_center.reshape(1, 3))[0]
        status = _classify_single(sdf_val)
    else:
        status = 1

    if depth >= max_depth or status == 1:
        # ── 叶节点: Gauss 积分 ──
        sub_size = (hi - lo) / 2.0
        local_pts = lo + (gauss_rule.points + 1.0) * sub_size  # (N_gauss, 3)
        n_gauss = len(gauss_rule.points)

        # 批量 _ref_to_phys (复用 hex8_shape_func_batch)
        phys_pts = _ref_to_phys_batch(local_pts, coords)

        # 批量 SDF → PMC 判定
        if csg_root is not None:
            gp_statuses = _classify_sdfs(csg_root.sdf_batch(phys_pts))
        else:
            gp_statuses = np.ones(n_gauss, dtype=np.int32)

        # 过滤有效 Gauss 点 (void → 跳过, solid/boundary 均用全 E)
        active = gp_statuses != -1
        n_active = int(np.sum(active))
        if n_active == 0:
            return

        active_idx = np.where(active)[0]
        active_local = local_pts[active_idx]
        active_w = gauss_rule.weights[active_idx] * np.prod(sub_size)

        # Hex8 快速路径: 批量 shape_grad + Jacobian + B + einsum
        if order == 1 and n_active > 0:
            _gauss_hex8_batch(active_local, active_w, coords, D, ke)
        else:
            # 高阶/回退: 逐点循环
            _gauss_pointwise(active_local, active_w, coords, npe,
                             shape_grad_func, D, ke)

        has_material[0] = True
        return

    # ── 细分: 批量映射 8 子域中心 + 批量 SDF ──
    half = (hi - lo) / 2.0
    sub_lo_all = lo + _OCTANT_OFFSETS * half           # (8, 3)
    sub_ctrs = sub_lo_all + half / 2.0
    phys_ctrs = _ref_to_phys_batch(sub_ctrs, coords)

    if csg_root is not None:
        sub_statuses = _classify_sdfs(csg_root.sdf_batch(phys_ctrs))
    else:
        sub_statuses = np.ones(8, dtype=np.int32)

    for i in range(8):
        if sub_statuses[i] == -1:
            continue
        sub_hi = sub_lo_all[i] + half
        _octree_integrate(
            sub_lo_all[i], sub_hi, coords, csg_root,
            D, E, nu, alpha, order, gauss_rule,
            ke, has_material, depth + 1, max_depth,
        )


# ── 向量化 Gauss 积分 helpers ─────────────────────────────────────

def _ref_to_phys_batch(xi_arr: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """批量参考坐标 → 物理坐标: (N, 3) → (N, 3)."""
    # 仅用 coords[:8] (Hex8 映射, 对所有阶次兼容)
    N = hex8_shape_func_batch(xi_arr[:, 0], xi_arr[:, 1], xi_arr[:, 2])
    return N @ coords[:8]


def _hex8_shape_grad_batch(xi_arr: np.ndarray) -> np.ndarray:
    """批量 Hex8 形函数梯度: (N, 3) → (N, 3, 8).

    dims: 0=batch, 1=d/d{x,y,z}, 2=node.

    完全向量化 — broadcasting (N,1) × (8,) 替代逐节点循环.
    """
    xi, eta, zeta = xi_arr[:, 0], xi_arr[:, 1], xi_arr[:, 2]  # (N,)
    xi_i = HEX8_NODES[:, 0]  # (8,)
    et_i = HEX8_NODES[:, 1]  # (8,)
    ze_i = HEX8_NODES[:, 2]  # (8,)

    # 公共子表达式 (N, 8)
    t_xi   = 1.0 + xi_i * xi[:, None]
    t_eta  = 1.0 + et_i * eta[:, None]
    t_zeta = 1.0 + ze_i * zeta[:, None]

    dN = np.empty((xi_arr.shape[0], 3, 8), dtype=np.float64)
    dN[:, 0, :] = 0.125 * xi_i * t_eta * t_zeta
    dN[:, 1, :] = 0.125 * t_xi * et_i * t_zeta
    dN[:, 2, :] = 0.125 * t_xi * t_eta * ze_i
    return dN


def _build_B_matrix_batch(dN_dx: np.ndarray) -> np.ndarray:
    """批量 B 矩阵: (N, 3, npe) → (N, 6, 3*npe).

    完全向量化 — 预计算 index arrays, 单次高级索引赋值.
    """
    N, _, npe = dN_dx.shape
    a = np.arange(npe)

    # 9 个应变分量 × npe 节点 = 9*npe 个位置
    row_idx = np.concatenate([
        np.full(npe, 0),  # ε_xx: ∂N/∂x → u
        np.full(npe, 1),  # ε_yy: ∂N/∂y → v
        np.full(npe, 2),  # ε_zz: ∂N/∂z → w
        np.full(npe, 3), np.full(npe, 3),  # γ_xy: ∂N/∂y→u, ∂N/∂x→v
        np.full(npe, 4), np.full(npe, 4),  # γ_yz: ∂N/∂z→v, ∂N/∂y→w
        np.full(npe, 5), np.full(npe, 5),  # γ_zx: ∂N/∂x→w, ∂N/∂z→u
    ])
    col_idx = np.concatenate([
        3 * a,          # u 位移列
        3 * a + 1,      # v 位移列
        3 * a + 2,      # w 位移列
        3 * a, 3 * a + 1,
        3 * a + 1, 3 * a + 2,
        3 * a, 3 * a + 2,
    ])
    src_dim = np.concatenate([
        np.full(npe, 0),  # dN/dx
        np.full(npe, 1),  # dN/dy
        np.full(npe, 2),  # dN/dz
        np.full(npe, 1), np.full(npe, 0),  # γ_xy
        np.full(npe, 2), np.full(npe, 1),  # γ_yz
        np.full(npe, 2), np.full(npe, 0),  # γ_zx
    ])
    src_node = np.tile(a, 9)

    B = np.zeros((N, 6, 3 * npe), dtype=np.float64)
    B[:, row_idx, col_idx] = dN_dx[:, src_dim, src_node]
    return B


def _gauss_hex8_batch(
    local_pts: np.ndarray, weights: np.ndarray,
    coords: np.ndarray, D: np.ndarray, ke: np.ndarray,
):
    """Hex8 批量 Gauss 积分 — 零 Python 逐点循环."""
    # dN: (M, 3, 8)
    dN = _hex8_shape_grad_batch(local_pts)
    # J: (M, 3, 3)  (broadcast: dN@coords does (3,8)@(8,3) per batch)
    J = dN @ coords
    detJ = np.linalg.det(J)
    valid = detJ > 0
    if not np.any(valid):
        return

    Jv, dNv, wv, djv = J[valid], dN[valid], weights[valid], detJ[valid]
    invJ = np.linalg.inv(Jv)              # (M', 3, 3)
    dN_dx = invJ @ dNv                    # (M', 3, 3) @ (M', 3, 8) = (M', 3, 8)
    B = _build_B_matrix_batch(dN_dx)      # (M', 6, 24)

    # ke += Σ w_g * detJ_g * B_g^T @ D @ B_g
    w_det = wv * djv                      # (M',)
    BT_D_B = np.einsum('gij,jk,gkl->gil', B.transpose(0, 2, 1), D, B)
    ke += np.einsum('g,gij->ij', w_det, BT_D_B)


def _gauss_pointwise(
    local_pts: np.ndarray, weights: np.ndarray,
    coords: np.ndarray, npe: int, shape_grad_func,
    D: np.ndarray,
    ke: np.ndarray,
):
    """逐点 Gauss 积分 (回退路径, 用于 Hex20/Hex32)."""
    for gp in range(len(local_pts)):
        xi, eta, zeta = local_pts[gp]
        dN = shape_grad_func(xi, eta, zeta)
        J = dN @ coords
        detJ = np.linalg.det(J)
        if detJ <= 0:
            continue
        invJ = np.linalg.inv(J)
        dN_dx = invJ @ dN
        B = _build_B_matrix(dN_dx, npe)
        ke += weights[gp] * (B.T @ D @ B) * detJ


def _ref_to_phys(xi_vec: np.ndarray, coords: np.ndarray, npe: int) -> np.ndarray:
    """将参考坐标 (ξ,η,ζ) 映射到物理坐标.

    使用 Hex8 形函数做映射 (适用于任意阶次的规则网格).
    """
    from .elements import hex8_shape_func
    N = hex8_shape_func(xi_vec[0], xi_vec[1], xi_vec[2])
    return (N @ coords[:8]).flatten()


def _classify_single(sdf_val: float) -> int:
    """单点 SDF 分类 — 与旧版 pmc_point_3d 行为完全一致 (无 tolerance)."""
    if sdf_val < 0:
        return 1      # solid
    elif sdf_val == 0:
        return 0      # boundary
    else:
        return -1     # void


def _classify_sdfs(sdf_vals: np.ndarray) -> np.ndarray:
    """向量化 SDF 分类 — 与 _classify_single 语义完全一致.

    sdf < 0 → 1 (solid), sdf == 0 → 0 (boundary), sdf > 0 → -1 (void).
    """
    result = np.ones(len(sdf_vals), dtype=np.int32)
    result[sdf_vals > 0] = -1
    result[sdf_vals == 0] = 0
    return result


def _get_elem_dofs_batch(elem_nodes: np.ndarray) -> np.ndarray:
    """向量化批量 DOF 计算 — 零 Python 循环.

    Args:
        elem_nodes: (n_elems, npe) 单元节点连接.
    Returns:
        (n_elems, 3*npe) DOF 索引, 布局 [ux0,uy0,uz0, ux1,uy1,uz1, ...].
    """
    npe = elem_nodes.shape[1]
    # (n_elems, npe, 1) * 3 + [0,1,2] → (n_elems, npe, 3) → (n_elems, 3*npe)
    return (elem_nodes[:, :, None] * 3 + np.arange(3, dtype=np.int32)).reshape(
        elem_nodes.shape[0], 3 * npe)


