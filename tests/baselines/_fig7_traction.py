# -*- coding: utf-8 -*-
"""Fig 7 牵引力等效节点力计算 — 新旧版本共用参考实现.

本模块从 examples/compare_fig7.py 提取, 供回归测试与基准生成脚本共享,
保证两者使用完全一致的手动牵引力计算逻辑 (原则 P1 数值保真).

计算 xmax 面上 Hex8 单元的牵引力等效节点力 (2×2 面 Gauss 积分).
"""
import numpy as np


def compute_traction_force(get_coords, elems, ndof, n_voxels, origin, lx,
                           traction, pmc_fn):
    """手动计算 xmax 面牵引力等效节点力.

    Args:
        get_coords: callable(eid) -> (npe, 3) 单元节点坐标.
        elems: (n_elems, npe) 单元-节点表.
        ndof: 总自由度数.
        n_voxels: 体素总数.
        origin: (ox, oy, oz).
        lx: X 方向长度.
        traction: (tx, ty, tz).
        pmc_fn: callable(x, y, z, eid) -> int status (-1/0/+1).

    Returns:
        np.ndarray: (ndof,) 等效节点力向量.
    """
    F_face = np.zeros(ndof)
    tx, ty, tz = traction
    for eid in range(n_voxels):
        coords = get_coords(eid)
        if abs(coords[1, 0] - (origin[0] + lx)) > 1e-10:
            continue
        elem_nodes = elems[eid]
        for gp_xi in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
            for gp_eta in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
                N4 = np.array([
                    0.25 * (1 - gp_xi) * (1 - gp_eta),
                    0.25 * (1 + gp_xi) * (1 - gp_eta),
                    0.25 * (1 + gp_xi) * (1 + gp_eta),
                    0.25 * (1 - gp_xi) * (1 + gp_eta),
                ])
                face_nodes = [1, 2, 6, 5]
                fc = coords[face_nodes]
                dNdxi = np.array([
                    -0.25 * (1 - gp_eta), 0.25 * (1 - gp_eta),
                    0.25 * (1 + gp_eta), -0.25 * (1 + gp_eta),
                ])
                dNdeta = np.array([
                    -0.25 * (1 - gp_xi), -0.25 * (1 + gp_xi),
                    0.25 * (1 + gp_xi), 0.25 * (1 - gp_xi),
                ])
                J_face = np.array([
                    [dNdxi @ fc[:, 1], dNdeta @ fc[:, 1]],
                    [dNdxi @ fc[:, 2], dNdeta @ fc[:, 2]],
                ])
                dS = abs(np.linalg.det(J_face))
                gp_xyz = N4 @ fc
                status = pmc_fn(gp_xyz[0], gp_xyz[1], gp_xyz[2], eid)
                if status == -1:
                    continue
                for a, ni in enumerate(face_nodes):
                    nid = elem_nodes[ni]
                    force = 1.0 * N4[a] * dS
                    F_face[nid * 3] += force * tx
                    F_face[nid * 3 + 1] += force * ty
                    F_face[nid * 3 + 2] += force * tz
    return F_face
