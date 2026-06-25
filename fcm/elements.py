# -*- coding: utf-8 -*-
"""
fcm/elements.py — 统一形函数模块 (Hex8/Hex20/Hex32)

包含: 形函数, 形函数梯度, Gauss 积分规则, 单元刚度矩阵.
消除 fem_base.py 和 fem_xvoxel.py 中的重复定义.
"""
import numpy as np
from dataclasses import dataclass


# ============================================================
# 元素类型枚举
# ============================================================
class ElementType:
    HEX8 = 1
    HEX20 = 2
    HEX32 = 3


# ============================================================
# Gauss 积分规则
# ============================================================
@dataclass
class GaussRule:
    points: np.ndarray    # (n_gauss, 3)
    weights: np.ndarray   # (n_gauss,)


def _make_gauss_rule(n: int) -> GaussRule:
    """构建 n×n×n Gauss 积分规则."""
    if n == 2:
        pts_1d = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        wts_1d = np.array([1.0, 1.0])
    elif n == 3:
        pts_1d = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        wts_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    elif n == 4:
        pts_1d = np.array([-0.8611363115940526, -0.3399810435848563,
                            0.3399810435848563,  0.8611363115940526])
        wts_1d = np.array([0.3478548451374538, 0.6521451548625461,
                            0.6521451548625461, 0.3478548451374538])
    else:
        raise ValueError(f"Unsupported Gauss order: {n}")

    XI, ETA, ZETA = np.meshgrid(pts_1d, pts_1d, pts_1d, indexing='ij')
    points = np.column_stack([XI.ravel(), ETA.ravel(), ZETA.ravel()])
    WXI, WETA, WZETA = np.meshgrid(wts_1d, wts_1d, wts_1d, indexing='ij')
    weights = (WXI * WETA * WZETA).ravel()
    return GaussRule(points=points, weights=weights)


GAUSS_2X2X2 = _make_gauss_rule(2)  # 8 Gauss points
GAUSS_3X3X3 = _make_gauss_rule(3)  # 27 Gauss points
GAUSS_4X4X4 = _make_gauss_rule(4)  # 64 Gauss points


def get_gauss_rule(order: int) -> GaussRule:
    """根据单元阶次返回对应的 Gauss 规则."""
    if order == 3:
        return GAUSS_4X4X4
    elif order == 2:
        return GAUSS_3X3X3
    else:
        return GAUSS_2X2X2


# ============================================================
# 参考单元节点坐标
# ============================================================
HEX8_NODES = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
], dtype=np.float64)

HEX20_NODES = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    [ 0, -1, -1], [ 1,  0, -1], [ 0,  1, -1], [-1,  0, -1],
    [ 0, -1,  1], [ 1,  0,  1], [ 0,  1,  1], [-1,  0,  1],
    [-1, -1,  0], [ 1, -1,  0], [ 1,  1,  0], [-1,  1,  0],
], dtype=np.float64)

HEX32_NODES = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    [-1.0/3, -1, -1], [ 1.0/3, -1, -1],
    [ 1, -1.0/3, -1], [ 1,  1.0/3, -1],
    [-1.0/3,  1, -1], [ 1.0/3,  1, -1],
    [-1, -1.0/3, -1], [-1,  1.0/3, -1],
    [-1.0/3, -1,  1], [ 1.0/3, -1,  1],
    [ 1, -1.0/3,  1], [ 1,  1.0/3,  1],
    [-1.0/3,  1,  1], [ 1.0/3,  1,  1],
    [-1, -1.0/3,  1], [-1,  1.0/3,  1],
    [-1, -1, -1.0/3], [-1, -1,  1.0/3],
    [ 1, -1, -1.0/3], [ 1, -1,  1.0/3],
    [ 1,  1, -1.0/3], [ 1,  1,  1.0/3],
    [-1,  1, -1.0/3], [-1,  1,  1.0/3],
], dtype=np.float64)

# Hex20 faces for traction/Nitsche
HEX20_FACES = {
    'zmin': [0, 1, 2, 3, 8, 9, 10, 11],
    'zmax': [4, 5, 6, 7, 12, 13, 14, 15],
    'ymin': [0, 1, 5, 4, 8, 12, 17, 16],
    'ymax': [2, 3, 7, 6, 10, 15, 19, 18],
    'xmin': [0, 3, 7, 4, 11, 15, 19, 16],
    'xmax': [1, 2, 6, 5, 9, 13, 18, 17],
}


# ============================================================
# Hex8 形函数
# ============================================================
def hex8_shape_func(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Hex8 形函数 (8,)."""
    N = np.empty(8, dtype=np.float64)
    for i in range(8):
        N[i] = 0.125 * (1 + HEX8_NODES[i, 0] * xi) \
                     * (1 + HEX8_NODES[i, 1] * eta) \
                     * (1 + HEX8_NODES[i, 2] * zeta)
    return N


def hex8_shape_grad(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Hex8 形函数梯度 (3, 8)."""
    dN = np.empty((3, 8), dtype=np.float64)
    for i in range(8):
        xi_i, et_i, ze_i = HEX8_NODES[i]
        dN[0, i] = 0.125 * xi_i * (1 + et_i * eta) * (1 + ze_i * zeta)
        dN[1, i] = 0.125 * (1 + xi_i * xi) * et_i * (1 + ze_i * zeta)
        dN[2, i] = 0.125 * (1 + xi_i * xi) * (1 + et_i * eta) * ze_i
    return dN


def hex8_shape_func_batch(xi: np.ndarray, eta: np.ndarray,
                          zeta: np.ndarray) -> np.ndarray:
    """批量 Hex8 形函数. (N,) → (N, 8)."""
    N = np.empty((len(xi), 8), dtype=np.float64)
    for i in range(8):
        N[:, i] = 0.125 * (1 + HEX8_NODES[i, 0] * xi) \
                        * (1 + HEX8_NODES[i, 1] * eta) \
                        * (1 + HEX8_NODES[i, 2] * zeta)
    return N


# ============================================================
# Hex20 形函数
# ============================================================
def hex20_shape_func(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Serendipity Hex20 形函数 (20,)."""
    N = np.empty(20, dtype=np.float64)
    for i in range(8):
        p = HEX20_NODES[i]
        N[i] = 0.125 * (1 + p[0]*xi) * (1 + p[1]*eta) * (1 + p[2]*zeta) \
               * (p[0]*xi + p[1]*eta + p[2]*zeta - 2)
    f14 = 0.25 * (1 - zeta)
    f44 = 0.25 * (1 + zeta)
    N[8]  = f14 * (1 - xi*xi) * (1 - eta)
    N[9]  = f14 * (1 + xi) * (1 - eta*eta)
    N[10] = f14 * (1 - xi*xi) * (1 + eta)
    N[11] = f14 * (1 - xi) * (1 - eta*eta)
    N[12] = f44 * (1 - xi*xi) * (1 - eta)
    N[13] = f44 * (1 + xi) * (1 - eta*eta)
    N[14] = f44 * (1 - xi*xi) * (1 + eta)
    N[15] = f44 * (1 - xi) * (1 - eta*eta)
    fz = 0.25 * (1 - zeta*zeta)
    N[16] = fz * (1 - xi) * (1 - eta)
    N[17] = fz * (1 + xi) * (1 - eta)
    N[18] = fz * (1 + xi) * (1 + eta)
    N[19] = fz * (1 - xi) * (1 + eta)
    return N


def hex20_shape_grad(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Hex20 形函数梯度 (3, 20)."""
    dN = np.empty((3, 20), dtype=np.float64)
    for i in range(8):
        p = HEX20_NODES[i]
        dN[0,i] = 0.125 * p[0] * (1+p[1]*eta) * (1+p[2]*zeta) * (2*p[0]*xi + p[1]*eta + p[2]*zeta - 1)
        dN[1,i] = 0.125 * (1+p[0]*xi) * p[1] * (1+p[2]*zeta) * (p[0]*xi + 2*p[1]*eta + p[2]*zeta - 1)
        dN[2,i] = 0.125 * (1+p[0]*xi) * (1+p[1]*eta) * p[2] * (p[0]*xi + p[1]*eta + 2*p[2]*zeta - 1)
    f14 = 0.25 * (1 - zeta); nf14 = -0.25
    f44 = 0.25 * (1 + zeta); nf44 = 0.25
    fz = 0.25 * (1 - zeta*zeta); nfz = -0.5 * zeta
    dN[0,8]  = f14*(-2*xi)*(1-eta);    dN[1,8]  = f14*(1-xi*xi)*(-1);      dN[2,8]  = nf14*(1-xi*xi)*(1-eta)
    dN[0,9]  = f14*1*(1-eta*eta);      dN[1,9]  = f14*(1+xi)*(-2*eta);     dN[2,9]  = nf14*(1+xi)*(1-eta*eta)
    dN[0,10] = f14*(-2*xi)*(1+eta);    dN[1,10] = f14*(1-xi*xi)*1;         dN[2,10] = nf14*(1-xi*xi)*(1+eta)
    dN[0,11] = f14*(-1)*(1-eta*eta);   dN[1,11] = f14*(1-xi)*(-2*eta);     dN[2,11] = nf14*(1-xi)*(1-eta*eta)
    dN[0,12] = f44*(-2*xi)*(1-eta);    dN[1,12] = f44*(1-xi*xi)*(-1);      dN[2,12] = nf44*(1-xi*xi)*(1-eta)
    dN[0,13] = f44*1*(1-eta*eta);      dN[1,13] = f44*(1+xi)*(-2*eta);     dN[2,13] = nf44*(1+xi)*(1-eta*eta)
    dN[0,14] = f44*(-2*xi)*(1+eta);    dN[1,14] = f44*(1-xi*xi)*1;         dN[2,14] = nf44*(1-xi*xi)*(1+eta)
    dN[0,15] = f44*(-1)*(1-eta*eta);   dN[1,15] = f44*(1-xi)*(-2*eta);     dN[2,15] = nf44*(1-xi)*(1-eta*eta)
    dN[0,16] = fz*(-1)*(1-eta);  dN[1,16] = fz*(1-xi)*(-1);   dN[2,16] = nfz*(1-xi)*(1-eta)
    dN[0,17] = fz*1*(1-eta);     dN[1,17] = fz*(1+xi)*(-1);   dN[2,17] = nfz*(1+xi)*(1-eta)
    dN[0,18] = fz*1*(1+eta);     dN[1,18] = fz*(1+xi)*1;      dN[2,18] = nfz*(1+xi)*(1+eta)
    dN[0,19] = fz*(-1)*(1+eta);  dN[1,19] = fz*(1-xi)*1;      dN[2,19] = nfz*(1-xi)*(1+eta)
    return dN


# ============================================================
# Hex32 形函数 (Cubic Serendipity)
# ============================================================
def hex32_shape_func(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Cubic Serendipity Hex32 形函数 (32,).

    Reference: Zienkiewicz & Taylor, 6th ed., Table 6.3.
    """
    N = np.empty(32, dtype=np.float64)
    r2 = xi*xi + eta*eta + zeta*zeta

    for i in range(32):
        p = HEX32_NODES[i]
        xi0, eta0, zeta0 = p[0], p[1], p[2]

        if abs(xi0) == 1 and abs(eta0) == 1 and abs(zeta0) == 1:
            # Corner node
            N[i] = 1.0/64.0 * (1 + xi0*xi) * (1 + eta0*eta) * (1 + zeta0*zeta) * (9*r2 - 19)
        elif abs(xi0) < 1:
            N[i] = 9.0/64.0 * (1 - xi*xi) * (1 + 9*xi0*xi) * (1 + eta0*eta) * (1 + zeta0*zeta)
        elif abs(eta0) < 1:
            N[i] = 9.0/64.0 * (1 + xi0*xi) * (1 - eta*eta) * (1 + 9*eta0*eta) * (1 + zeta0*zeta)
        else:
            N[i] = 9.0/64.0 * (1 + xi0*xi) * (1 + eta0*eta) * (1 - zeta*zeta) * (1 + 9*zeta0*zeta)

    return N


def hex32_shape_grad(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Hex32 形函数梯度 (3, 32)."""
    dN = np.empty((3, 32), dtype=np.float64)
    r2 = xi*xi + eta*eta + zeta*zeta

    for i in range(32):
        p = HEX32_NODES[i]
        xi0, eta0, zeta0 = p[0], p[1], p[2]

        if abs(xi0) == 1 and abs(eta0) == 1 and abs(zeta0) == 1:
            A, B, C = 1 + xi0*xi, 1 + eta0*eta, 1 + zeta0*zeta
            D = 9*r2 - 19
            dN[0, i] = 1.0/64.0 * (xi0 * B * C * D + A * B * C * 18*xi)
            dN[1, i] = 1.0/64.0 * (eta0 * A * C * D + A * B * C * 18*eta)
            dN[2, i] = 1.0/64.0 * (zeta0 * A * B * D + A * B * C * 18*zeta)

        elif abs(xi0) < 1:
            B, C = 1 + eta0*eta, 1 + zeta0*zeta
            dN[0, i] = 9.0/64.0 * ((-2*xi)*(1+9*xi0*xi) + (1-xi*xi)*9*xi0) * B * C
            dN[1, i] = 9.0/64.0 * (1 - xi*xi) * (1 + 9*xi0*xi) * eta0 * C
            dN[2, i] = 9.0/64.0 * (1 - xi*xi) * (1 + 9*xi0*xi) * B * zeta0

        elif abs(eta0) < 1:
            A, C = 1 + xi0*xi, 1 + zeta0*zeta
            dN[0, i] = 9.0/64.0 * xi0 * (1 - eta*eta) * (1 + 9*eta0*eta) * C
            dN[1, i] = 9.0/64.0 * A * ((-2*eta)*(1+9*eta0*eta) + (1-eta*eta)*9*eta0) * C
            dN[2, i] = 9.0/64.0 * A * (1 - eta*eta) * (1 + 9*eta0*eta) * zeta0

        else:
            A, B = 1 + xi0*xi, 1 + eta0*eta
            dN[0, i] = 9.0/64.0 * xi0 * B * (1 - zeta*zeta) * (1 + 9*zeta0*zeta)
            dN[1, i] = 9.0/64.0 * A * eta0 * (1 - zeta*zeta) * (1 + 9*zeta0*zeta)
            dN[2, i] = 9.0/64.0 * A * B * ((-2*zeta)*(1+9*zeta0*zeta) + (1-zeta*zeta)*9*zeta0)

    return dN


# ============================================================
# 弹性矩阵
# ============================================================
def elastic_matrix_D(E: float, nu: float) -> np.ndarray:
    """各向同性弹性矩阵 D (6×6)."""
    c = E / ((1 + nu) * (1 - 2 * nu))
    D = np.array([
        [1-nu, nu,  nu, 0, 0, 0],
        [nu,  1-nu, nu, 0, 0, 0],
        [nu,  nu, 1-nu, 0, 0, 0],
        [0,   0,   0,   (1-2*nu)/2, 0, 0],
        [0,   0,   0,   0, (1-2*nu)/2, 0],
        [0,   0,   0,   0, 0, (1-2*nu)/2],
    ], dtype=np.float64) * c
    return D


# ============================================================
# 单元刚度矩阵 (标量版本, 兼容旧 API)
# ============================================================
def _build_B_matrix(dN_dx: np.ndarray, npe: int) -> np.ndarray:
    """构建应变-位移 B 矩阵 (6, 3*npe)."""
    B = np.zeros((6, 3 * npe), dtype=np.float64)
    for a in range(npe):
        col = a * 3
        B[0, col]   = dN_dx[0, a]
        B[1, col+1] = dN_dx[1, a]
        B[2, col+2] = dN_dx[2, a]
        B[3, col]   = dN_dx[1, a]; B[3, col+1] = dN_dx[0, a]
        B[4, col+1] = dN_dx[2, a]; B[4, col+2] = dN_dx[1, a]
        B[5, col]   = dN_dx[2, a]; B[5, col+2] = dN_dx[0, a]
    return B


def hex8_element_stiffness(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Hex8 单元刚度矩阵 (24×24)."""
    D = elastic_matrix_D(E, nu)
    gauss = GAUSS_2X2X2
    ke = np.zeros((24, 24), dtype=np.float64)

    for gp in range(len(gauss.points)):
        xi, eta, zeta = gauss.points[gp]
        w = gauss.weights[gp]
        dN = hex8_shape_grad(xi, eta, zeta)
        J = dN @ coords
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Negative or zero Jacobian determinant")
        invJ = np.linalg.inv(J)
        dN_dx = invJ @ dN
        B = _build_B_matrix(dN_dx, 8)
        ke += w * (B.T @ D @ B) * detJ
    return ke


def hex20_element_stiffness(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Hex20 单元刚度矩阵 (60×60)."""
    D = elastic_matrix_D(E, nu)
    gauss = GAUSS_3X3X3
    ke = np.zeros((60, 60), dtype=np.float64)

    for gp in range(len(gauss.points)):
        xi, eta, zeta = gauss.points[gp]
        w = gauss.weights[gp]
        dN = hex20_shape_grad(xi, eta, zeta)
        J = dN @ coords
        detJ = np.linalg.det(J)
        if detJ <= 0:
            continue
        invJ = np.linalg.inv(J)
        dN_dx = invJ @ dN
        B = _build_B_matrix(dN_dx, 20)
        ke += w * (B.T @ D @ B) * detJ
    return ke


def hex32_element_stiffness(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Hex32 单元刚度矩阵 (96×96)."""
    D = elastic_matrix_D(E, nu)
    gauss = GAUSS_4X4X4
    ke = np.zeros((96, 96), dtype=np.float64)

    for gp in range(len(gauss.points)):
        xi, eta, zeta = gauss.points[gp]
        w = gauss.weights[gp]
        dN = hex32_shape_grad(xi, eta, zeta)
        J = dN @ coords
        detJ = np.linalg.det(J)
        if detJ <= 0:
            continue
        invJ = np.linalg.inv(J)
        dN_dx = invJ @ dN
        B = _build_B_matrix(dN_dx, 32)
        ke += w * (B.T @ D @ B) * detJ
    return ke


# ============================================================
# 单元信息工厂
# ============================================================
def get_element_info(order: int) -> dict:
    """根据阶次返回 (npe, ndof_per_elem, shape_func, shape_grad, ke_func, gauss_rule)."""
    if order == 1:
        return {
            'npe': 8, 'ndof_per_elem': 24,
            'shape_func': hex8_shape_func,
            'shape_grad': hex8_shape_grad,
            'ke_func': hex8_element_stiffness,
            'gauss_rule': GAUSS_2X2X2,
            'nodes_ref': HEX8_NODES,
        }
    elif order == 2:
        return {
            'npe': 20, 'ndof_per_elem': 60,
            'shape_func': hex20_shape_func,
            'shape_grad': hex20_shape_grad,
            'ke_func': hex20_element_stiffness,
            'gauss_rule': GAUSS_3X3X3,
            'nodes_ref': HEX20_NODES,
        }
    elif order == 3:
        return {
            'npe': 32, 'ndof_per_elem': 96,
            'shape_func': hex32_shape_func,
            'shape_grad': hex32_shape_grad,
            'ke_func': hex32_element_stiffness,
            'gauss_rule': GAUSS_4X4X4,
            'nodes_ref': HEX32_NODES,
        }
    else:
        raise ValueError(f"Unsupported element order: {order}")
