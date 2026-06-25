# -*- coding: utf-8 -*-
"""
fem_base.py — 基础3D八节点六面体(Hex8)有限元求解器
支持规则网格、稀疏装配、边界条件处理
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

# ============================================================
#  Hex8 形函数与数值积分
# ============================================================

# 参考单元坐标 (ξ, η, ζ) ∈ [-1, 1]³ 的8个节点
HEX8_NODES = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
], dtype=np.float64)

# 2×2×2 Gauss 积分点和权重
GAUSS_2 = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
GAUSS_WT = np.array([1.0, 1.0])


def hex8_shape_func( xi, eta, zeta):
    """返回8个形函数在(xi,eta,zeta)处的值"""
    N = np.empty(8, dtype=np.float64)
    for i in range(8):
        N[i] = 0.125 * (1 + HEX8_NODES[i, 0] * xi) \
                     * (1 + HEX8_NODES[i, 1] * eta) \
                     * (1 + HEX8_NODES[i, 2] * zeta)
    return N


def hex8_shape_grad( xi, eta, zeta):
    """返回形函数对自然坐标的导数 dN/dξ, dN/dη, dN/dζ, shape=(3,8)"""
    dN = np.empty((3, 8), dtype=np.float64)
    for i in range(8):
        xi_i, et_i, ze_i = HEX8_NODES[i]
        dN[0, i] = 0.125 * xi_i   * (1 + et_i * eta)  * (1 + ze_i * zeta)
        dN[1, i] = 0.125 * (1 + xi_i * xi)  * et_i        * (1 + ze_i * zeta)
        dN[2, i] = 0.125 * (1 + xi_i * xi)  * (1 + et_i * eta) * ze_i
    return dN


def hex8_element_stiffness( coords, E, nu):
    """
    计算单个 Hex8 单元刚度矩阵 (24×24)
    coords: (8×3) 节点物理坐标
    E: 杨氏模量
    nu: 泊松比
    """
    # 各向同性弹性矩阵 D (6×6)
    c = E / ((1 + nu) * (1 - 2 * nu))
    D = np.array([
        [1-nu, nu,  nu, 0, 0, 0],
        [nu,  1-nu, nu, 0, 0, 0],
        [nu,  nu, 1-nu, 0, 0, 0],
        [0,   0,   0,   (1-2*nu)/2, 0, 0],
        [0,   0,   0,   0, (1-2*nu)/2, 0],
        [0,   0,   0,   0, 0, (1-2*nu)/2],
    ], dtype=np.float64) * c

    ke = np.zeros((24, 24), dtype=np.float64)
    for i, xi in enumerate(GAUSS_2):
        for j, eta in enumerate(GAUSS_2):
            for k, zeta in enumerate(GAUSS_2):
                w = GAUSS_WT[i] * GAUSS_WT[j] * GAUSS_WT[k]
                dN = hex8_shape_grad(xi, eta, zeta)  # (3,8)
                # Jacobian: J = dN/dξ · x_nodes,  J(3×3)
                J = dN @ coords  # (3,8) @ (8,3) → (3,3)
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise ValueError("负或零 Jacobian 行列式")
                invJ = np.linalg.inv(J)
                # dN/dx = invJ · dN/dξ  (3×8)
                dN_dx = invJ @ dN
                # B 矩阵 (6×24)
                B = np.zeros((6, 24), dtype=np.float64)
                for a in range(8):
                    col = a * 3
                    B[0, col]   = dN_dx[0, a]
                    B[1, col+1] = dN_dx[1, a]
                    B[2, col+2] = dN_dx[2, a]
                    B[3, col]   = dN_dx[1, a]
                    B[3, col+1] = dN_dx[0, a]
                    B[4, col+1] = dN_dx[2, a]
                    B[4, col+2] = dN_dx[1, a]
                    B[5, col]   = dN_dx[2, a]
                    B[5, col+2] = dN_dx[0, a]
                ke += w * (B.T @ D @ B) * detJ
    return ke

# ============================================================
#  Hex20 形函数与数值积分 (serendipity 二次单元, 3×3×3 Gauss)
# ============================================================

# 3×3×3 Gauss 积分点和权重
GAUSS_3 = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
GAUSS_3_WT = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

# Hex20 serendipity 参考单元节点 (20节点)
HEX20_NODES = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],  # 0-3: z=-1 面角节点
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],  # 4-7: z=+1 面角节点
    [ 0, -1, -1], [ 1,  0, -1], [ 0,  1, -1], [-1,  0, -1],  # 8-11: z=-1 面边中点
    [ 0, -1,  1], [ 1,  0,  1], [ 0,  1,  1], [-1,  0,  1],  # 12-15: z=+1 面边中点
    [-1, -1,  0], [ 1, -1,  0], [ 1,  1,  0], [-1,  1,  0],  # 16-19: 竖直边中点
], dtype=np.float64)

# Hex20 面定义 (用于面载荷)
HEX20_FACES = {
    'zmin': [0, 1, 2, 3, 8, 9, 10, 11],
    'zmax': [4, 5, 6, 7, 12, 13, 14, 15],
    'ymin': [0, 1, 5, 4, 8, 12, 17, 16],
    'ymax': [2, 3, 7, 6, 10, 15, 19, 18],
    'xmin': [0, 3, 7, 4, 11, 15, 19, 16],
    'xmax': [1, 2, 6, 5, 9, 13, 18, 17],
}


def hex20_shape_func(xi, eta, zeta):
    """Serendipity Hex20 形函数"""
    N = np.empty(20, dtype=np.float64)
    # 角节点: N_i = (1/8)(1+ξξ_i)(1+ηη_i)(1+ζζ_i)(ξξ_i+ηη_i+ζζ_i-2)
    for i in range(8):
        p = HEX20_NODES[i]
        N[i] = 0.125 * (1 + p[0]*xi) * (1 + p[1]*eta) * (1 + p[2]*zeta) \
               * (p[0]*xi + p[1]*eta + p[2]*zeta - 2)
    # 边中点 z=±1
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
    # 竖直边 ζ=0
    fz = 0.25 * (1 - zeta*zeta)
    N[16] = fz * (1 - xi) * (1 - eta)
    N[17] = fz * (1 + xi) * (1 - eta)
    N[18] = fz * (1 + xi) * (1 + eta)
    N[19] = fz * (1 - xi) * (1 + eta)
    return N


def hex20_shape_grad(xi, eta, zeta):
    """Hex20 形函数对自然坐标的导数 (3×20)"""
    dN = np.empty((3, 20), dtype=np.float64)
    # 角节点梯度
    for i in range(8):
        p = HEX20_NODES[i]
        dN[0,i] = 0.125 * p[0] * (1+p[1]*eta) * (1+p[2]*zeta) * (2*p[0]*xi + p[1]*eta + p[2]*zeta - 1)
        dN[1,i] = 0.125 * (1+p[0]*xi) * p[1] * (1+p[2]*zeta) * (p[0]*xi + 2*p[1]*eta + p[2]*zeta - 1)
        dN[2,i] = 0.125 * (1+p[0]*xi) * (1+p[1]*eta) * p[2] * (p[0]*xi + p[1]*eta + 2*p[2]*zeta - 1)
    # 边中点 z=-1 (8-11)
    f14 = 0.25 * (1 - zeta)
    nf14 = -0.25
    dN[0,8]  = f14 * (-2*xi) * (1-eta);   dN[1,8]  = f14 * (1-xi*xi) * (-1);    dN[2,8]  = nf14 * (1-xi*xi) * (1-eta)
    dN[0,9]  = f14 * 1 * (1-eta*eta);     dN[1,9]  = f14 * (1+xi) * (-2*eta);    dN[2,9]  = nf14 * (1+xi) * (1-eta*eta)
    dN[0,10] = f14 * (-2*xi) * (1+eta);   dN[1,10] = f14 * (1-xi*xi) * 1;        dN[2,10] = nf14 * (1-xi*xi) * (1+eta)
    dN[0,11] = f14 * (-1) * (1-eta*eta);  dN[1,11] = f14 * (1-xi) * (-2*eta);    dN[2,11] = nf14 * (1-xi) * (1-eta*eta)
    # 边中点 z=+1 (12-15)
    f44 = 0.25 * (1 + zeta)
    nf44 = 0.25
    dN[0,12] = f44 * (-2*xi) * (1-eta);   dN[1,12] = f44 * (1-xi*xi) * (-1);     dN[2,12] = nf44 * (1-xi*xi) * (1-eta)
    dN[0,13] = f44 * 1 * (1-eta*eta);     dN[1,13] = f44 * (1+xi) * (-2*eta);     dN[2,13] = nf44 * (1+xi) * (1-eta*eta)
    dN[0,14] = f44 * (-2*xi) * (1+eta);   dN[1,14] = f44 * (1-xi*xi) * 1;         dN[2,14] = nf44 * (1-xi*xi) * (1+eta)
    dN[0,15] = f44 * (-1) * (1-eta*eta);  dN[1,15] = f44 * (1-xi) * (-2*eta);     dN[2,15] = nf44 * (1-xi) * (1-eta*eta)
    # 竖直边 ζ=0 (16-19)
    fz = 0.25 * (1 - zeta*zeta)
    nfz = -0.5 * zeta
    dN[0,16] = fz * (-1) * (1-eta);  dN[1,16] = fz * (1-xi) * (-1);   dN[2,16] = nfz * (1-xi) * (1-eta)
    dN[0,17] = fz * 1 * (1-eta);     dN[1,17] = fz * (1+xi) * (-1);   dN[2,17] = nfz * (1+xi) * (1-eta)
    dN[0,18] = fz * 1 * (1+eta);     dN[1,18] = fz * (1+xi) * 1;      dN[2,18] = nfz * (1+xi) * (1+eta)
    dN[0,19] = fz * (-1) * (1+eta);  dN[1,19] = fz * (1-xi) * 1;      dN[2,19] = nfz * (1-xi) * (1+eta)
    return dN


def hex20_element_stiffness(coords, E, nu):
    """
    Hex20 单元刚度矩阵 (60×60), 使用 3×3×3 Gauss 积分
    coords: (20×3) 节点物理坐标
    """
    c = E / ((1+nu) * (1-2*nu))
    D = np.array([
        [1-nu, nu,  nu, 0, 0, 0],
        [nu,  1-nu, nu, 0, 0, 0],
        [nu,  nu, 1-nu, 0, 0, 0],
        [0,   0,   0,   (1-2*nu)/2, 0, 0],
        [0,   0,   0,   0, (1-2*nu)/2, 0],
        [0,   0,   0,   0, 0, (1-2*nu)/2],
    ], dtype=np.float64) * c

    ke = np.zeros((60, 60), dtype=np.float64)
    for i, xi in enumerate(GAUSS_3):
        for j, eta in enumerate(GAUSS_3):
            for k, zeta in enumerate(GAUSS_3):
                w = GAUSS_3_WT[i] * GAUSS_3_WT[j] * GAUSS_3_WT[k]
                dN = hex20_shape_grad(xi, eta, zeta)  # (3, 20)
                J = dN @ coords  # (3,20) @ (20,3) → (3,3)
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    continue
                invJ = np.linalg.inv(J)
                dN_dx = invJ @ dN  # (3, 20)
                B = np.zeros((6, 60), dtype=np.float64)
                for a in range(20):
                    col = a * 3
                    B[0, col]   = dN_dx[0, a]
                    B[1, col+1] = dN_dx[1, a]
                    B[2, col+2] = dN_dx[2, a]
                    B[3, col]   = dN_dx[1, a]
                    B[3, col+1] = dN_dx[0, a]
                    B[4, col+1] = dN_dx[2, a]
                    B[4, col+2] = dN_dx[1, a]
                    B[5, col]   = dN_dx[2, a]
                    B[5, col+2] = dN_dx[0, a]
                ke += w * (B.T @ D @ B) * detJ
    return ke


# ============================================================
#  Hex32 形函数与数值积分 (serendipity 三次单元, 4×4×4 Gauss)
# ============================================================

# 4×4×4 Gauss 积分点和权重
_GP4 = np.array([-0.8611363115940526, -0.3399810435848563,
                  0.3399810435848563,  0.8611363115940526])
_GP4_WT = np.array([0.3478548451374538, 0.6521451548625461,
                     0.6521451548625461, 0.3478548451374538])

# Hex32 参考单元节点 (cubic Serendipity, 32 nodes)
# 角节点 (0-7) + ξ-edge interior nodes (8-23) + ζ-edge interior nodes (24-31)
HEX32_NODES = np.array([
    # 角节点 (0-7)
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    # ξ-edge interior nodes on z=-1 face (8-15), 4 edges × 2 nodes
    [-1.0/3, -1, -1], [ 1.0/3, -1, -1],  # edge 0→1
    [ 1, -1.0/3, -1], [ 1,  1.0/3, -1],  # edge 1→2
    [-1.0/3,  1, -1], [ 1.0/3,  1, -1],  # edge 2→3
    [-1, -1.0/3, -1], [-1,  1.0/3, -1],  # edge 3→0
    # ξ/η-edge interior nodes on z=+1 face (16-23), 4 edges × 2 nodes
    [-1.0/3, -1,  1], [ 1.0/3, -1,  1],  # edge 4→5
    [ 1, -1.0/3,  1], [ 1,  1.0/3,  1],  # edge 5→6
    [-1.0/3,  1,  1], [ 1.0/3,  1,  1],  # edge 6→7
    [-1, -1.0/3,  1], [-1,  1.0/3,  1],  # edge 7→4
    # ζ-edge interior nodes (24-31), 4 vertical edges × 2 nodes
    [-1, -1, -1.0/3], [-1, -1,  1.0/3],  # edge 0→4
    [ 1, -1, -1.0/3], [ 1, -1,  1.0/3],  # edge 1→5
    [ 1,  1, -1.0/3], [ 1,  1,  1.0/3],  # edge 2→6
    [-1,  1, -1.0/3], [-1,  1,  1.0/3],  # edge 3→7
], dtype=np.float64)


def _cubic_lagrange_1d(xi):
    """1D cubic Lagrange basis at nodes [-1, -1/3, 1/3, 1], returns (4,)
    
    Kept for backward compatibility with face quadrature code.
    """
    L = np.empty(4, dtype=np.float64)
    L[0] = -(9.0/16) * (xi + 1.0/3) * (xi - 1.0/3) * (xi - 1)
    L[1] =  (27.0/16) * (xi + 1) * (xi - 1.0/3) * (xi - 1)
    L[2] = -(27.0/16) * (xi + 1) * (xi + 1.0/3) * (xi - 1)
    L[3] =  (9.0/16) * (xi + 1) * (xi + 1.0/3) * (xi - 1.0/3)
    return L


def _cubic_lagrange_grad_1d(xi):
    """Derivative of 1D cubic Lagrange basis, returns (4,)"""
    dL = np.empty(4, dtype=np.float64)
    dL[0] = -(9.0/16) * ((xi-1.0/3)*(xi-1) + (xi+1.0/3)*(xi-1) + (xi+1.0/3)*(xi-1.0/3))
    dL[1] =  (27.0/16) * ((xi-1.0/3)*(xi-1) + (xi+1)*(xi-1) + (xi+1)*(xi-1.0/3))
    dL[2] = -(27.0/16) * ((xi+1.0/3)*(xi-1) + (xi+1)*(xi-1) + (xi+1)*(xi+1.0/3))
    dL[3] =  (9.0/16) * ((xi+1.0/3)*(xi-1.0/3) + (xi+1)*(xi-1.0/3) + (xi+1)*(xi+1.0/3))
    return dL


def _node_to_idx(val):
    """Map node coordinate to 1D Lagrange basis index: -1→0, -1/3→1, 1/3→2, 1→3
    
    Kept for backward compatibility.
    """
    if abs(val - (-1)) < 1e-12:
        return 0
    elif abs(val - (-1.0/3)) < 1e-9:
        return 1
    elif abs(val - (1.0/3)) < 1e-9:
        return 2
    elif abs(val - 1) < 1e-12:
        return 3
    raise ValueError(f"Invalid node coordinate: {val}")


def hex32_shape_func(xi, eta, zeta):
    """Cubic Serendipity Hex32 shape functions (32,)
    
    Correct serendipity formulas (not tensor product of 1D Lagrange).
    Reference: Zienkiewicz & Taylor, 6th ed., Table 6.3.
    
    Corner nodes: N = 1/64 (1+ξ₀ξ)(1+η₀η)(1+ζ₀ζ)[9(ξ²+η²+ζ²)-19]
    ξ-edge nodes: N = 9/64 (1-ξ²)(1+9ξ₀ξ)(1+η₀η)(1+ζ₀ζ)
    η-edge nodes: N = 9/64 (1+ξ₀ξ)(1-η²)(1+9η₀η)(1+ζ₀ζ)
    ζ-edge nodes: N = 9/64 (1+ξ₀ξ)(1+η₀η)(1-ζ²)(1+9ζ₀ζ)
    """
    N = np.empty(32, dtype=np.float64)
    r2 = xi*xi + eta*eta + zeta*zeta

    for i in range(32):
        p = HEX32_NODES[i]
        xi0, eta0, zeta0 = p[0], p[1], p[2]

        # Determine node type: corner, xi-edge, eta-edge, or zeta-edge
        if abs(xi0) == 1 and abs(eta0) == 1 and abs(zeta0) == 1:
            # Corner node
            N[i] = 1.0/64.0 * (1 + xi0*xi) * (1 + eta0*eta) * (1 + zeta0*zeta) * (9*r2 - 19)
        elif abs(xi0) < 1:  # xi-edge node (|xi0|=1/3)
            N[i] = 9.0/64.0 * (1 - xi*xi) * (1 + 9*xi0*xi) * (1 + eta0*eta) * (1 + zeta0*zeta)
        elif abs(eta0) < 1:  # eta-edge node (|eta0|=1/3)
            N[i] = 9.0/64.0 * (1 + xi0*xi) * (1 - eta*eta) * (1 + 9*eta0*eta) * (1 + zeta0*zeta)
        else:  # zeta-edge node (|zeta0|=1/3)
            N[i] = 9.0/64.0 * (1 + xi0*xi) * (1 + eta0*eta) * (1 - zeta*zeta) * (1 + 9*zeta0*zeta)

    return N


def hex32_shape_grad(xi, eta, zeta):
    """Hex32 shape function derivatives w.r.t. natural coordinates (3,32)
    
    See hex32_shape_func for the node formulas.
    """
    dN = np.empty((3, 32), dtype=np.float64)
    r2 = xi*xi + eta*eta + zeta*zeta

    for i in range(32):
        p = HEX32_NODES[i]
        xi0, eta0, zeta0 = p[0], p[1], p[2]

        if abs(xi0) == 1 and abs(eta0) == 1 and abs(zeta0) == 1:
            # Corner node:
            # N = (1/64) * A * B * C * D  where
            # A=1+xi0*xi, B=1+eta0*eta, C=1+zeta0*zeta, D=9*r²-19
            A, B, C = 1 + xi0*xi, 1 + eta0*eta, 1 + zeta0*zeta
            D = 9*r2 - 19
            dN[0, i] = 1.0/64.0 * (xi0 * B * C * D + A * B * C * 18*xi)
            dN[1, i] = 1.0/64.0 * (eta0 * A * C * D + A * B * C * 18*eta)
            dN[2, i] = 1.0/64.0 * (zeta0 * A * B * D + A * B * C * 18*zeta)

        elif abs(xi0) < 1:  # xi-edge: N = 9/64 * (1-ξ²) * (1+9ξ₀ξ) * B * C
            B, C = 1 + eta0*eta, 1 + zeta0*zeta
            dN[0, i] = 9.0/64.0 * ((-2*xi)*(1+9*xi0*xi) + (1-xi*xi)*9*xi0) * B * C
            dN[1, i] = 9.0/64.0 * (1 - xi*xi) * (1 + 9*xi0*xi) * eta0 * C
            dN[2, i] = 9.0/64.0 * (1 - xi*xi) * (1 + 9*xi0*xi) * B * zeta0

        elif abs(eta0) < 1:  # eta-edge: N = 9/64 * A * (1-η²) * (1+9η₀η) * C
            A, C = 1 + xi0*xi, 1 + zeta0*zeta
            dN[0, i] = 9.0/64.0 * xi0 * (1 - eta*eta) * (1 + 9*eta0*eta) * C
            dN[1, i] = 9.0/64.0 * A * ((-2*eta)*(1+9*eta0*eta) + (1-eta*eta)*9*eta0) * C
            dN[2, i] = 9.0/64.0 * A * (1 - eta*eta) * (1 + 9*eta0*eta) * zeta0

        else:  # zeta-edge: N = 9/64 * A * B * (1-ζ²) * (1+9ζ₀ζ)
            A, B = 1 + xi0*xi, 1 + eta0*eta
            dN[0, i] = 9.0/64.0 * xi0 * B * (1 - zeta*zeta) * (1 + 9*zeta0*zeta)
            dN[1, i] = 9.0/64.0 * A * eta0 * (1 - zeta*zeta) * (1 + 9*zeta0*zeta)
            dN[2, i] = 9.0/64.0 * A * B * ((-2*zeta)*(1+9*zeta0*zeta) + (1-zeta*zeta)*9*zeta0)

    return dN


def hex32_face_shape_12(face_type, xi, eta):
    """12-node cubic Serendipity face shape functions and gradients.
    
    Used for face integration (Nitsche, traction).
    
    Returns: N_face (12,), dN_dxi (12,), dN_deta (12,)
    
    Formulas for 12-node quadrilateral:
    Corners: N = 1/32 (1+ξ₀ξ)(1+η₀η)[9(ξ²+η²)-10]
    ξ-edges (ξ₀=±1/3, η₀=±1): N = 9/32 (1-ξ²)(1+9ξ₀ξ)(1+η₀η)
    η-edges (ξ₀=±1, η₀=±1/3): N = 9/32 (1+ξ₀ξ)(1-η²)(1+9η₀η)
    """
    # 12 face nodes in standard order matching HEX32_NODES face ordering
    # The face node local coordinates depend on which face we're on.
    # We define them with 2D local coords (a, b) where a,b ∈ [-1,1]:
    #
    #    3----12----13----2
    #    |                |
    #    11               10
    #    |                |
    #    15               9
    #    |                |
    #    0----8-----14----1
    #
    # 4 corners (0,1,2,3): (-1,-1), (1,-1), (1,1), (-1,1)
    # 8 edge nodes (8-15):
    #   e8(-1/3,-1), e9(1/3,-1), e10(1,-1/3), e11(1,1/3)
    #   e12(-1/3,1), e13(1/3,1), e14(-1,-1/3), e15(-1,1/3)
    
    # All faces share this 2D pattern; the mapping to 3D is handled by face_coords
    face_nodes_2d = np.array([
        [-1, -1], [ 1, -1], [ 1,  1], [-1,  1],  # corners 0-3
        [-1.0/3, -1], [ 1.0/3, -1],  # 4-5: bottom edge
        [ 1, -1.0/3], [ 1,  1.0/3],  # 6-7: right edge
        [-1.0/3,  1], [ 1.0/3,  1],  # 8-9: top edge
        [-1, -1.0/3], [-1,  1.0/3],  # 10-11: left edge
    ], dtype=np.float64)
    
    a0 = face_nodes_2d[:, 0]  # (12,)
    b0 = face_nodes_2d[:, 1]  # (12,)
    
    N = np.empty(12, dtype=np.float64)
    dN_dxi = np.empty(12, dtype=np.float64)
    dN_deta = np.empty(12, dtype=np.float64)
    
    r2 = xi*xi + eta*eta
    
    for i in range(12):
        ai, bi = a0[i], b0[i]
        
        if abs(ai) == 1 and abs(bi) == 1:
            # Corner
            N[i] = 1.0/32.0 * (1 + ai*xi) * (1 + bi*eta) * (9*r2 - 10)
            dN_dxi[i] = 1.0/32.0 * ai * (1 + bi*eta) * (9*r2 - 10) + 1.0/32.0 * (1 + ai*xi) * (1 + bi*eta) * 18*xi
            dN_deta[i] = 1.0/32.0 * bi * (1 + ai*xi) * (9*r2 - 10) + 1.0/32.0 * (1 + ai*xi) * (1 + bi*eta) * 18*eta
        elif abs(ai) < 1:  # xi-edge (bottom/top): ai=±1/3, bi=±1
            N[i] = 9.0/32.0 * (1 - xi*xi) * (1 + 9*ai*xi) * (1 + bi*eta)
            dN_dxi[i] = 9.0/32.0 * ((-2*xi)*(1+9*ai*xi) + (1-xi*xi)*9*ai) * (1 + bi*eta)
            dN_deta[i] = 9.0/32.0 * (1 - xi*xi) * (1 + 9*ai*xi) * bi
        else:  # eta-edge (left/right): ai=±1, bi=±1/3
            N[i] = 9.0/32.0 * (1 + ai*xi) * (1 - eta*eta) * (1 + 9*bi*eta)
            dN_dxi[i] = 9.0/32.0 * ai * (1 - eta*eta) * (1 + 9*bi*eta)
            dN_deta[i] = 9.0/32.0 * (1 + ai*xi) * ((-2*eta)*(1+9*bi*eta) + (1-eta*eta)*9*bi)
    
    return N, dN_dxi, dN_deta


def hex32_element_stiffness(coords, E, nu):
    """Hex32 单元刚度矩阵 (96×96), 4×4×4 Gauss 积分"""
    c = E / ((1+nu) * (1-2*nu))
    D = np.array([
        [1-nu, nu,  nu, 0, 0, 0],
        [nu,  1-nu, nu, 0, 0, 0],
        [nu,  nu, 1-nu, 0, 0, 0],
        [0,   0,   0,   (1-2*nu)/2, 0, 0],
        [0,   0,   0,   0, (1-2*nu)/2, 0],
        [0,   0,   0,   0, 0, (1-2*nu)/2],
    ], dtype=np.float64) * c

    ke = np.zeros((96, 96), dtype=np.float64)
    for i, xi in enumerate(_GP4):
        for j, eta in enumerate(_GP4):
            for k, zeta in enumerate(_GP4):
                w = _GP4_WT[i] * _GP4_WT[j] * _GP4_WT[k]
                dN = hex32_shape_grad(xi, eta, zeta)  # (3,32)
                J = dN @ coords  # (3,32) @ (32,3) → (3,3)
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    continue
                invJ = np.linalg.inv(J)
                dN_dx = invJ @ dN
                B = np.zeros((6, 96))
                for a in range(32):
                    col = a * 3
                    B[0, col]   = dN_dx[0, a]
                    B[1, col+1] = dN_dx[1, a]
                    B[2, col+2] = dN_dx[2, a]
                    B[3, col]   = dN_dx[1, a]
                    B[3, col+1] = dN_dx[0, a]
                    B[4, col+1] = dN_dx[2, a]
                    B[4, col+2] = dN_dx[1, a]
                    B[5, col]   = dN_dx[2, a]
                    B[5, col+2] = dN_dx[0, a]
                ke += w * (B.T @ D @ B) * detJ
    return ke


# ============================================================
#  网格与装配
# ============================================================

class HexMesh:
    """规则六面体网格"""
    def __init__(self, nx, ny, nz, lx, ly, lz):
        """
        nx, ny, nz: 每个方向上的单元数
        lx, ly, lz: 几何尺寸
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.lx, self.ly, self.lz = lx, ly, lz
        self.n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        self.n_elems = nx * ny * nz
        self._build()

    def _build(self):
        # 节点坐标
        x = np.linspace(0, self.lx, self.nx + 1)
        y = np.linspace(0, self.ly, self.ny + 1)
        z = np.linspace(0, self.lz, self.nz + 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        # 使用 Fortran 顺序: i 变化最快, 与索引公式一致
        self.nodes = np.column_stack([X.ravel('F'), Y.ravel('F'), Z.ravel('F')])

        # 单元连接: 每个单元8个节点索引
        self.elems = np.zeros((self.n_elems, 8), dtype=np.int32)
        eid = 0
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    # 8个节点: 前方4个, 后方4个
                    n0 = i + j * (self.nx+1) + k * (self.nx+1)*(self.ny+1)
                    self.elems[eid] = [
                        n0, n0+1, n0+1+(self.nx+1), n0+(self.nx+1),
                        n0 + (self.nx+1)*(self.ny+1),
                        n0+1 + (self.nx+1)*(self.ny+1),
                        n0+1+(self.nx+1) + (self.nx+1)*(self.ny+1),
                        n0+(self.nx+1) + (self.nx+1)*(self.ny+1),
                    ]
                    eid += 1

    def elem_center(self, eid):
        """返回单元 eid 的中心坐标"""
        nodes_xyz = self.nodes[self.elems[eid]]
        return nodes_xyz.mean(axis=0)


class FEMSolver:
    """3D 线性弹性 FEM 求解器"""
    def __init__(self, mesh, E=2e11, nu=0.3):
        self.mesh = mesh
        self.E = E
        self.nu = nu
        self.ndof = mesh.n_nodes * 3

    def assemble_stiffness(self, elem_mask=None, elem_E=None, cache=None):
        """
        装配全局刚度矩阵
        elem_mask: 布尔数组 (n_elems,), True=计算该单元
        elem_E: 每个单元的杨氏模量（可用于 FCM 的 α 方法）
        cache: 预计算的单元刚度矩阵字典 {eid: ke}
        """
        K = lil_matrix((self.ndof, self.ndof), dtype=np.float64)

        if elem_mask is None:
            elem_mask = np.ones(self.mesh.n_elems, dtype=bool)

        if elem_E is not None:
            assert len(elem_E) == self.mesh.n_elems

        for eid in np.where(elem_mask)[0]:
            nodes = self.mesh.elems[eid]
            coords = self.mesh.nodes[nodes]
            E_local = self.E if elem_E is None else elem_E[eid]
            if E_local < 1e-12:
                continue
            nu_local = self.nu
            ke = hex8_element_stiffness(coords, E_local, nu_local)
            dofs = np.array([n*3 + d for n in nodes for d in range(3)])
            for a in range(24):
                for b in range(24):
                    K[dofs[a], dofs[b]] += ke[a, b]
        return K.tocsr()

    def apply_dirichlet(self, K, F, fixed_dofs, fixed_vals=None):
        """
        Dirichlet 边界条件：强形式
        fixed_dofs: 自由度索引数组
        fixed_vals: 固定值数组（默认0）
        """
        if fixed_vals is None:
            fixed_vals = np.zeros(len(fixed_dofs))
        K = K.tolil()
        for i, dof in enumerate(fixed_dofs):
            K[dof, :] = 0
            K[:, dof] = 0
            K[dof, dof] = 1.0
            F[dof] = fixed_vals[i]
        return K.tocsr(), F

    def solve(self, K, F):
        """求解 Ku = F"""
        return spsolve(K, F).astype(np.float64)

    def compute_stress(self, u):
        """在单元高斯点上计算 von Mises 应力"""
        # 对于每个单元，计算所有节点的应力
        von_mises = np.zeros(self.mesh.n_elems, dtype=np.float64)
        c = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        D = np.array([
            [1-self.nu, self.nu,  self.nu, 0, 0, 0],
            [self.nu,  1-self.nu, self.nu, 0, 0, 0],
            [self.nu,  self.nu, 1-self.nu, 0, 0, 0],
            [0,   0,   0,   (1-2*self.nu)/2, 0, 0],
            [0,   0,   0,   0, (1-2*self.nu)/2, 0],
            [0,   0,   0,   0, 0, (1-2*self.nu)/2],
        ], dtype=np.float64) * c

        for eid in range(self.mesh.n_elems):
            nodes = self.mesh.elems[eid]
            dofs = np.array([n*3 + d for n in nodes for d in range(3)])
            u_e = u[dofs]
            sigma_gp = np.zeros(6)
            n_gp = 0
            for xi in GAUSS_2:
                for eta in GAUSS_2:
                    for zeta in GAUSS_2:
                        dN = hex8_shape_grad(xi, eta, zeta)
                        coords = self.mesh.nodes[nodes]
                        J = dN @ coords
                        invJ = np.linalg.inv(J)
                        dN_dx = invJ @ dN
                        B = np.zeros((6, 24))
                        for a in range(8):
                            col = a * 3
                            B[0, col]   = dN_dx[0, a]
                            B[1, col+1] = dN_dx[1, a]
                            B[2, col+2] = dN_dx[2, a]
                            B[3, col]   = dN_dx[1, a]
                            B[3, col+1] = dN_dx[0, a]
                            B[4, col+1] = dN_dx[2, a]
                            B[4, col+2] = dN_dx[1, a]
                            B[5, col]   = dN_dx[2, a]
                            B[5, col+2] = dN_dx[0, a]
                        epsilon = B @ u_e
                        sigma_gp += D @ epsilon
                        n_gp += 1
            sigma_avg = sigma_gp / n_gp
            sxx, syy, szz, sxy, syz, sxz = sigma_avg
            # von Mises
            svm = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2
                               + 6*(sxy**2 + syz**2 + sxz**2)))
            von_mises[eid] = svm
        return von_mises

    def compute_element_displacement_norm(self, u):
        """计算每个单元中心的位移范数"""
        disp_norm = np.zeros(self.mesh.n_elems, dtype=np.float64)
        for eid in range(self.mesh.n_elems):
            nodes = self.mesh.elems[eid]
            dofs = np.array([n*3 + d for n in nodes for d in range(3)])
            u_e = u[dofs].reshape(-1, 3)
            center_u = u_e.mean(axis=0)
            disp_norm[eid] = np.linalg.norm(center_u)
        return disp_norm