# -*- coding: utf-8 -*-
"""
fcm — 有限胞元法 (FCM) 求解器层

依赖 xvoxel/ (几何层) + numpy/scipy.

Modules:
    elements : 形函数、Gauss 法则、弹性矩阵、单元刚度
    mesh     : UniformHexMesh 规则六面体网格 → Hex8/20/32
    assembly : 全局刚度装配、八叉树边界积分
    boundary : Dirichlet BC、面牵引力
    solver   : FCMSolver 统一外观
"""
from .elements import (
    ElementType, HEX8_NODES, HEX20_NODES, HEX32_NODES,
    GAUSS_2X2X2, GAUSS_3X3X3, GAUSS_4X4X4, get_gauss_rule,
    elastic_matrix_D, get_element_info,
)
from .mesh import UniformHexMesh
from .assembly import assemble_fcm_k
from .boundary import apply_dirichlet, get_face_fixed_dofs, get_face_traction_nodal_forces
from .solver import FCMSolver

__all__ = [
    # elements
    'ElementType', 'HEX8_NODES', 'HEX20_NODES', 'HEX32_NODES',
    'GAUSS_2X2X2', 'GAUSS_3X3X3', 'GAUSS_4X4X4', 'get_gauss_rule',
    'elastic_matrix_D', 'get_element_info',
    # mesh
    'UniformHexMesh',
    # assembly
    'assemble_fcm_k',
    # boundary
    'apply_dirichlet', 'get_face_fixed_dofs', 'get_face_traction_nodal_forces',
    # solver
    'FCMSolver',
]
