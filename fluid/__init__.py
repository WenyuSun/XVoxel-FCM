# -*- coding: utf-8 -*-
"""
fluid/ — XVoxel-FVM-IBM 流体求解器层

有限体积法 (FVM) + 交错网格 (MAC) + 浸入边界法 (IBM) + SIMPLE 算法.
对 xvoxel/ 只读消费, 与 fcm/ 平行.

公开 API:
    StaggeredGrid  — MAC 交错网格数据容器
    IBMForce       — 全向量化浸入边界力
    FluidBC        — 流体边界条件管理器
    MomentumSolver — 动量方程求解器
    PressureSolver — 压力 Poisson 求解器
    SIMPLESolver   — SIMPLE 主循环
    FluidSolver    — 流体求解器外观类 (对齐 FCMSolver)
"""
from .staggered_grid import StaggeredGrid
from .ibm import IBMForce
from .sharp_ibm import SharpIBMHandler
from .boundary import FluidBC
from .momentum import MomentumSolver
from .pressure import PressureSolver
from .simple_solver import SIMPLESolver
from .solver import FluidSolver

__all__ = [
    'StaggeredGrid',
    'IBMForce',
    'SharpIBMHandler',
    'FluidBC',
    'MomentumSolver',
    'PressureSolver',
    'SIMPLESolver',
    'FluidSolver',
]
