# -*- coding: utf-8 -*-
"""
boundary.py — 流体边界条件管理器

对齐 fcm/boundary.py 的设计:
    - add_*() 惰性注册 (存储配置)
    - apply() 集中施加 (原地修改 grid 数组)
    - 面命名约定与 fcm/ 一致: 'xmin'/'xmax'/'ymin'/'ymax'/'zmin'/'zmax'

支持的边界类型:
    - 速度入口 (inlet): 固定 (u, v, w)
    - 压力出口 (outlet): 固定 p, 速度 Neumann
    - 自由滑移壁 (slip wall): 法向速度 = 0, 切向 Neumann
    - 无滑移壁 (no-slip wall): u = v = w = 0

全向量化: 所有 BC 施加用 numpy 切片操作, 零 Python for 循环.
"""
import numpy as np
from typing import Dict, Set, Tuple


class FluidBC:
    """流体边界条件管理器 (惰性注册 + 集中施加).

    Attributes:
        grid: StaggeredGrid 实例
        _inlets: {face: (u, v, w)} 速度入口
        _outlets: {face: p_val} 压力出口
        _slip_faces: set of face 自由滑移壁
        _noslip_faces: set of face 无滑移壁
    """

    def __init__(self, grid):
        """初始化 BC 管理器.

        Args:
            grid: StaggeredGrid 实例.
        """
        self.grid = grid
        self._inlets: Dict[str, Tuple[float, float, float]] = {}
        self._outlets: Dict[str, float] = {}
        self._slip_faces: Set[str] = set()
        self._noslip_faces: Set[str] = set()

    # ------------------------------------------------------------------
    # 惰性注册
    # ------------------------------------------------------------------
    def add_inlet(self, face: str, velocity: Tuple[float, float, float]) -> None:
        """注册速度入口 BC.

        Args:
            face: 'xmin'/'xmax'/'ymin'/'ymax'/'zmin'/'zmax'.
            velocity: (u, v, w) 速度向量.
        """
        self._inlets[face] = (float(velocity[0]), float(velocity[1]),
                              float(velocity[2]))

    def add_outlet(self, face: str, pressure: float = 0.0) -> None:
        """注册压力出口 BC.

        Args:
            face: 面名称.
            pressure: 出口压力值.
        """
        self._outlets[face] = float(pressure)

    def add_slip_wall(self, face: str) -> None:
        """注册自由滑移壁面 (法向速度=0, 切向 Neumann)."""
        self._slip_faces.add(face)

    def add_noslip_wall(self, face: str) -> None:
        """注册无滑移壁面 (u=v=w=0)."""
        self._noslip_faces.add(face)

    # ------------------------------------------------------------------
    # 集中施加 (全向量化切片)
    # ------------------------------------------------------------------
    def apply(self, grid) -> None:
        """集中施加所有已注册 BC — 全向量化切片操作.

        Args:
            grid: StaggeredGrid 实例 (原地修改).
        """
        # 速度入口
        for face, (u_val, v_val, w_val) in self._inlets.items():
            self._apply_inlet(grid, face, u_val, v_val, w_val)

        # 压力出口 (速度 Neumann + 压力 Dirichlet)
        for face, p_val in self._outlets.items():
            self._apply_outlet(grid, face, p_val)

        # 自由滑移壁
        for face in self._slip_faces:
            self._apply_slip_wall(grid, face)

        # 无滑移壁
        for face in self._noslip_faces:
            self._apply_noslip_wall(grid, face)

    # ------------------------------------------------------------------
    # 各类 BC 的向量化施加
    # ------------------------------------------------------------------
    def _apply_inlet(self, grid, face: str,
                     u_val: float, v_val: float, w_val: float) -> None:
        """速度入口: 固定速度 + 压力 Neumann."""
        if face == 'xmin':
            grid.u[0, :, :] = u_val
            grid.v[0, :, :] = v_val
            grid.w[0, :, :] = w_val
            # 压力 Neumann: p[0] = p[1]
            grid.p[0, :, :] = grid.p[1, :, :]
        elif face == 'xmax':
            grid.u[-1, :, :] = u_val
            grid.v[-1, :, :] = v_val
            grid.w[-1, :, :] = w_val
            grid.p[-1, :, :] = grid.p[-2, :, :]
        elif face == 'ymin':
            grid.u[:, 0, :] = u_val
            grid.v[:, 0, :] = v_val
            grid.w[:, 0, :] = w_val
            grid.p[:, 0, :] = grid.p[:, 1, :]
        elif face == 'ymax':
            grid.u[:, -1, :] = u_val
            grid.v[:, -1, :] = v_val
            grid.w[:, -1, :] = w_val
            grid.p[:, -1, :] = grid.p[:, -2, :]
        elif face == 'zmin':
            grid.u[:, :, 0] = u_val
            grid.v[:, :, 0] = v_val
            grid.w[:, :, 0] = w_val
            grid.p[:, :, 0] = grid.p[:, :, 1]
        elif face == 'zmax':
            grid.u[:, :, -1] = u_val
            grid.v[:, :, -1] = v_val
            grid.w[:, :, -1] = w_val
            grid.p[:, :, -1] = grid.p[:, :, -2]

    def _apply_outlet(self, grid, face: str, p_val: float) -> None:
        """压力出口: 压力 Dirichlet + 速度 Neumann (零梯度)."""
        if face == 'xmin':
            grid.p[0, :, :] = p_val
            grid.u[0, :, :] = grid.u[1, :, :]
            grid.v[0, :, :] = grid.v[1, :, :]
            grid.w[0, :, :] = grid.w[1, :, :]
        elif face == 'xmax':
            grid.p[-1, :, :] = p_val
            grid.u[-1, :, :] = grid.u[-2, :, :]
            grid.v[-1, :, :] = grid.v[-2, :, :]
            grid.w[-1, :, :] = grid.w[-2, :, :]
        elif face == 'ymin':
            grid.p[:, 0, :] = p_val
            grid.u[:, 0, :] = grid.u[:, 1, :]
            grid.v[:, 0, :] = grid.v[:, 1, :]
            grid.w[:, 0, :] = grid.w[:, 1, :]
        elif face == 'ymax':
            grid.p[:, -1, :] = p_val
            grid.u[:, -1, :] = grid.u[:, -2, :]
            grid.v[:, -1, :] = grid.v[:, -2, :]
            grid.w[:, -1, :] = grid.w[:, -2, :]
        elif face == 'zmin':
            grid.p[:, :, 0] = p_val
            grid.u[:, :, 0] = grid.u[:, :, 1]
            grid.v[:, :, 0] = grid.v[:, :, 1]
            grid.w[:, :, 0] = grid.w[:, :, 1]
        elif face == 'zmax':
            grid.p[:, :, -1] = p_val
            grid.u[:, :, -1] = grid.u[:, :, -2]
            grid.v[:, :, -1] = grid.v[:, :, -2]
            grid.w[:, :, -1] = grid.w[:, :, -2]

    def _apply_slip_wall(self, grid, face: str) -> None:
        """自由滑移壁: 法向速度=0, 切向 Neumann, 压力 Neumann."""
        if face == 'xmin':
            grid.u[0, :, :] = 0.0
            grid.v[0, :, :] = grid.v[1, :, :]
            grid.w[0, :, :] = grid.w[1, :, :]
            grid.p[0, :, :] = grid.p[1, :, :]
        elif face == 'xmax':
            grid.u[-1, :, :] = 0.0
            grid.v[-1, :, :] = grid.v[-2, :, :]
            grid.w[-1, :, :] = grid.w[-2, :, :]
            grid.p[-1, :, :] = grid.p[-2, :, :]
        elif face == 'ymin':
            grid.v[:, 0, :] = 0.0
            grid.u[:, 0, :] = grid.u[:, 1, :]
            grid.w[:, 0, :] = grid.w[:, 1, :]
            grid.p[:, 0, :] = grid.p[:, 1, :]
        elif face == 'ymax':
            grid.v[:, -1, :] = 0.0
            grid.u[:, -1, :] = grid.u[:, -2, :]
            grid.w[:, -1, :] = grid.w[:, -2, :]
            grid.p[:, -1, :] = grid.p[:, -2, :]
        elif face == 'zmin':
            grid.w[:, :, 0] = 0.0
            grid.u[:, :, 0] = grid.u[:, :, 1]
            grid.v[:, :, 0] = grid.v[:, :, 1]
            grid.p[:, :, 0] = grid.p[:, :, 1]
        elif face == 'zmax':
            grid.w[:, :, -1] = 0.0
            grid.u[:, :, -1] = grid.u[:, :, -2]
            grid.v[:, :, -1] = grid.v[:, :, -2]
            grid.p[:, :, -1] = grid.p[:, :, -2]

    def _apply_noslip_wall(self, grid, face: str) -> None:
        """无滑移壁: u=v=w=0, 压力 Neumann."""
        if face == 'xmin':
            grid.u[0, :, :] = 0.0
            grid.v[0, :, :] = 0.0
            grid.w[0, :, :] = 0.0
            grid.p[0, :, :] = grid.p[1, :, :]
        elif face == 'xmax':
            grid.u[-1, :, :] = 0.0
            grid.v[-1, :, :] = 0.0
            grid.w[-1, :, :] = 0.0
            grid.p[-1, :, :] = grid.p[-2, :, :]
        elif face == 'ymin':
            grid.u[:, 0, :] = 0.0
            grid.v[:, 0, :] = 0.0
            grid.w[:, 0, :] = 0.0
            grid.p[:, 0, :] = grid.p[:, 1, :]
        elif face == 'ymax':
            grid.u[:, -1, :] = 0.0
            grid.v[:, -1, :] = 0.0
            grid.w[:, -1, :] = 0.0
            grid.p[:, -1, :] = grid.p[:, -2, :]
        elif face == 'zmin':
            grid.u[:, :, 0] = 0.0
            grid.v[:, :, 0] = 0.0
            grid.w[:, :, 0] = 0.0
            grid.p[:, :, 0] = grid.p[:, :, 1]
        elif face == 'zmax':
            grid.u[:, :, -1] = 0.0
            grid.v[:, :, -1] = 0.0
            grid.w[:, :, -1] = 0.0
            grid.p[:, :, -1] = grid.p[:, :, -2]
