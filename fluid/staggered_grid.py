# -*- coding: utf-8 -*-
"""
staggered_grid.py — MAC 交错网格数据容器

维度约定 (MAC 网格):
    u: (nx+1, ny,   nz)   — x-面心速度
    v: (nx,   ny+1, nz)   — y-面心速度
    w: (nx,   ny,   nz+1) — z-面心速度
    p: (nx,   ny,   nz)   — 体素中心压力

设计对齐 fcm/mesh.py 的 UniformHexMesh.from_xvoxel 工厂模式.
全向量化: 散度/梯度等几何运算零 Python for 循环.
"""
import numpy as np
from typing import Optional


class StaggeredGrid:
    """MAC 交错网格数据容器.

    Attributes:
        nx, ny, nz: 体素数量
        dx, dy, dz: 体素尺寸
        ox, oy, oz: 原点坐标
        u: (nx+1, ny, nz) float64 — x-面心速度
        v: (nx, ny+1, nz) float64 — y-面心速度
        w: (nx, ny, nz+1) float64 — z-面心速度
        p: (nx, ny, nz) float64 — 体素中心压力
    """

    def __init__(self, nx: int, ny: int, nz: int,
                 dx: float, dy: float, dz: float,
                 origin: tuple = (0.0, 0.0, 0.0)):
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.dx, self.dy, self.dz = float(dx), float(dy), float(dz)
        self.ox, self.oy, self.oz = (float(origin[0]), float(origin[1]),
                                     float(origin[2]))

        self.u = np.zeros((self.nx + 1, self.ny, self.nz), dtype=np.float64)
        self.v = np.zeros((self.nx, self.ny + 1, self.nz), dtype=np.float64)
        self.w = np.zeros((self.nx, self.ny, self.nz + 1), dtype=np.float64)
        self.p = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)

    # ------------------------------------------------------------------
    # 工厂方法 — 对齐 UniformHexMesh.from_xvoxel
    # ------------------------------------------------------------------
    @classmethod
    def from_xvoxel(cls, xv) -> 'StaggeredGrid':
        """从 XVoxelModel 创建交错网格 (工厂模式).

        Args:
            xv: XVoxelModel 实例 (只读消费).

        Returns:
            StaggeredGrid 实例.
        """
        return cls(xv.nx, xv.ny, xv.nz, xv.dx, xv.dy, xv.dz,
                   origin=(xv.ox, xv.oy, xv.oz))

    # ------------------------------------------------------------------
    # 几何坐标查询 (向量化)
    # ------------------------------------------------------------------
    def u_face_centers(self) -> np.ndarray:
        """x-面心坐标 (nx+1, ny, nz, 3)."""
        x = self.ox + np.arange(self.nx + 1) * self.dx
        y = self.oy + (np.arange(self.ny) + 0.5) * self.dy
        z = self.oz + (np.arange(self.nz) + 0.5) * self.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return np.stack([X, Y, Z], axis=-1)

    def v_face_centers(self) -> np.ndarray:
        """y-面心坐标 (nx, ny+1, nz, 3)."""
        x = self.ox + (np.arange(self.nx) + 0.5) * self.dx
        y = self.oy + np.arange(self.ny + 1) * self.dy
        z = self.oz + (np.arange(self.nz) + 0.5) * self.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return np.stack([X, Y, Z], axis=-1)

    def cell_centers(self) -> np.ndarray:
        """体素中心坐标 (nx, ny, nz, 3)."""
        x = self.ox + (np.arange(self.nx) + 0.5) * self.dx
        y = self.oy + (np.arange(self.ny) + 0.5) * self.dy
        z = self.oz + (np.arange(self.nz) + 0.5) * self.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return np.stack([X, Y, Z], axis=-1)

    # ------------------------------------------------------------------
    # 散度 / 梯度 (全向量化)
    # ------------------------------------------------------------------
    def divergence(self) -> np.ndarray:
        """全向量化散度计算 ∇·u 在体素中心.

        Returns:
            (nx, ny, nz) float64 — 每个体素中心的散度.
        """
        div = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        # x 方向: (u[i+1,j,k] - u[i,j,k]) / dx
        div += (self.u[1:, :, :] - self.u[:-1, :, :]) / self.dx
        # y 方向: (v[i,j+1,k] - v[i,j,k]) / dy
        div += (self.v[:, 1:, :] - self.v[:, :-1, :]) / self.dy
        # z 方向: (w[i,j,k+1] - w[i,j,k]) / dz
        div += (self.w[:, :, 1:] - self.w[:, :, :-1]) / self.dz
        return div

    def pressure_gradient_x(self) -> np.ndarray:
        """体素中心压力 → x-面心梯度 (nx+1, ny, nz).

        内部面: (p[i,j,k] - p[i-1,j,k]) / dx
        边界面: 0 (由 BC 处理).
        """
        grad = np.zeros((self.nx + 1, self.ny, self.nz), dtype=np.float64)
        grad[1:-1, :, :] = (self.p[1:, :, :] - self.p[:-1, :, :]) / self.dx
        return grad

    def pressure_gradient_y(self) -> np.ndarray:
        """体素中心压力 → y-面心梯度 (nx, ny+1, nz)."""
        grad = np.zeros((self.nx, self.ny + 1, self.nz), dtype=np.float64)
        grad[:, 1:-1, :] = (self.p[:, 1:, :] - self.p[:, :-1, :]) / self.dy
        return grad

    def pressure_gradient_z(self) -> np.ndarray:
        """体素中心压力 → z-面心梯度 (nx, ny, nz+1)."""
        grad = np.zeros((self.nx, self.ny, self.nz + 1), dtype=np.float64)
        grad[:, :, 1:-1] = (self.p[:, :, 1:] - self.p[:, :, :-1]) / self.dz
        return grad

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def cell_volume(self) -> float:
        """单个体素体积."""
        return self.dx * self.dy * self.dz

    def copy(self) -> 'StaggeredGrid':
        """深拷贝."""
        g = StaggeredGrid(self.nx, self.ny, self.nz,
                          self.dx, self.dy, self.dz,
                          origin=(self.ox, self.oy, self.oz))
        g.u = self.u.copy()
        g.v = self.v.copy()
        g.w = self.w.copy()
        g.p = self.p.copy()
        return g
