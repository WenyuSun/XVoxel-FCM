# -*- coding: utf-8 -*-
"""
fcm/mesh.py — 规则六面体网格生成

支持 Hex8, Hex20, Hex32 节点布局.
UniformHexMesh 从 XVoxelModel 工厂创建, 节点坐标参数化自动对齐体素网格.
"""
import numpy as np
from typing import Optional
from .elements import (ElementType, HEX8_NODES, HEX20_NODES, HEX32_NODES,
                        get_element_info)


class UniformHexMesh:
    """规则六面体网格 (支持 Hex8/20/32).

    Attributes:
        nx, ny, nz: 单元数
        lx, ly, lz: 几何尺寸
        ox, oy, oz: 原点
        dx, dy, dz: 单元尺寸
        n_nodes: 节点总数
        n_elems: 单元总数
        nodes: (n_nodes, 3) 节点坐标
        elems: (n_elems, npe) 单元连接
        element_order: 单元阶次
        npe: 每个单元的节点数
        ndof: 总自由度数
    """
    def __init__(self, nx: int, ny: int, nz: int,
                 lx: float, ly: float, lz: float,
                 ox: float = 0.0, oy: float = 0.0, oz: float = 0.0,
                 element_order: int = ElementType.HEX8):
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.lx, self.ly, self.lz = float(lx), float(ly), float(lz)
        self.ox, self.oy, self.oz = float(ox), float(oy), float(oz)
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.dz = self.lz / self.nz
        self.element_order = element_order
        self.n_elems = self.nx * self.ny * self.nz

        info = get_element_info(element_order)
        self.npe = info['npe']

        self.nodes: Optional[np.ndarray] = None
        self.elems: Optional[np.ndarray] = None
        self.n_nodes: int = 0
        self.ndof: int = 0
        self._build()

    def _build(self):
        """构建节点坐标和单元连接."""
        x = np.linspace(self.ox, self.ox + self.lx, self.nx + 1)
        y = np.linspace(self.oy, self.oy + self.ly, self.ny + 1)
        z = np.linspace(self.oz, self.oz + self.lz, self.nz + 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        if self.element_order == ElementType.HEX8:
            self._build_hex8(X, Y, Z)
        elif self.element_order == ElementType.HEX20:
            self._build_hex20(X, Y, Z, x, y, z)
        elif self.element_order == ElementType.HEX32:
            self._build_hex32(X, Y, Z, x, y, z)
        else:
            raise ValueError(f"Unsupported element_order={self.element_order}")

        self.ndof = self.n_nodes * 3

    def _build_hex8(self, X, Y, Z):
        """Hex8: 只有角节点."""
        self.nodes = np.column_stack([X.ravel('F'), Y.ravel('F'), Z.ravel('F')])
        self.n_nodes = len(self.nodes)

        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            self.elems = np.zeros((0, 8), dtype=np.int32)
            return

        self.elems = np.zeros((self.n_elems, 8), dtype=np.int32)
        eid = 0
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    n0 = i + j*(self.nx+1) + k*(self.nx+1)*(self.ny+1)
                    self.elems[eid] = [
                        n0, n0+1, n0+1+(self.nx+1), n0+(self.nx+1),
                        n0 + (self.nx+1)*(self.ny+1),
                        n0+1 + (self.nx+1)*(self.ny+1),
                        n0+1+(self.nx+1) + (self.nx+1)*(self.ny+1),
                        n0+(self.nx+1) + (self.nx+1)*(self.ny+1),
                    ]
                    eid += 1

    def _build_hex20(self, X, Y, Z, x, y, z):
        """Hex20: 角节点 + 边中点."""
        nx, ny, nz = self.nx, self.ny, self.nz
        n_corner = (nx+1)*(ny+1)*(nz+1)
        corner_nodes = np.column_stack([X.ravel('F'), Y.ravel('F'), Z.ravel('F')])

        def _build_edge_nodes(dim: str):
            if dim == 'x':
                n_e = nx * (ny+1) * (nz+1)
                en = np.zeros((n_e, 3))
                idx = 0
                for k in range(nz+1):
                    for j in range(ny+1):
                        for i in range(nx):
                            en[idx, 0] = (x[i] + x[i+1]) / 2
                            en[idx, 1] = y[j]
                            en[idx, 2] = z[k]
                            idx += 1
                return en
            elif dim == 'y':
                n_e = (nx+1) * ny * (nz+1)
                en = np.zeros((n_e, 3))
                idx = 0
                for k in range(nz+1):
                    for j in range(ny):
                        for i in range(nx+1):
                            en[idx, 0] = x[i]
                            en[idx, 1] = (y[j] + y[j+1]) / 2
                            en[idx, 2] = z[k]
                            idx += 1
                return en
            else:  # 'z'
                n_e = (nx+1) * (ny+1) * nz
                en = np.zeros((n_e, 3))
                idx = 0
                for k in range(nz):
                    for j in range(ny+1):
                        for i in range(nx+1):
                            en[idx, 0] = x[i]
                            en[idx, 1] = y[j]
                            en[idx, 2] = (z[k] + z[k+1]) / 2
                            idx += 1
                return en

        xedge = _build_edge_nodes('x')
        yedge = _build_edge_nodes('y')
        zedge = _build_edge_nodes('z')

        self.nodes = np.vstack([corner_nodes, xedge, yedge, zedge])
        self.n_nodes = len(self.nodes)

        n_xe = nx * (ny+1) * (nz+1)
        n_ye = (nx+1) * ny * (nz+1)
        off_x = n_corner
        off_y = off_x + n_xe
        off_z = off_y + n_ye

        self.elems = np.zeros((self.n_elems, 20), dtype=np.int32)
        eid = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n0 = i + j*(nx+1) + k*(nx+1)*(ny+1)
                    corners = [n0, n0+1, n0+1+(nx+1), n0+(nx+1),
                               n0+(nx+1)*(ny+1), n0+1+(nx+1)*(ny+1),
                               n0+1+(nx+1)+(nx+1)*(ny+1), n0+(nx+1)+(nx+1)*(ny+1)]
                    edges = [
                        i + j*nx + k*nx*(ny+1) + off_x,
                        i+1 + j*(nx+1) + k*(nx+1)*ny + off_y,
                        i + (j+1)*nx + k*nx*(ny+1) + off_x,
                        i + j*(nx+1) + k*(nx+1)*ny + off_y,
                        i + j*nx + (k+1)*nx*(ny+1) + off_x,
                        i+1 + j*(nx+1) + (k+1)*(nx+1)*ny + off_y,
                        i + (j+1)*nx + (k+1)*nx*(ny+1) + off_x,
                        i + j*(nx+1) + (k+1)*(nx+1)*ny + off_y,
                        i + j*(nx+1) + k*(nx+1)*(ny+1) + off_z,
                        i+1 + j*(nx+1) + k*(nx+1)*(ny+1) + off_z,
                        i+1 + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + off_z,
                        i + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + off_z,
                    ]
                    self.elems[eid] = corners + edges
                    eid += 1

    def _build_hex32(self, X, Y, Z, x, y, z):
        """Hex32: 角节点 + 每条边2个内部节点."""
        nx, ny, nz = self.nx, self.ny, self.nz
        n_corner = (nx+1)*(ny+1)*(nz+1)
        corner_nodes = np.column_stack([X.ravel('F'), Y.ravel('F'), Z.ravel('F')])

        def _build_edge_nodes_2(dim: str):
            n_e = 0
            if dim == 'x':
                n_e = nx * (ny+1) * (nz+1)
            elif dim == 'y':
                n_e = (nx+1) * ny * (nz+1)
            else:
                n_e = (nx+1) * (ny+1) * nz

            en1 = np.zeros((n_e, 3))
            en2 = np.zeros((n_e, 3))
            idx = 0
            if dim == 'x':
                for k in range(nz+1):
                    for j in range(ny+1):
                        for i in range(nx):
                            en1[idx, 0] = x[i] + (x[i+1]-x[i])/3
                            en1[idx, 1] = y[j]
                            en1[idx, 2] = z[k]
                            en2[idx, 0] = x[i] + 2*(x[i+1]-x[i])/3
                            en2[idx, 1] = y[j]
                            en2[idx, 2] = z[k]
                            idx += 1
            elif dim == 'y':
                for k in range(nz+1):
                    for j in range(ny):
                        for i in range(nx+1):
                            en1[idx, 0] = x[i]
                            en1[idx, 1] = y[j] + (y[j+1]-y[j])/3
                            en1[idx, 2] = z[k]
                            en2[idx, 0] = x[i]
                            en2[idx, 1] = y[j] + 2*(y[j+1]-y[j])/3
                            en2[idx, 2] = z[k]
                            idx += 1
            else:
                for k in range(nz):
                    for j in range(ny+1):
                        for i in range(nx+1):
                            en1[idx, 0] = x[i]
                            en1[idx, 1] = y[j]
                            en1[idx, 2] = z[k] + (z[k+1]-z[k])/3
                            en2[idx, 0] = x[i]
                            en2[idx, 1] = y[j]
                            en2[idx, 2] = z[k] + 2*(z[k+1]-z[k])/3
                            idx += 1
            return en1, en2

        xe1, xe2 = _build_edge_nodes_2('x')
        ye1, ye2 = _build_edge_nodes_2('y')
        ze1, ze2 = _build_edge_nodes_2('z')

        self.nodes = np.vstack([corner_nodes, xe1, xe2, ye1, ye2, ze1, ze2])
        self.n_nodes = len(self.nodes)

        n_xe = nx * (ny+1) * (nz+1)
        n_ye = (nx+1) * ny * (nz+1)
        n_ze = (nx+1) * (ny+1) * nz
        off_x1 = n_corner
        off_x2 = off_x1 + n_xe
        off_y1 = off_x2 + n_xe
        off_y2 = off_y1 + n_ye
        off_z1 = off_y2 + n_ye
        off_z2 = off_z1 + n_ze

        self.elems = np.zeros((self.n_elems, 32), dtype=np.int32)
        eid = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n0 = i + j*(nx+1) + k*(nx+1)*(ny+1)
                    corners = [n0, n0+1, n0+1+(nx+1), n0+(nx+1),
                               n0+(nx+1)*(ny+1), n0+1+(nx+1)*(ny+1),
                               n0+1+(nx+1)+(nx+1)*(ny+1), n0+(nx+1)+(nx+1)*(ny+1)]
                    xi = i + j*nx
                    edges = [
                        xi + k*nx*(ny+1) + off_x1, xi + k*nx*(ny+1) + off_x2,
                        (i+1) + j*(nx+1) + k*(nx+1)*ny + off_y1,
                        (i+1) + j*(nx+1) + k*(nx+1)*ny + off_y2,
                        i + (j+1)*nx + k*nx*(ny+1) + off_x1,
                        i + (j+1)*nx + k*nx*(ny+1) + off_x2,
                        i + j*(nx+1) + k*(nx+1)*ny + off_y1,
                        i + j*(nx+1) + k*(nx+1)*ny + off_y2,
                        xi + (k+1)*nx*(ny+1) + off_x1,
                        xi + (k+1)*nx*(ny+1) + off_x2,
                        (i+1) + j*(nx+1) + (k+1)*(nx+1)*ny + off_y1,
                        (i+1) + j*(nx+1) + (k+1)*(nx+1)*ny + off_y2,
                        i + (j+1)*nx + (k+1)*nx*(ny+1) + off_x1,
                        i + (j+1)*nx + (k+1)*nx*(ny+1) + off_x2,
                        i + j*(nx+1) + (k+1)*(nx+1)*ny + off_y1,
                        i + j*(nx+1) + (k+1)*(nx+1)*ny + off_y2,
                        i + j*(nx+1) + k*(nx+1)*(ny+1) + off_z1,
                        i + j*(nx+1) + k*(nx+1)*(ny+1) + off_z2,
                        (i+1) + j*(nx+1) + k*(nx+1)*(ny+1) + off_z1,
                        (i+1) + j*(nx+1) + k*(nx+1)*(ny+1) + off_z2,
                        (i+1) + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + off_z1,
                        (i+1) + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + off_z2,
                        i + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + off_z1,
                        i + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + off_z2,
                    ]
                    self.elems[eid] = corners + edges
                    eid += 1

    @classmethod
    def from_xvoxel(cls, xvoxel_model, element_order: int = ElementType.HEX8):
        """从 XVoxelModel 创建网格."""
        return cls(
            xvoxel_model.nx, xvoxel_model.ny, xvoxel_model.nz,
            xvoxel_model.lx, xvoxel_model.ly, xvoxel_model.lz,
            xvoxel_model.ox, xvoxel_model.oy, xvoxel_model.oz,
            element_order,
        )

    def elem_center(self, eid: int) -> np.ndarray:
        """返回单元 eid 的中心坐标."""
        return self.nodes[self.elems[eid]].mean(axis=0)

    def get_elem_coords(self, eid: int) -> np.ndarray:
        """返回单元 eid 的节点坐标 (npe, 3)."""
        return self.nodes[self.elems[eid]]
