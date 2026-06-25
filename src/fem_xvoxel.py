# -*- coding: utf-8 -*-
"""
fem_xvoxel.py — XVoxel-aware FCM 求解器
结合 XVoxel 数据结构与 FCM，实现局部更新
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from .fem_base import (hex8_element_stiffness, GAUSS_2, GAUSS_WT,
                        hex20_element_stiffness, HEX20_NODES,
                        hex20_shape_func, hex20_shape_grad,
                        hex32_element_stiffness, HEX32_NODES,
                        hex32_shape_func, hex32_shape_grad)
from .pmc import pmc_point_3d


class XVoxelFEMSolver:
    """
    XVoxel-aware FEM 求解器
    在规则体素网格上使用 FCM 方法
    """
    def __init__(self, xvoxel_model, E=2e11, nu=0.3, alpha=1e-8, element_order=1):
        """
        element_order: 1=Hex8 (linear), 2=Hex20 (serendipity quadratic)
        """
        self.xvoxel = xvoxel_model
        self.E = E
        self.nu = nu
        self.alpha = alpha
        self.element_order = element_order
        nx, ny, nz = xvoxel_model.nx, xvoxel_model.ny, xvoxel_model.nz
        self.nx, self.ny, self.nz = nx, ny, nz

        # 构建节点坐标和单元连接
        self._build_mesh()

        # 刚度缓存
        self._K_cached = None
        self._cached_types = None
        self._ke_cache = None

    def _build_mesh(self):
        """构建规则网格 (支持 Hex8 和 Hex20)"""
        xv = self.xvoxel
        nx, ny, nz = xv.nx, xv.ny, xv.nz
        x = np.linspace(xv.ox, xv.ox + xv.lx, nx + 1)
        y = np.linspace(xv.oy, xv.oy + xv.ly, ny + 1)
        z = np.linspace(xv.oz, xv.oz + xv.lz, nz + 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        if self.element_order == 1:
            # Hex8: 只有角节点
            self.nodes = np.column_stack([X.ravel('F'), Y.ravel('F'), Z.ravel('F')])
            self.n_nodes = len(self.nodes)
            self.ndof = self.n_nodes * 3
            self.nodes_per_elem = 8
            # 单元连接
            self.elems = np.zeros((nx * ny * nz, 8), dtype=np.int32)
            eid = 0
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        n0 = i + j*(nx+1) + k*(nx+1)*(ny+1)
                        self.elems[eid] = [
                            n0, n0+1, n0+1+(nx+1), n0+(nx+1),
                            n0 + (nx+1)*(ny+1),
                            n0+1 + (nx+1)*(ny+1),
                            n0+1+(nx+1) + (nx+1)*(ny+1),
                            n0+(nx+1) + (nx+1)*(ny+1),
                        ]
                        eid += 1
        elif self.element_order == 2:
            # Hex20: 角节点 + 边中点
            # 角节点 (同 Hex8)
            n_corner = (nx+1)*(ny+1)*(nz+1)
            corner_nodes = np.column_stack([X.ravel('F'), Y.ravel('F'), Z.ravel('F')])

            # X 方向边中点: nx * (ny+1) * (nz+1)
            n_xedge = nx * (ny+1) * (nz+1)
            xedge_nodes = np.zeros((n_xedge, 3))
            idx = 0
            for k in range(nz+1):
                for j in range(ny+1):
                    for i in range(nx):
                        xedge_nodes[idx, 0] = (x[i] + x[i+1]) / 2
                        xedge_nodes[idx, 1] = y[j]
                        xedge_nodes[idx, 2] = z[k]
                        idx += 1

            # Y 方向边中点: (nx+1) * ny * (nz+1)
            n_yedge = (nx+1) * ny * (nz+1)
            yedge_nodes = np.zeros((n_yedge, 3))
            idx = 0
            for k in range(nz+1):
                for j in range(ny):
                    for i in range(nx+1):
                        yedge_nodes[idx, 0] = x[i]
                        yedge_nodes[idx, 1] = (y[j] + y[j+1]) / 2
                        yedge_nodes[idx, 2] = z[k]
                        idx += 1

            # Z 方向边中点: (nx+1) * (ny+1) * nz
            n_zedge = (nx+1) * (ny+1) * nz
            zedge_nodes = np.zeros((n_zedge, 3))
            idx = 0
            for k in range(nz):
                for j in range(ny+1):
                    for i in range(nx+1):
                        zedge_nodes[idx, 0] = x[i]
                        zedge_nodes[idx, 1] = y[j]
                        zedge_nodes[idx, 2] = (z[k] + z[k+1]) / 2
                        idx += 1

            self.nodes = np.vstack([corner_nodes, xedge_nodes, yedge_nodes, zedge_nodes])
            self.n_nodes = len(self.nodes)
            self.ndof = self.n_nodes * 3
            self.nodes_per_elem = 20

            # 偏移量
            offset_x = n_corner
            offset_y = offset_x + n_xedge
            offset_z = offset_y + n_yedge

            # 单元连接
            self.elems = np.zeros((nx * ny * nz, 20), dtype=np.int32)
            eid = 0
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        # 角节点 (8个)
                        n0 = i + j*(nx+1) + k*(nx+1)*(ny+1)
                        corners = [
                            n0, n0+1, n0+1+(nx+1), n0+(nx+1),
                            n0 + (nx+1)*(ny+1),
                            n0+1 + (nx+1)*(ny+1),
                            n0+1+(nx+1) + (nx+1)*(ny+1),
                            n0+(nx+1) + (nx+1)*(ny+1),
                        ]
                        # X 边中点 (4个): edges parallel to X
                        xe = [i + j*(nx) + k*(nx)*(ny+1) + offset_x]
                        # 按 Hex20 标准顺序: e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19
                        # e8  (0,-1,-1): xi-edge  at eta=j,   zeta=k
                        # e9  (1,0,-1):  eta-edge at xi=i+1,  zeta=k
                        # e10 (0,1,-1):  xi-edge  at eta=j+1, zeta=k
                        # e11 (-1,0,-1): eta-edge at xi=i,    zeta=k
                        # e12 (0,-1,1):  xi-edge  at eta=j,   zeta=k+1
                        # e13 (1,0,1):   eta-edge at xi=i+1,  zeta=k+1
                        # e14 (0,1,1):   xi-edge  at eta=j+1, zeta=k+1
                        # e15 (-1,0,1):  eta-edge at xi=i,    zeta=k+1
                        edges = [
                            i + j*nx + k*nx*(ny+1) + offset_x,              # e8
                            i+1 + j*(nx+1) + k*(nx+1)*ny + offset_y,        # e9
                            i + (j+1)*nx + k*nx*(ny+1) + offset_x,          # e10
                            i + j*(nx+1) + k*(nx+1)*ny + offset_y,          # e11
                            i + j*nx + (k+1)*nx*(ny+1) + offset_x,          # e12
                            i+1 + j*(nx+1) + (k+1)*(nx+1)*ny + offset_y,    # e13
                            i + (j+1)*nx + (k+1)*nx*(ny+1) + offset_x,      # e14
                            i + j*(nx+1) + (k+1)*(nx+1)*ny + offset_y,      # e15
                            i + j*(nx+1) + k*(nx+1)*(ny+1) + offset_z,      # e16
                            i+1 + j*(nx+1) + k*(nx+1)*(ny+1) + offset_z,    # e17
                            i+1 + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + offset_z, # e18
                            i + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + offset_z,  # e19
                        ]

                        self.elems[eid] = corners + edges
                        eid += 1

        elif self.element_order == 3:
            # Hex32: corner nodes + 2 edge-interior nodes per edge (24 edge nodes)
            n_corner = (nx+1)*(ny+1)*(nz+1)
            corner_nodes = np.column_stack([X.ravel('F'), Y.ravel('F'), Z.ravel('F')])

            # X-edge: 2 nodes per edge, nx * (ny+1) * (nz+1) edges
            n_xe = nx * (ny+1) * (nz+1)
            xedge1 = np.zeros((n_xe, 3))
            xedge2 = np.zeros((n_xe, 3))
            idx = 0
            for k in range(nz+1):
                for j in range(ny+1):
                    for i in range(nx):
                        xedge1[idx, 0] = x[i] + (x[i+1]-x[i])/3
                        xedge1[idx, 1] = y[j]
                        xedge1[idx, 2] = z[k]
                        xedge2[idx, 0] = x[i] + 2*(x[i+1]-x[i])/3
                        xedge2[idx, 1] = y[j]
                        xedge2[idx, 2] = z[k]
                        idx += 1

            # Y-edge: 2 nodes per edge, (nx+1) * ny * (nz+1) edges
            n_ye = (nx+1) * ny * (nz+1)
            yedge1 = np.zeros((n_ye, 3))
            yedge2 = np.zeros((n_ye, 3))
            idx = 0
            for k in range(nz+1):
                for j in range(ny):
                    for i in range(nx+1):
                        yedge1[idx, 0] = x[i]
                        yedge1[idx, 1] = y[j] + (y[j+1]-y[j])/3
                        yedge1[idx, 2] = z[k]
                        yedge2[idx, 0] = x[i]
                        yedge2[idx, 1] = y[j] + 2*(y[j+1]-y[j])/3
                        yedge2[idx, 2] = z[k]
                        idx += 1

            # Z-edge: 2 nodes per edge, (nx+1) * (ny+1) * nz edges
            n_ze = (nx+1) * (ny+1) * nz
            zedge1 = np.zeros((n_ze, 3))
            zedge2 = np.zeros((n_ze, 3))
            idx = 0
            for k in range(nz):
                for j in range(ny+1):
                    for i in range(nx+1):
                        zedge1[idx, 0] = x[i]
                        zedge1[idx, 1] = y[j]
                        zedge1[idx, 2] = z[k] + (z[k+1]-z[k])/3
                        zedge2[idx, 0] = x[i]
                        zedge2[idx, 1] = y[j]
                        zedge2[idx, 2] = z[k] + 2*(z[k+1]-z[k])/3
                        idx += 1

            self.nodes = np.vstack([corner_nodes, xedge1, xedge2,
                                     yedge1, yedge2, zedge1, zedge2])
            self.n_nodes = len(self.nodes)
            self.ndof = self.n_nodes * 3
            self.nodes_per_elem = 32

            off_x1 = n_corner
            off_x2 = off_x1 + n_xe
            off_y1 = off_x2 + n_xe
            off_y2 = off_y1 + n_ye
            off_z1 = off_y2 + n_ye
            off_z2 = off_z1 + n_ze

            self.elems = np.zeros((nx*ny*nz, 32), dtype=np.int32)
            eid = 0
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        n0 = i + j*(nx+1) + k*(nx+1)*(ny+1)
                        corners = [
                            n0, n0+1, n0+1+(nx+1), n0+(nx+1),
                            n0+(nx+1)*(ny+1), n0+1+(nx+1)*(ny+1),
                            n0+1+(nx+1)+(nx+1)*(ny+1), n0+(nx+1)+(nx+1)*(ny+1),
                        ]
                        # Hex32 edge connectivity matching HEX32_NODES order
                        # z=-1 face: edges 8-15
                        # e8(-1/3,-1,-1), e9(1/3,-1,-1): edge 0→1 (bottom front)
                        # e10(1,-1/3,-1), e11(1,1/3,-1): edge 1→2 (bottom right)
                        # e12(-1/3,1,-1), e13(1/3,1,-1): edge 2→3 (bottom back, reversed)
                        # e14(-1,-1/3,-1), e15(-1,1/3,-1): edge 3→0 (bottom left)
                        # z=+1 face: edges 16-23
                        # ζ-edges: 24-31

                        # xedge idx: i + j*nx + k*nx*(ny+1)
                        # yedge idx: i + j*(nx+1) + k*(nx+1)*ny
                        # zedge idx: i + j*(nx+1) + k*(nx+1)*(ny+1)
                        xi = i + j*nx

                        edges = [
                            # z=-1 face edges (8-15)
                            xi + k*nx*(ny+1) + off_x1,              # e8
                            xi + k*nx*(ny+1) + off_x2,              # e9
                            (i+1) + j*(nx+1) + k*(nx+1)*ny + off_y1,  # e10 (fixed: right edge)
                            (i+1) + j*(nx+1) + k*(nx+1)*ny + off_y2,  # e11 (fixed)
                            i + (j+1)*nx + k*nx*(ny+1) + off_x1,   # e12 (fixed: back edge)
                            i + (j+1)*nx + k*nx*(ny+1) + off_x2,   # e13 (fixed)
                            i + j*(nx+1) + k*(nx+1)*ny + off_y1,   # e14 (left edge)
                            i + j*(nx+1) + k*(nx+1)*ny + off_y2,   # e15 (left edge)
                            # z=+1 face edges (16-23)
                            xi + (k+1)*nx*(ny+1) + off_x1,          # e16
                            xi + (k+1)*nx*(ny+1) + off_x2,          # e17
                            (i+1) + j*(nx+1) + (k+1)*(nx+1)*ny + off_y1,  # e18 (fixed: right edge)
                            (i+1) + j*(nx+1) + (k+1)*(nx+1)*ny + off_y2,  # e19 (fixed)
                            i + (j+1)*nx + (k+1)*nx*(ny+1) + off_x1,  # e20 (fixed: back edge)
                            i + (j+1)*nx + (k+1)*nx*(ny+1) + off_x2,  # e21 (fixed)
                            i + j*(nx+1) + (k+1)*(nx+1)*ny + off_y1,  # e22 (left edge)
                            i + j*(nx+1) + (k+1)*(nx+1)*ny + off_y2,  # e23 (left edge)
                            # zeta-edges (24-31)
                            i + j*(nx+1) + k*(nx+1)*(ny+1) + off_z1,  # e24
                            i + j*(nx+1) + k*(nx+1)*(ny+1) + off_z2,  # e25
                            (i+1) + j*(nx+1) + k*(nx+1)*(ny+1) + off_z1,  # e26 (fixed)
                            (i+1) + j*(nx+1) + k*(nx+1)*(ny+1) + off_z2,  # e27 (fixed)
                            (i+1) + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + off_z1,  # e28 (fixed)
                            (i+1) + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + off_z2,  # e29 (fixed)
                            i + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + off_z1,  # e30 (fixed)
                            i + (j+1)*(nx+1) + k*(nx+1)*(ny+1) + off_z2,  # e31 (fixed)
                        ]

                        self.elems[eid] = corners + edges
                        eid += 1
        else:
            raise ValueError(f"Unsupported element_order={self.element_order}. Use 1 (Hex8), 2 (Hex20), or 3 (Hex32).")

    def _precompute_ke(self):
        """预计算标准单元刚度矩阵 (E=1)"""
        if self._ke_cache is not None:
            return
        # 构建一个参考单元（单位立方体）
        ref_coords = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64)
        # 缩放到实际的体素尺寸
        xv = self.xvoxel
        coords = ref_coords * np.array([xv.dx, xv.dy, xv.dz])
        self._ke_unit = hex8_element_stiffness(coords, 1.0, self.nu)

    def _get_elem_coords(self, eid):
        """获取单元 eid 的节点坐标"""
        return self.nodes[self.elems[eid]]

    def _get_ke_func(self, order):
        if order == 1:
            return hex8_element_stiffness
        elif order == 2:
            return hex20_element_stiffness
        elif order == 3:
            return hex32_element_stiffness
        raise ValueError(f"Unsupported element_order: {order}")

    def assemble_FCM_system(self, active_voxels=None):
        """
        装配 FCM 刚度矩阵和载荷向量
        支持 Hex8 (order=1), Hex20 (order=2), Hex32 (order=3)
        """
        xv = self.xvoxel
        n_elems = xv.n_voxels
        K = lil_matrix((self.ndof, self.ndof), dtype=np.float64)
        F = np.zeros(self.ndof, dtype=np.float64)
        order = self.element_order
        npe = self.nodes_per_elem
        ndof_per_elem = npe * 3
        ke_func = self._get_ke_func(order)

        _, elem_type = xv.get_fem_mesh_info()

        for eid in range(n_elems):
            etype = elem_type[eid]
            coords = self._get_elem_coords(eid)
            dofs = np.array([n*3 + d for n in self.elems[eid] for d in range(3)])

            if etype == 0:  # 虚空体素 → α 方法小刚度
                ke = ke_func(coords, self.E * self.alpha, self.nu)
                for a in range(ndof_per_elem):
                    for b in range(ndof_per_elem):
                        K[dofs[a], dofs[b]] += ke[a, b]
                if (eid + 1) % 100 == 0:
                    print(f"\r  Assembly: {eid+1}/{n_elems} elements...", end='', flush=True)
                continue

            if etype == 1:  # 全固体体素
                ke = ke_func(coords, self.E, self.nu)
                for a in range(ndof_per_elem):
                    for b in range(ndof_per_elem):
                        K[dofs[a], dofs[b]] += ke[a, b]
                if (eid + 1) % 100 == 0:
                    print(f"\r  Assembly: {eid+1}/{n_elems} elements...", end='', flush=True)

            else:  # etype == 2 边界体素 → 八叉树自适应积分
                ke = self._assemble_boundary_ke(eid, coords)
                if ke is not None:
                    for a in range(ndof_per_elem):
                        for b in range(ndof_per_elem):
                            K[dofs[a], dofs[b]] += ke[a, b]
                else:
                    ke = ke_func(coords, self.E * self.alpha, self.nu)
                    for a in range(ndof_per_elem):
                        for b in range(ndof_per_elem):
                            K[dofs[a], dofs[b]] += ke[a, b]
                if etype == 2 and (eid + 1) % 50 == 0:
                    print(f"\r  Assembly: {eid+1}/{n_elems} (cut voxels...)", end='', flush=True)

        print(f"\r  Assembly: {n_elems}/{n_elems} elements done.   ")

        return K.tocsr(), F

    def _assemble_boundary_ke(self, eid, coords):
        """
        对边界体素用自适应八叉树积分
        支持 Hex8/Hex20/Hex32
        """
        voxel_attrs = self.xvoxel.voxel_attrs[eid]
        features = self.xvoxel.features
        ndof_per_elem = self.nodes_per_elem * 3
        ke = np.zeros((ndof_per_elem, ndof_per_elem), dtype=np.float64)
        has_material = np.array([False])

        self._octree_integrate(
            np.array([-1.0, -1.0, -1.0]),
            np.array([ 1.0,  1.0,  1.0]),
            coords, voxel_attrs, features, ke, has_material, depth=0
        )

        return ke if has_material[0] else None

    def _octree_integrate(self, lo, hi, coords, voxel_attrs, features,
                          ke, has_material, depth):
        """八叉树递归积分, 支持 Hex8/Hex20/Hex32"""
        MAX_DEPTH = 4  # Paper requires d=4 for Example #1 (L-shape)
        order = self.element_order

        # 子单元8个角点的 PMC 状态
        corners = np.array([
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
        ])
        all_same, first_status = True, None
        for c in corners:
            if order == 3:
                N = hex32_shape_func(c[0], c[1], c[2])
            elif order == 2:
                N = hex20_shape_func(c[0], c[1], c[2])
            else:
                N = self._hex8_shape_func(c[0], c[1], c[2])
            phys = N @ coords
            st = pmc_point_3d(phys[0], phys[1], phys[2], voxel_attrs, features)
            if first_status is None:
                first_status = st
            elif st != first_status:
                all_same = False

        if (all_same and first_status != -1) or depth >= MAX_DEPTH:
            self._integrate_subcell(lo, hi, coords, voxel_attrs,
                                    features, ke, has_material)
            return

        if all_same and first_status == -1:
            return

        mid = (lo + hi) * 0.5
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    sub_lo = np.array([
                        lo[0] if i == 0 else mid[0],
                        lo[1] if j == 0 else mid[1],
                        lo[2] if k == 0 else mid[2],
                    ])
                    sub_hi = np.array([
                        mid[0] if i == 0 else hi[0],
                        mid[1] if j == 0 else hi[1],
                        mid[2] if k == 0 else hi[2],
                    ])
                    self._octree_integrate(sub_lo, sub_hi, coords,
                                           voxel_attrs, features,
                                           ke, has_material, depth + 1)

    def _integrate_subcell(self, lo, hi, coords, voxel_attrs, features,
                           ke, has_material):
        """
        在子单元 [lo, hi] 上做 2×2×2 Gauss 积分
        支持 Hex8, Hex20, Hex32 形函数
        """
        order = self.element_order
        npe = self.nodes_per_elem
        ndof = npe * 3
        half = (hi - lo) * 0.5
        center = (lo + hi) * 0.5
        s = 1.0 / np.sqrt(3)

        # Pre-compute B-matrix column indices
        cols0 = np.arange(npe) * 3       # 0, 3, 6, ...
        cols1 = cols0 + 1                 # 1, 4, 7, ...
        cols2 = cols0 + 2                 # 2, 5, 8, ...

        # Pre-compute D matrix (same for all Gauss points in subcell)
        c = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        nu = self.nu
        D = np.array([
            [1-nu, nu,  nu, 0, 0, 0],
            [nu,  1-nu, nu, 0, 0, 0],
            [nu,  nu, 1-nu, 0, 0, 0],
            [0,   0,   0,   (1-2*nu)/2, 0, 0],
            [0,   0,   0,   0, (1-2*nu)/2, 0],
            [0,   0,   0,   0, 0, (1-2*nu)/2],
        ], dtype=np.float64) * c

        B = np.zeros((6, ndof))

        for si in [-s, s]:
            for sj in [-s, s]:
                for sk in [-s, s]:
                    xi = center[0] + half[0] * si
                    eta = center[1] + half[1] * sj
                    zeta = center[2] + half[2] * sk
                    w = half[0] * half[1] * half[2]

                    if order == 3:
                        N = hex32_shape_func(xi, eta, zeta)
                    elif order == 2:
                        N = hex20_shape_func(xi, eta, zeta)
                    else:
                        N = self._hex8_shape_func(xi, eta, zeta)
                    phys_pt = N @ coords
                    status = pmc_point_3d(
                        phys_pt[0], phys_pt[1], phys_pt[2],
                        voxel_attrs, features
                    )
                    if status == -1:
                        continue

                    if order == 3:
                        dN_nat = hex32_shape_grad(xi, eta, zeta)
                    elif order == 2:
                        dN_nat = hex20_shape_grad(xi, eta, zeta)
                    else:
                        dN_nat = self._hex8_shape_grad(xi, eta, zeta)
                    J = dN_nat @ coords
                    detJ = np.linalg.det(J)
                    if detJ <= 0:
                        continue
                    invJ = np.linalg.inv(J)
                    dN_dx = invJ @ dN_nat

                    # Vectorized B-matrix fill
                    B.fill(0)
                    B[0, cols0] = dN_dx[0, :]
                    B[1, cols1] = dN_dx[1, :]
                    B[2, cols2] = dN_dx[2, :]
                    B[3, cols0] = dN_dx[1, :]
                    B[3, cols1] = dN_dx[0, :]
                    B[4, cols1] = dN_dx[2, :]
                    B[4, cols2] = dN_dx[1, :]
                    B[5, cols0] = dN_dx[2, :]
                    B[5, cols2] = dN_dx[0, :]

                    ke += w * (B.T @ D @ B) * detJ
                    has_material[0] = True

    # ---- Hex8 shape functions ----
    _HEX8_NODES = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ], dtype=np.float64)

    def _hex8_shape_func(self, xi, eta, zeta):
        N = np.empty(8)
        for i in range(8):
            N[i] = 0.125 * (1 + self._HEX8_NODES[i,0]*xi) \
                         * (1 + self._HEX8_NODES[i,1]*eta) \
                         * (1 + self._HEX8_NODES[i,2]*zeta)
        return N

    def _hex8_shape_grad(self, xi, eta, zeta):
        dN = np.empty((3, 8))
        for i in range(8):
            xi_i, et_i, ze_i = self._HEX8_NODES[i]
            dN[0,i] = 0.125 * xi_i * (1+et_i*eta) * (1+ze_i*zeta)
            dN[1,i] = 0.125 * (1+xi_i*xi) * et_i * (1+ze_i*zeta)
            dN[2,i] = 0.125 * (1+xi_i*xi) * (1+et_i*eta) * ze_i
        return dN

    # ---- 边界条件 ----
    def apply_dirichlet(self, K, F, fixed_dofs, fixed_vals=None):
        if fixed_vals is None:
            fixed_vals = np.zeros(len(fixed_dofs))
        K = K.tolil()
        for i, dof in enumerate(fixed_dofs):
            K[dof, :] = 0
            K[:, dof] = 0
            K[dof, dof] = 1.0
            F[dof] = fixed_vals[i]
        return K.tocsr(), F

    def apply_nitsche_dirichlet(self, K, F, face_type, u_D=None, gamma=None):
        """
        Nitsche 弱 Dirichlet BC on a mesh-aligned face.
        face_type: 'xmin','xmax','ymin','ymax','zmin','zmax'
        u_D: prescribed displacement (default: zero)
        gamma: stabilization parameter (default: auto-compute as 100*E/h)
        
        K += K_nitsche, F += F_nitsche (in-place on lil_matrix)
        """
        if u_D is None:
            u_D = np.zeros(3)

        xv = self.xvoxel
        order = self.element_order
        npe = self.nodes_per_elem
        ndof_per_elem = npe * 3
        E, nu = self.E, self.nu

        # Auto-compute gamma: γ = β * E / h
        if gamma is None:
            h = min(xv.dx, xv.dy, xv.dz)
            gamma = 100.0 * E / h

        # Face node indices (local) and outward normal
        if order == 3:
            face_nodes_map = {
                'xmin': [0, 3, 7, 4, 14, 15, 30, 31, 22, 23, 24, 25],
                'xmax': [1, 2, 6, 5, 10, 11, 28, 29, 18, 19, 26, 27],
                'ymin': [0, 1, 5, 4, 8, 9, 26, 27, 16, 17, 24, 25],
                'ymax': [3, 2, 6, 7, 12, 13, 28, 29, 20, 21, 30, 31],
                'zmin': [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15],
                'zmax': [4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23],
            }
        elif order == 2:
            face_nodes_map = {
                'xmin': [0, 3, 7, 4, 11, 15, 19, 16],
                'xmax': [1, 2, 6, 5, 9, 13, 18, 17],
                'ymin': [0, 1, 5, 4, 8, 12, 17, 16],
                'ymax': [2, 3, 7, 6, 10, 15, 19, 18],
                'zmin': [0, 1, 2, 3, 8, 9, 10, 11],
                'zmax': [4, 5, 6, 7, 12, 13, 14, 15],
            }
        else:
            face_nodes_map = {
                'xmin': [0, 3, 7, 4], 'xmax': [1, 2, 6, 5],
                'ymin': [0, 1, 5, 4], 'ymax': [2, 3, 7, 6],
                'zmin': [0, 1, 2, 3], 'zmax': [4, 5, 6, 7],
            }

        outward_normals = {
            'xmin': np.array([-1, 0, 0]), 'xmax': np.array([1, 0, 0]),
            'ymin': np.array([0, -1, 0]), 'ymax': np.array([0, 1, 0]),
            'zmin': np.array([0, 0, -1]), 'zmax': np.array([0, 0, 1]),
        }

        n_vec = outward_normals[face_type]
        fn = face_nodes_map[face_type]
        n_face_nodes = len(fn)

        # Elasticity matrix
        c = E / ((1+nu)*(1-2*nu))
        D = np.array([
            [1-nu, nu,  nu, 0, 0, 0],
            [nu,  1-nu, nu, 0, 0, 0],
            [nu,  nu, 1-nu, 0, 0, 0],
            [0,   0,   0,   (1-2*nu)/2, 0, 0],
            [0,   0,   0,   0, (1-2*nu)/2, 0],
            [0,   0,   0,   0, 0, (1-2*nu)/2],
        ], dtype=np.float64) * c

        # Normal-stress projection matrix N_sigma (3×6)
        nx, ny, nz = n_vec
        N_sigma = np.array([
            [nx, 0,  0,  ny, 0,  nz],
            [0,  ny, 0,  nx, nz, 0],
            [0,  0,  nz, 0,  ny, nx],
        ], dtype=np.float64)

        # Face Gauss quadrature: 3×3 for Hex32 (cubic serendipity), 2×2 for others
        gauss_2_pts = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
        gauss_3_pts = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
        gauss_3_wts = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

        K = K.tolil()
        nitsche_count = 0
        nitsche_max_k = 0.0

        for eid in range(xv.n_voxels):
            elem_nodes = self.elems[eid]
            coords = self._get_elem_coords(eid)
            face_coords = coords[fn]
            tol = 1e-10

            # Check if this element is on the target face
            is_face = False
            if face_type == 'xmin' and abs(coords[0,0] - xv.ox) < tol:
                is_face = True
            elif face_type == 'xmax' and abs(coords[1,0] - (xv.ox+xv.lx)) < tol:
                is_face = True
            elif face_type == 'ymin' and abs(coords[0,1] - xv.oy) < tol:
                is_face = True
            elif face_type == 'ymax' and abs(coords[2,1] - (xv.oy+xv.ly)) < tol:
                is_face = True
            elif face_type == 'zmin' and abs(coords[0,2] - xv.oz) < tol:
                is_face = True
            elif face_type == 'zmax' and abs(coords[4,2] - (xv.oz+xv.lz)) < tol:
                is_face = True

            if not is_face:
                continue

            nitsche_count += 1
            dofs = np.array([n*3 + d for n in elem_nodes for d in range(3)])

            # Select Gauss rule based on face node count
            if n_face_nodes == 12:
                gauss_iter = [(xi, eta, gauss_3_wts[i]*gauss_3_wts[j])
                              for i, xi in enumerate(gauss_3_pts)
                              for j, eta in enumerate(gauss_3_pts)]
            else:
                gauss_iter = [(xi, eta, 1.0)
                              for xi in gauss_2_pts
                              for eta in gauss_2_pts]

            for xi, eta, w in gauss_iter:

                # Face shape functions and Jacobian
                if n_face_nodes == 4:
                    N_face = np.array([
                        0.25*(1-xi)*(1-eta), 0.25*(1+xi)*(1-eta),
                        0.25*(1+xi)*(1+eta), 0.25*(1-xi)*(1+eta),
                    ])
                elif n_face_nodes == 8:
                    N_face = np.array([
                        0.25*(1-xi)*(1-eta)*(-xi-eta-1),
                        0.25*(1+xi)*(1-eta)*( xi-eta-1),
                        0.25*(1+xi)*(1+eta)*( xi+eta-1),
                        0.25*(1-xi)*(1+eta)*(-xi+eta-1),
                        0.5*(1-xi*xi)*(1-eta),
                        0.5*(1+xi)*(1-eta*eta),
                        0.5*(1-xi*xi)*(1+eta),
                        0.5*(1-xi)*(1-eta*eta),
                    ])
                else:  # n_face_nodes == 12 (Hex32)
                    # Use correct cubic serendipity face shape functions
                    from .fem_base import hex32_face_shape_12
                    N_face, dN_xi, dN_eta = hex32_face_shape_12(face_type, xi, eta)

                gp_xyz = N_face @ face_coords

                # PMC check - skip void Gauss points
                gp_status = pmc_point_3d(
                    gp_xyz[0], gp_xyz[1], gp_xyz[2],
                    xv.voxel_attrs[eid], xv.features
                )
                if gp_status == -1:
                    continue

                # Face Jacobian via parametric derivatives
                if n_face_nodes == 4:
                    dN_xi = np.array([-0.25*(1-eta), 0.25*(1-eta),
                                      0.25*(1+eta), -0.25*(1+eta)])
                    dN_eta = np.array([-0.25*(1-xi), -0.25*(1+xi),
                                       0.25*(1+xi), 0.25*(1-xi)])
                elif n_face_nodes == 8:
                    dN_xi = np.array([
                        0.25*(1-eta)*(2*xi+eta), 0.25*(1-eta)*(2*xi-eta),
                        0.25*(1+eta)*(2*xi+eta), 0.25*(1+eta)*(2*xi-eta),
                        -xi*(1-eta), 0.5*(1-eta*eta),
                        -xi*(1+eta), -0.5*(1-eta*eta),
                    ])
                    dN_eta = np.array([
                        0.25*(1-xi)*(xi+2*eta), 0.25*(1+xi)*(-xi+2*eta),
                        0.25*(1+xi)*(xi+2*eta), 0.25*(1-xi)*(-xi+2*eta),
                        -0.5*(1-xi*xi), -(1+xi)*eta,
                        0.5*(1-xi*xi), -(1-xi)*eta,
                    ])
                # else n_face_nodes==12: dN_xi, dN_eta already from hex32_face_shape_12

                # Face Jacobian via parametric derivatives
                if face_type in ['xmin', 'xmax']:
                    dr_dxi  = np.array([0.0, dN_xi @ face_coords[:,1], dN_xi @ face_coords[:,2]])
                    dr_deta = np.array([0.0, dN_eta @ face_coords[:,1], dN_eta @ face_coords[:,2]])
                elif face_type in ['ymin', 'ymax']:
                    dr_dxi  = np.array([dN_xi @ face_coords[:,0], 0.0, dN_xi @ face_coords[:,2]])
                    dr_deta = np.array([dN_eta @ face_coords[:,0], 0.0, dN_eta @ face_coords[:,2]])
                else:
                    dr_dxi  = np.array([dN_xi @ face_coords[:,0], dN_xi @ face_coords[:,1], 0.0])
                    dr_deta = np.array([dN_eta @ face_coords[:,0], dN_eta @ face_coords[:,1], 0.0])

                detJ = np.linalg.norm(np.cross(dr_dxi, dr_deta))
                # For face integration, only face nodes are non-zero
                # Evaluate full 3D shape functions at the Gauss point mapped to 3D
                if face_type in ['xmin', 'xmax']:
                    face_xi_3d = xi
                    face_eta_3d = eta
                    # Find zeta for this face
                    if face_type == 'xmin':
                        zeta_3d = -1.0
                    else:
                        zeta_3d = 1.0
                elif face_type in ['ymin', 'ymax']:
                    face_xi_3d = xi
                    zeta_3d = eta
                    if face_type == 'ymin':
                        face_eta_3d = -1.0
                    else:
                        face_eta_3d = 1.0
                else:
                    face_xi_3d = xi
                    face_eta_3d = eta
                    if face_type == 'zmin':
                        zeta_3d = -1.0
                    else:
                        zeta_3d = 1.0

                # Evaluate 3D shape functions at face Gauss point
                if order == 3:
                    N_3d = hex32_shape_func(face_xi_3d, face_eta_3d, zeta_3d)
                    dN_3d = hex32_shape_grad(face_xi_3d, face_eta_3d, zeta_3d)
                elif order == 2:
                    N_3d = hex20_shape_func(face_xi_3d, face_eta_3d, zeta_3d)
                    dN_3d = hex20_shape_grad(face_xi_3d, face_eta_3d, zeta_3d)
                else:
                    N_3d = self._hex8_shape_func(face_xi_3d, face_eta_3d, zeta_3d)
                    dN_3d = self._hex8_shape_grad(face_xi_3d, face_eta_3d, zeta_3d)

                # B matrix at the face Gauss point
                J = dN_3d @ coords
                invJ = np.linalg.inv(J)
                dN_dx = invJ @ dN_3d

                # Vectorized B-matrix fill
                B = np.zeros((6, ndof_per_elem))
                cols0 = np.arange(npe) * 3
                cols1 = cols0 + 1
                cols2 = cols0 + 2
                B[0, cols0] = dN_dx[0, :]
                B[1, cols1] = dN_dx[1, :]
                B[2, cols2] = dN_dx[2, :]
                B[3, cols0] = dN_dx[1, :]
                B[3, cols1] = dN_dx[0, :]
                B[4, cols1] = dN_dx[2, :]
                B[4, cols2] = dN_dx[1, :]
                B[5, cols0] = dN_dx[2, :]
                B[5, cols2] = dN_dx[0, :]

                # N matrix (3 × ndof_elem) — vectorized
                N_mat = np.zeros((3, ndof_per_elem))
                N_mat[0, cols0] = N_3d
                N_mat[1, cols1] = N_3d
                N_mat[2, cols2] = N_3d

                # Nitsche contributions
                NDB = N_sigma @ D @ B  # (3 × ndof_elem)

                # Penalty: γ * N^T * N * detJ * w
                K_penalty = gamma * (N_mat.T @ N_mat) * detJ * w

                # Consistency: -N^T * NDB * detJ * w
                K_cons = -(N_mat.T @ NDB) * detJ * w

                # Symmetry: -B^T * D^T * N_sigma^T * N * detJ * w
                K_sym = -(B.T @ D.T @ N_sigma.T @ N_mat) * detJ * w

                K_nitsche_gp = K_penalty + K_cons + K_sym
                nitsche_max_k = max(nitsche_max_k, np.max(np.abs(K_nitsche_gp)))

                for a in range(ndof_per_elem):
                    for b in range(ndof_per_elem):
                        K[dofs[a], dofs[b]] += K_nitsche_gp[a, b]

        print(f"  [Nitsche BC] Applied to {nitsche_count} face elements, max|K_nitsche|={nitsche_max_k:.4e}")
        return K.tocsr(), F

    def apply_face_traction(self, K, F, face_type, tractions):
        """
        面载荷: 在指定面上施加分布力, 仅在固体/边界材料区域施加载荷
        face_type: 'xmin','xmax','ymin','ymax','zmin','zmax'
        traction: (tx, ty, tz) 面力分量
        PMC 过滤: 只对物理材料区域 (status != -1) 施加载荷
        """
        xv = self.xvoxel
        F = F.copy()

        for eid in range(xv.n_voxels):
            elem_nodes = self.elems[eid]

            # 确定单元在目标面上的节点
            face_nodes_map = {
                'zmin': [0, 1, 2, 3],    # z=-1 面
                'zmax': [4, 5, 6, 7],    # z=+1 面
                'ymin': [0, 1, 5, 4],    # y=-1 面
                'ymax': [2, 3, 7, 6],    # y=+1 面
                'xmin': [0, 3, 7, 4],    # x=-1 面
                'xmax': [1, 2, 6, 5],    # x=+1 面
            }

            # 判断该单元是否在目标面上
            tol = 1e-10
            is_face = False
            coords = self._get_elem_coords(eid)
            if face_type == 'xmin' and abs(coords[0,0] - xv.ox) < tol:
                is_face = True
            elif face_type == 'xmax' and abs(coords[1,0] - (xv.ox + xv.lx)) < tol:
                is_face = True
            elif face_type == 'ymin' and abs(coords[0,1] - xv.oy) < tol:
                is_face = True
            elif face_type == 'ymax' and abs(coords[2,1] - (xv.oy + xv.ly)) < tol:
                is_face = True
            elif face_type == 'zmin' and abs(coords[0,2] - xv.oz) < tol:
                is_face = True
            elif face_type == 'zmax' and abs(coords[6,2] - (xv.oz + xv.lz)) < tol:
                is_face = True

            if not is_face:
                continue

            fn = face_nodes_map[face_type]
            face_coords = coords[fn]
            # 面载荷积分 (面上 2x2 Gauss)
            face_xi = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
            face_w = np.array([1.0, 1.0])

            for p, xi in enumerate(face_xi):
                for q, eta in enumerate(face_xi):
                    w = face_w[p] * face_w[q]

                    # 2D 形函数在面上
                    N_face = np.array([
                        0.25*(1-xi)*(1-eta),
                        0.25*(1+xi)*(1-eta),
                        0.25*(1+xi)*(1+eta),
                        0.25*(1-xi)*(1+eta),
                    ])

                    # Gauss 点物理坐标
                    gp_xyz = N_face @ face_coords

                    # PMC 过滤: 只对物理材料区域施加载荷
                    gp_status = pmc_point_3d(
                        gp_xyz[0], gp_xyz[1], gp_xyz[2],
                        xv.voxel_attrs[eid], xv.features
                    )
                    if gp_status == -1:  # 虚空 → 跳过
                        continue

                    # Jacobian 行列式
                    dN_face_xi = np.array([
                        -0.25*(1-eta),  0.25*(1-eta),
                         0.25*(1+eta), -0.25*(1+eta),
                    ])
                    dN_face_eta = np.array([
                        -0.25*(1-xi), -0.25*(1+xi),
                         0.25*(1+xi),  0.25*(1-xi),
                    ])

                    if face_type in ['xmin', 'xmax']:
                        dxdxi = dN_face_xi @ face_coords[:, 1]
                        dxdeta = dN_face_eta @ face_coords[:, 1]
                        dzdxi = dN_face_xi @ face_coords[:, 2]
                        dzdeta = dN_face_eta @ face_coords[:, 2]
                        J_face = np.array([[dxdxi, dzdxi], [dxdeta, dzdeta]])
                    elif face_type in ['ymin', 'ymax']:
                        dxdxi = dN_face_xi @ face_coords[:, 0]
                        dxdeta = dN_face_eta @ face_coords[:, 0]
                        dzdxi = dN_face_xi @ face_coords[:, 2]
                        dzdeta = dN_face_eta @ face_coords[:, 2]
                        J_face = np.array([[dxdxi, dzdxi], [dxdeta, dzdeta]])
                    else:
                        dxdxi = dN_face_xi @ face_coords[:, 0]
                        dxdeta = dN_face_eta @ face_coords[:, 0]
                        dydxi = dN_face_xi @ face_coords[:, 1]
                        dydeta = dN_face_eta @ face_coords[:, 1]
                        J_face = np.array([[dxdxi, dydxi], [dxdeta, dydeta]])

                    detJ_face = np.linalg.det(J_face)

                    for a, ni in enumerate(fn):
                        dof = elem_nodes[ni] * 3
                        F[dof]   += w * N_face[a] * tractions[0] * detJ_face
                        F[dof+1] += w * N_face[a] * tractions[1] * detJ_face
                        F[dof+2] += w * N_face[a] * tractions[2] * detJ_face

        return K, F

    # ---- 求解 ----
    def solve(self, K, F):
        u = spsolve(K, F)
        return u

    def compute_element_results(self, u):
        """计算每个单元的位移范数和 von Mises 应力 (Gauss点平均, 支持Hex8/Hex20/Hex32)"""
        xv = self.xvoxel
        n_elems = xv.n_voxels
        disp_norm = np.zeros(n_elems)
        von_mises = np.zeros(n_elems)
        order = self.element_order
        npe = self.nodes_per_elem

        E, nu = self.E, self.nu
        c = E / ((1+nu) * (1-2*nu))
        D = np.array([
            [1-nu, nu,  nu, 0, 0, 0],
            [nu,  1-nu, nu, 0, 0, 0],
            [nu,  nu, 1-nu, 0, 0, 0],
            [0,   0,   0,   (1-2*nu)/2, 0, 0],
            [0,   0,   0,   0, (1-2*nu)/2, 0],
            [0,   0,   0,   0, 0, (1-2*nu)/2],
        ]) * c

        # Gauss points and weights
        if order == 3:
            from .fem_base import _GP4 as gauss_pts, _GP4_WT as gauss_wt
        elif order == 2:
            from .fem_base import GAUSS_3 as gauss_pts, GAUSS_3_WT as gauss_wt
        else:
            gauss_pts = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
            gauss_wt = np.array([1.0, 1.0])

        for eid in range(n_elems):
            nodes = self.elems[eid]
            dofs = np.array([n*3 + d for n in nodes for d in range(3)])
            u_e = u[dofs]

            _, elem_type = xv.get_fem_mesh_info()
            if elem_type[eid] == 0:
                continue

            u_nodes = u_e.reshape(-1, 3)
            disp_norm[eid] = np.linalg.norm(u_nodes.mean(axis=0))

            # Compute von Mises at each Gauss point, then take the MAXIMUM
            # (averaging stress components first would cancel tension/compression in bending)
            coords = self._get_elem_coords(eid)
            ndof = npe * 3
            vm_list = []

            for i, xi in enumerate(gauss_pts):
                for j, eta in enumerate(gauss_pts):
                    for k, zeta in enumerate(gauss_pts):
                        if order == 3:
                            dN_nat = hex32_shape_grad(xi, eta, zeta)
                        elif order == 2:
                            dN_nat = hex20_shape_grad(xi, eta, zeta)
                        else:
                            dN_nat = self._hex8_shape_grad(xi, eta, zeta)

                        J = dN_nat @ coords
                        detJ = np.linalg.det(J)
                        if detJ <= 0:
                            continue
                        invJ = np.linalg.inv(J)
                        dN_dx = invJ @ dN_nat

                        B = np.zeros((6, ndof))
                        for a in range(npe):
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
                        sigma = D @ epsilon
                        sxx, syy, szz, sxy, syz, sxz = sigma
                        svm = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2
                                           + 6*(sxy**2 + syz**2 + sxz**2)))
                        vm_list.append(svm)

            if vm_list:
                # Use maximum von Mises across Gauss points (captures surface stress)
                # For smoother visualization, also track: median and mean
                von_mises[eid] = max(vm_list)

        return disp_norm, von_mises