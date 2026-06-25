# -*- coding: utf-8 -*-
"""
xvoxel.py — XVoxel 扩展体素数据结构 v2

核心变化 (v2 vs v1):
    - 显式 CSG 树替代扁平特征列表 + voxel_attrs
    - voxel_nature 通过 csg_root.sdf_batch() 直接求值, 单一事实来源
    - 增量更新: add/delete/edit 只重算脏体素
    - 向量化 _voxelize_feature: 批量 SDF, 0 个 Python for 循环
    - 删除 voxel_attrs — PMC 统一走 csg_root.sdf_batch()
"""
import numpy as np
from typing import List, Optional, Dict, Set
from .csg import Feature, Boolean, BoolOp, classify_sdfs


class XVoxelModel:
    """XVoxel 模型 v2: 显式 CSG 树 + 增量体素化.

    Attributes:
        nx, ny, nz: 体素数量
        lx, ly, lz: 总体几何尺寸
        ox, oy, oz: 原点坐标
        n_voxels: 总体素数
        dx, dy, dz: 体素尺寸
        csg_root: CSG 树根节点 (Feature | None)
        features: 特征历史列表 (按添加顺序, feature_id = 索引)
        _feature_ops: {fid: BoolOp} 每个特征的操作类型
        feature_voxels: {fid: np.ndarray[int32]} 每个特征覆盖的脏体素索引
        voxel_nature: (n_voxels,) int8 — +1 固体 / 0 边界 / -1 虚空
        _voxel_centers: (n_voxels, 3) float64 — 预计算体素中心坐标
    """
    def __init__(self, nx: int, ny: int, nz: int,
                 lx: float, ly: float, lz: float,
                 origin: tuple = (0.0, 0.0, 0.0)):
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.lx, self.ly, self.lz = float(lx), float(ly), float(lz)
        self.ox, self.oy, self.oz = float(origin[0]), float(origin[1]), float(origin[2])
        self.n_voxels = self.nx * self.ny * self.nz
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.dz = self.lz / self.nz

        # 预计算所有体素中心坐标 (n_voxels, 3), i 最快
        i_vals = np.arange(self.nx) + 0.5
        j_vals = np.arange(self.ny) + 0.5
        k_vals = np.arange(self.nz) + 0.5
        I, J, K = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
        self._voxel_centers = np.column_stack([
            self.ox + I.ravel('F') * self.dx,
            self.oy + J.ravel('F') * self.dy,
            self.oz + K.ravel('F') * self.dz,
        ])  # (n_voxels, 3) — Fortran order: x fastest, z slowest (matches _idx)

        # CSG 树根
        self.csg_root: Optional[Feature] = None

        # 特征管理
        self.features: List[Feature] = []
        self._next_id: int = 0
        self._feature_ops: Dict[int, int] = {}
        self.feature_voxels: Dict[int, np.ndarray] = {}

        # 体素分类结果
        self.voxel_nature = np.full(self.n_voxels, -1, dtype=np.int8)

    # ---------- 索引 ----------
    def _idx(self, i: int, j: int, k: int) -> int:
        return k * self.nx * self.ny + j * self.nx + i

    def _ijk(self, idx: int):
        k = idx // (self.nx * self.ny)
        residual = idx - k * self.nx * self.ny
        j = residual // self.nx
        i = residual - j * self.nx
        return i, j, k

    def voxel_center(self, i: int, j: int, k: int) -> np.ndarray:
        return np.array([
            self.ox + (i + 0.5) * self.dx,
            self.oy + (j + 0.5) * self.dy,
            self.oz + (k + 0.5) * self.dz,
        ])

    # ---------- 查询 ----------
    def is_solid_voxel(self, vi: int) -> bool:
        return self.voxel_nature[vi] == 1

    def is_void_voxel(self, vi: int) -> bool:
        return self.voxel_nature[vi] == -1

    def is_boundary_voxel(self, vi: int) -> bool:
        return self.voxel_nature[vi] == 0

    def get_fem_mesh_info(self):
        """返回 FEM 网格信息.

        Returns:
            (elem_E, elem_type):
                elem_E: (n_voxels,) float64
                elem_type: (n_voxels,) int8 — 0=void, 1=solid, 2=boundary
        """
        elem_type = self.voxel_nature.copy()
        elem_E = np.ones(self.n_voxels, dtype=np.float64)
        elem_E[elem_type == -1] = 1e-8  # void → αE
        elem_E[elem_type == 0] = 1.0    # boundary → E (fixed: no longer αE!)
        elem_E[elem_type == 1] = 1.0    # solid → E
        return elem_E, elem_type

    # ---------- 特征管理 ----------
    def add_feature(self, feature: Feature, op: int = BoolOp.UNION) -> int:
        """添加特征: 插入 CSG 树 + 增量体素化.

        Args:
            feature: Feature 实例 (Primitive 叶子 或 Boolean 子树).
            op: CSG 操作类型 (BoolOp.UNION / DIFFERENCE / INTERSECTION).

        Returns:
            feature_id.
        """
        fid = self._next_id
        self._next_id += 1
        feature.feature_id = fid
        self.features.append(feature)
        self._feature_ops[fid] = op

        # 插入 CSG 树
        if self.csg_root is None:
            self.csg_root = feature
        else:
            self.csg_root = Boolean(op, [self.csg_root, feature])

        # 增量体素化
        self._voxelize_feature(fid)
        self._update_voxel_nature(fid)
        return fid

    def delete_feature(self, fid: int) -> None:
        """删除特征: 标记删除 + CSG 树摘除 + 脏体素重算.

        通用语义:
            1. 删除以该节点为根的整棵子树
            2. 若父节点变为单孩子, 用孩子替换父节点 (树压缩)
        """
        if fid >= len(self.features):
            return
        feature = self.features[fid]
        feature._deleted = True

        # 取出并清理脏体素
        dirty = self.feature_voxels.pop(fid, None)
        self._feature_ops.pop(fid, None)

        # CSG 树摘除
        self._remove_from_csg_tree(feature)

        # 增量重算脏体素
        if dirty is not None and len(dirty) > 0:
            self._update_voxel_nature_for_indices(dirty)

    def edit_parameter(self, fid: int, param_name: str, new_val) -> np.ndarray:
        """修改参数: 增量更新 CSG 树叶子参数.

        Returns:
            (M,) int32 新旧覆盖体素的并集, 供 FEM 层重装配.
        """
        old_affected = self.feature_voxels.get(fid)
        if old_affected is not None:
            old_affected = old_affected.copy()
        else:
            old_affected = np.array([], dtype=np.int32)

        # 更新参数
        feature = self.features[fid]
        feature.set_param(param_name, new_val)

        # 重新扫描
        self._voxelize_feature(fid)

        # 增量重算脏体素
        new_affected = self.feature_voxels.get(fid)
        if new_affected is None:
            new_affected = np.array([], dtype=np.int32)

        dirty = np.union1d(old_affected, new_affected)
        if len(dirty) > 0:
            self._update_voxel_nature_for_indices(dirty)

        return dirty

    # ---------- 体素化 (内部) ----------
    def _voxelize_feature(self, fid: int) -> None:
        """向量化体素化: 批量 SDF → 批量 inside_mask → 存 feature_voxels.

        只记录该 feature 的覆盖范围 (用于 dirty-flag 优化).
        性能: 单特征 SDF, O(n), ~0.2ms (72000 体素). 0 个 Python for 循环.
        """
        feature = self.features[fid]
        sdf_vals = feature.sdf_batch(self._voxel_centers)          # (n_voxels,)

        # 向量化: 找 SDF < 0 的体素 (该特征覆盖的体素)
        inside_mask = sdf_vals < 0
        affected_idx = np.where(inside_mask)[0]                     # (M,)

        # 直接存 np.ndarray[int32], 零转换
        self.feature_voxels[fid] = affected_idx.astype(np.int32)

    def _update_voxel_nature(self, fid: Optional[int] = None) -> None:
        """增量更新体素 nature.

        Args:
            fid: 可选, 指定时只更新该 feature 覆盖的脏体素; 省略时全量重算.
        """
        if self.csg_root is None:
            if fid is not None and fid in self.feature_voxels:
                idx = self.feature_voxels[fid]
                self.voxel_nature[idx] = -1
            else:
                self.voxel_nature[:] = -1
            return

        if fid is not None:
            idx = self.feature_voxels.get(fid)
            if idx is None or len(idx) == 0:
                return
            self.voxel_nature[idx] = classify_sdfs(
                self.csg_root.sdf_batch(self._voxel_centers[idx])
            )
        else:
            self.voxel_nature = classify_sdfs(
                self.csg_root.sdf_batch(self._voxel_centers)
            )

    def _update_voxel_nature_for_indices(self, idx: np.ndarray) -> None:
        """对指定体素索引重新求 CSG 树 SDF + classify."""
        if self.csg_root is None:
            self.voxel_nature[idx] = -1
        else:
            self.voxel_nature[idx] = classify_sdfs(
                self.csg_root.sdf_batch(self._voxel_centers[idx])
            )

    # ---------- CSG 树操作 (内部) ----------
    def _remove_from_csg_tree(self, target: Feature) -> None:
        """从 CSG 树中删除以 target 为根的子树, 并压缩单孩子父节点.

        通用算法:
            1. 递归删除 target 及其所有后代
            2. 从父 Boolean.children 中移除 target
            3. 若父 Boolean 只剩一个孩子, 用孩子替换父节点
            4. 向上传播直到不再有单孩子节点或到达根
        """
        if self.csg_root is target:
            self.csg_root = None
            return

        parent = self._find_parent(self.csg_root, target)
        if parent is None:
            return

        parent.children.remove(target)
        self._collapse_single_child(parent)

    def _collapse_single_child(self, node: Boolean) -> None:
        """若 node 只剩一个孩子, 用孩子替换 node, 递归向上."""
        while True:
            nc = len(node.children)
            if nc >= 2:
                return
            if nc == 0:
                return

            only_child = node.children[0]
            grandparent = self._find_parent(self.csg_root, node)

            if grandparent is None:
                self.csg_root = only_child
                return
            else:
                idx = grandparent.children.index(node)
                grandparent.children[idx] = only_child

            if not isinstance(only_child, Boolean):
                return
            node = only_child

    @staticmethod
    def _find_parent(root: Feature, target: Feature) -> Optional[Boolean]:
        """在 CSG 树中查找 target 的父 Boolean 节点."""
        if not isinstance(root, Boolean):
            return None
        for child in root.children:
            if child is target:
                return root
        for child in root.children:
            result = XVoxelModel._find_parent(child, target)
            if result is not None:
                return result
        return None

    # ---------- 网格信息 ----------
    def get_voxel_centers(self) -> np.ndarray:
        """返回所有体素中心坐标 (n_voxels, 3)."""
        return self._voxel_centers
