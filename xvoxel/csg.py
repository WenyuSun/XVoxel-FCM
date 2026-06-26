# -*- coding: utf-8 -*-
"""
csg.py — Feature 抽象基类 + Boolean 组合节点 + SDF 阈值分类

设计原则:
    - Feature 是抽象基类，定义 sdf_batch(points) 统一接口
    - Boolean 是 CSG 组合节点，本身也是 Feature，递归求 SDF
    - CSG 树是显式的单一事实来源 (Single Source of Truth)
    - Feature 不含 nature 字段——操作语义由 add_feature 的 op 参数承载
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional


class BoolOp:
    """CSG 布尔操作类型 (整数枚举)."""
    UNION = 0         # A ∪ B  → min(sdf_A, sdf_B)
    INTERSECTION = 1  # A ∩ B  → max(sdf_A, sdf_B)
    DIFFERENCE = 2    # A \ B  → max(sdf_A, -sdf_B)


class Feature(ABC):
    """特征抽象基类.

    所有可求 SDF 的实体都继承 Feature:
    - 叶子节点: Primitive (Cube, Sphere, ...)
    - 内部节点: Boolean (CSG 布尔组合)

    Attributes:
        feature_id: 唯一 ID (由 XVoxelModel 分配)
        name: 特征名称
    """
    def __init__(self, feature_id: int = -1, name: str = ""):
        self.feature_id = feature_id
        self.name = name
        self._deleted = False

    @abstractmethod
    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        """批量 SDF. points: (N, 3) → (N,)."""
        ...

    def sdf(self, x: float, y: float, z: float) -> float:
        """单点 SDF — 兼容旧 API, 调试用."""
        return float(self.sdf_batch(np.array([[x, y, z]]))[0])


class Boolean(Feature):
    """CSG 布尔组合节点 — 一般树 (n-ary children).

    本身也是一个 Feature, 递归求 SDF.

    Attributes:
        op: BoolOp 枚举值
        children: List[Feature] — 子节点列表

    Examples:
        # 带孔板: Cube - CylinderZ
        plate = Boolean(BoolOp.DIFFERENCE, [Cube(...), CylinderZ(...)])

        # 嵌套: (Cube - CylinderZ) U Sphere
        result = Boolean(BoolOp.UNION, [plate, Sphere(...)])

        # 链式 UNION: A ∪ B ∪ C
        combined = Boolean(BoolOp.UNION, [Cube(...), Sphere(...), CylinderZ(...)])
    """
    def __init__(self, op: int, children: List[Feature],
                 feature_id: int = -1, name: str = "") -> None:
        super().__init__(feature_id, name)
        self.op = op
        self.children = list(children)  # 浅拷贝, 隔离外部修改

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        """递归求 SDF. 对 children 列表做向量化 reduce."""
        if not self.children:
            # 空 Boolean 节点: 返回无穷大 (不影响 min/max reduce)
            return np.full(points.shape[0], np.inf, dtype=np.float64)

        sdfs = [child.sdf_batch(points) for child in self.children]

        if self.op == BoolOp.UNION:
            return np.minimum.reduce(sdfs)        # min(sdf_A, sdf_B, ...)
        elif self.op == BoolOp.INTERSECTION:
            return np.maximum.reduce(sdfs)        # max(sdf_A, sdf_B, ...)
        else:  # DIFFERENCE: A\B\C\... = max(sdf_A, -sdf_B, -sdf_C, ...)
            # 全向量化: 符号数组 [1, -1, -1, ...] × stack → max(axis=0)
            signs = -np.ones(len(sdfs), dtype=np.float64)
            signs[0] = 1.0
            stacked = np.stack(sdfs, axis=0)
            # 广播符号到 (len(sdfs),) 或 (len(sdfs), 1)
            return np.max(stacked * signs[:, None], axis=0)


def classify_sdfs(sdf_vals: np.ndarray, min_half_dim: float = 0.1) -> np.ndarray:
    """
    批量 SDF 分类 (模块级纯函数) — 与旧版 occupancy-based 算法对齐.

    旧版算法: 对每个 feature 单独判断:
        sdf <= -min_half_dim → occupancy=1 (完全占据)
        -min_half_dim < sdf < 0 → occupancy=0 (部分占据/边界)
    对 CSG UNION 树, min(sdf_i) 等价于逐 feature 判断的 union.

    Args:
        sdf_vals: (N,) SDF 值数组.
        min_half_dim: 体素最小半尺寸 = min(dx,dy,dz)*0.5, 默认 0.1 兼容旧版.

    Returns:
        (N,) int8 数组: +1=内部 (solid), 0=边界 (boundary), -1=外部 (void).
    """
    result = np.full_like(sdf_vals, -1, dtype=np.int8)
    # 关键: 与旧版对齐 — sdf <= -min_half_dim → 完全占据 (solid)
    result[sdf_vals <= -min_half_dim] = 1
    # -min_half_dim < sdf < 0 → 边界 (与旧版 occupancy=0 一致)
    boundary_mask = (sdf_vals > -min_half_dim) & (sdf_vals < 0)
    result[boundary_mask] = 0
    return result


# ============================================================
# 保留旧接口兼容 (仅用于 src/ 旧代码, 新代码不使用)
# ============================================================

class AttribEntry:
    """体素属性条目 (保留用于旧代码兼容)."""
    __slots__ = ('feature_id', 'nature', 'occupancy', 'sdf_at_center')

    def __init__(self, feature_id, nature, occupancy, sdf_at_center=0.0):
        self.feature_id = feature_id
        self.nature = nature
        self.occupancy = occupancy  # 1=complete, 0=partial
        self.sdf_at_center = sdf_at_center


def classify_point_sdf(sdf_val, tol=1e-8):
    """单点 SDF 分类 (复用 classify_sdfs)."""
    return int(classify_sdfs(np.array([sdf_val]), tol=tol)[0])
