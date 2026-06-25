# -*- coding: utf-8 -*-
"""
csg.py — CSG布尔操作与特征管理
Feature: 一个基本体素 + 操作性质 (+1 加材料, -1 减材料)
"""
import numpy as np
from .primitives import Primitive


class Feature:
    """
    单一特征：一个基本体素 + 布尔操作性质
    nature: +1 = 加材料 (additive), -1 = 减材料 (subtractive)
    """
    def __init__(self, primitive, nature=1, feature_id=None, name=""):
        self.primitive = primitive
        self.nature = nature  # +1 加, -1 减
        self.feature_id = feature_id
        self.name = name if name else primitive.name
        self._deleted = False

    def sdf(self, x, y, z):
        return self.primitive.sdf(x, y, z)

    def get_params(self):
        return self.primitive.get_params()

    def set_param(self, name, val):
        self.primitive.set_param(name, val)


class AttribEntry:
    """
    XVoxel 体素属性条目
    对应论文中的有序三元组:
      - feature_id: 特征索引
      - nature: +1 (additive) / -1 (subtractive)
      - occupancy: 0=partial, 1=complete
      - sdf_at_center: 该体素中心处的SDF值(用于排序)
    """
    def __init__(self, feature_id, nature, occupancy, sdf_at_center=0):
        self.feature_id = feature_id
        self.nature = nature
        self.occupancy = occupancy  # 0=部分占据, 1=完全占据
        self.sdf_at_center = sdf_at_center

    def __repr__(self):
        occ = "C" if self.occupancy else "P"
        nat = "+" if self.nature > 0 else "-"
        return f"[F{self.feature_id}{nat}{occ}]"


def classify_point_sdf(sdf_val, tol=1e-8):
    """根据 SDF 值分类: +1=内部, 0=边界, -1=外部"""
    if sdf_val < -tol:
        return 1   # IN
    elif sdf_val > tol:
        return -1  # OUT
    return 0       # ON (边界)


def csg_union(sdf_a, sdf_b):
    """CSG 并集: min(sdf_a, sdf_b)"""
    return min(sdf_a, sdf_b)


def csg_intersect(sdf_a, sdf_b):
    """CSG 交集: max(sdf_a, sdf_b)"""
    return max(sdf_a, sdf_b)


def csg_subtract(sdf_a, sdf_b):
    """CSG 差集: max(sdf_a, -sdf_b)"""
    return max(sdf_a, -sdf_b)