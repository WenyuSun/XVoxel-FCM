# -*- coding: utf-8 -*-
"""
xvoxel/ — XVoxel-FCM 几何建模层

纯 numpy 实现，不依赖 scipy。职责是将用户几何描述翻译为体素分类。

公开 API:
    XVoxelModel       — 体素模型主类
    Feature           — 特征抽象基类
    Boolean           — CSG 布尔组合节点
    BoolOp            — CSG 操作枚举
    classify_sdfs     — SDF 阈值分类
    Primitive         — 几何基元抽象类
    Cube, CylinderZ, CylinderY, Sphere, RoundCorner2D  — 具体基元
"""
from .csg import Feature, Boolean, BoolOp, classify_sdfs
from .primitives import (Primitive, Cube, CylinderZ, CylinderY, Sphere,
                          RoundCorner2D)
from .xvoxel import XVoxelModel

__all__ = [
    'XVoxelModel',
    'Feature', 'Boolean', 'BoolOp', 'classify_sdfs',
    'Primitive', 'Cube', 'CylinderZ', 'CylinderY', 'Sphere', 'RoundCorner2D',
]
