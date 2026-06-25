# -*- coding: utf-8 -*-
"""
pmc.py — 点隶属分类 (Point Membership Classification)
基于特征历史确定空间点的材料归属
"""
import numpy as np
from .csg import AttribEntry, classify_point_sdf


def pmc_point_3d(x, y, z, voxel_attrs, features):
    """
    在给定体素上，对点 (x,y,z) 进行隶属分类
    返回: +1 = 固体内部, 0 = 边界, -1 = 虚空

    算法（论文第4.2节）：
    从特征列表末尾向前追溯（逆向遍历），对每个特征计算 SDF。
    第一个包含该点（SDF < 0）的特征决定最终归属：
    - nature > 0: 固体 (+1)
    - nature < 0: 虚空 (-1)
    如果点恰好在边界上（SDF == 0）：边界 (0)
    如果没有任何特征包含该点：虚空 (-1)
    """
    if not voxel_attrs:
        return -1  # 无特征 → 虚空

    # 论文算法: 从后向前遍历特征列表（逆向/筛选）
    # 先构建 feature_id → feature 的快速查找表
    feature_map = {}
    for f in features:
        if not f._deleted:
            feature_map[f.feature_id] = f

    # 逆向遍历：最后一个添加的特征优先
    for entry in reversed(voxel_attrs):
        feature = feature_map.get(entry.feature_id)
        if feature is None:
            continue

        sdf_val = feature.sdf(x, y, z)

        if sdf_val < 0:  # 点在特征内部 → 该特征决定归属
            if entry.nature > 0:
                return 1   # 加材料 → 固体
            else:
                return -1  # 减材料 → 虚空
        elif sdf_val == 0:  # 点在边界上
            return 0

    # 没有任何特征包含该点（都在外部）
    return -1


def pmc_batch(centers, voxel_attrs_list, features):
    """
    批量 PMC 判断
    centers: (N, 3) 点坐标数组
    voxel_attrs_list: List of voxel attribute lists
    features: 特征列表
    返回: (N,) 状态数组
    """
    N = len(centers)
    results = np.zeros(N, dtype=np.int8)
    for i in range(N):
        results[i] = pmc_point_3d(
            centers[i, 0], centers[i, 1], centers[i, 2],
            voxel_attrs_list[i], features
        )
    return results