# -*- coding: utf-8 -*-
"""
xvoxel.py — XVoxel 扩展体素数据结构
体素 = 规则立方网格，每个体素携带特征属性链表
"""
import numpy as np
from copy import deepcopy
from .csg import Feature, AttribEntry, classify_point_sdf


class XVoxelModel:
    """
    XVoxel 模型
    由无序特征列表 + 规则体素网格 + 体素属性链表组成
    """
    def __init__(self, nx, ny, nz, lx, ly, lz, origin=(0,0,0)):
        """
        nx, ny, nz: 体素数量
        lx, ly, lz: 总体几何尺寸
        origin: 左下角坐标 (ox, oy, oz)
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.lx, self.ly, self.lz = lx, ly, lz
        self.ox, self.oy, self.oz = origin
        self.n_voxels = nx * ny * nz
        self.dx = lx / nx  # 体素尺寸
        self.dy = ly / ny
        self.dz = lz / nz

        # 特征列表 (无序)
        self.features = []   # List[Feature]
        self._next_id = 0

        # 体素属性: 每个体素维护一个 AttribEntry 列表
        # 索引方式: [k * nx*ny + j * nx + i]
        self.voxel_attrs = [[] for _ in range(self.n_voxels)]

        # 每个特征关联的体素索引列表 (用于快速定位)
        self.feature_voxels = {}  # {feature_id: set(voxel_idx)}

        # 体素nature缓存: +1=固体, -1=虚空, 0=混合(边界)
        self.voxel_nature = np.zeros(self.n_voxels, dtype=np.int8)

    # ---------- 索引 ----------
    def _idx(self, i, j, k):
        return k * self.nx * self.ny + j * self.nx + i

    def _ijk(self, idx):
        k = idx // (self.nx * self.ny)
        j = (idx - k * self.nx * self.ny) // self.nx
        i = idx - k * self.nx * self.ny - j * self.nx
        return i, j, k

    def voxel_center(self, i, j, k):
        """返回体素 (i,j,k) 的中心坐标"""
        x = self.ox + (i + 0.5) * self.dx
        y = self.oy + (j + 0.5) * self.dy
        z = self.oz + (k + 0.5) * self.dz
        return np.array([x, y, z])

    # ---------- 特征管理 ----------
    def add_feature(self, primitive, nature=1, name=""):
        """
        添加一个新特征并进行体素化
        返回 feature_id
        """
        fid = self._next_id
        self._next_id += 1
        feature = Feature(primitive, nature, fid, name)
        self.features.append(feature)
        self.feature_voxels[fid] = set()
        # 体素化: 确定该特征占据哪些体素
        self._voxelize_feature(fid)
        # 更新体素 nature 缓存
        self._update_voxel_nature()
        return fid

    def delete_feature(self, fid):
        """
        删除特征（逻辑删除：标记 deleted=True，保留条目）
        对应论文的"虚拟删除"——不破坏 feature 间的依赖关系
        """
        for f in self.features:
            if f.feature_id == fid:
                f._deleted = True
        # 清空特征关联的体素
        if fid in self.feature_voxels:
            self.feature_voxels[fid].clear()
        # 清理所有体素属性中该特征的条目
        for vi in range(self.n_voxels):
            self.voxel_attrs[vi] = [a for a in self.voxel_attrs[vi]
                                    if a.feature_id != fid]
        self._update_voxel_nature()

    def edit_parameter(self, fid, param_name, new_val):
        """
        修改特征的参数（Delete + Re-add）
        对应论文的 edit_parameter 操作
        """
        # 找到特征
        feature = None
        for f in self.features:
            if f.feature_id == fid and not f._deleted:
                feature = f
                break
        if feature is None:
            raise ValueError(f"Feature {fid} not found")

        # 记录旧参数影响的体素
        old_affected = self.feature_voxels.get(fid, set()).copy()

        # 更新参数
        feature.set_param(param_name, new_val)

        # 重新体素化该特征
        self._revoxelize_feature(fid)

        # 获取新影响的体素
        new_affected = self.feature_voxels.get(fid, set())

        # 返回活跃体素（受影响的体素 = 新旧影响的并集）
        active_voxels = old_affected | new_affected
        return active_voxels

    def get_active_voxels(self, fid):
        """返回编辑特征 fid 时受影响的体素索引"""
        return self.feature_voxels.get(fid, set()).copy()

    # ---------- 体素化 ----------
    def _voxelize_feature(self, fid):
        """
        对单个特征进行体素化：
        1. 遍历所有体素，计算中心点 SDF
        2. 更新体素属性列表
        """
        feature = self.features[fid]
        prim = feature.primitive
        affected = set()

        for vi in range(self.n_voxels):
            i, j, k = self._ijk(vi)
            center = self.voxel_center(i, j, k)
            sdf_val = prim.sdf(center[0], center[1], center[2])

            # 判断体素是否被该特征占据
            # 占据条件：体素中心在特征内部 (SDF < 0)
            if sdf_val < 0:  # 体素中心在特征内部
                # 确定占据类型
                # 使用各向异性阈值：分别检查每个方向的边界距离
                # 只有当体素在所有方向上都远离边界时，才标记为完全占据
                # 对于薄板问题，z方向总是完全覆盖的
                half_dx = self.dx * 0.5
                half_dy = self.dy * 0.5
                half_dz = self.dz * 0.5
                
                # 检查体素是否完全在特征内部
                # 体素完全在内部的条件：SDF < -max(half_dx, half_dy, half_dz)
                # 即体素中心距离边界比体素半尺寸还远
                min_half = min(half_dx, half_dy, half_dz)
                if sdf_val <= -min_half:
                    occupancy = 1  # 完全占据
                else:
                    occupancy = 0  # 部分占据（边界体素）

                entry = AttribEntry(fid, feature.nature, occupancy, sdf_val)
                self.voxel_attrs[vi].append(entry)
                affected.add(vi)

        self.feature_voxels[fid] = affected

    def _revoxelize_feature(self, fid):
        """重新体素化一个特征（用于参数修改后）"""
        # 先从所有体素属性中移除该特征的所有条目
        for vi in range(self.n_voxels):
            self.voxel_attrs[vi] = [a for a in self.voxel_attrs[vi]
                                    if a.feature_id != fid]
        self.feature_voxels[fid].clear()
        # 重新体素化
        self._voxelize_feature(fid)
        self._update_voxel_nature()

    def _voxel_diag(self):
        """体素对角线长度"""
        return np.sqrt(self.dx**2 + self.dy**2 + self.dz**2)

    def _update_voxel_nature(self):
        """
        更新每个体素的 nature：综合所有特征判断该体素是固体还是虚空
        
        CSG 逻辑（论文 Section 3.2）：
        - 按特征添加顺序处理
        - nature=+1 (加材料): 如果特征覆盖该体素，设为固体
        - nature=-1 (减材料): 如果特征覆盖该体素，设为虚空
        - 并集(union): 多个加材料特征覆盖同一区域 → 固体
        - 差集(difference): 加材料后减材料 → 虚空
        
        实现：模拟 CSG 布尔运算
        - 初始状态: 虚空 (-1)
        - 对每个特征按顺序处理:
          * nature=+1 且 occupancy=1 (完全覆盖): 固体 (1)
          * nature=+1 且 occupancy=0 (部分覆盖): 边界 (0)
          * nature=-1 且 occupancy=1 (完全覆盖): 虚空 (-1)
          * nature=-1 且 occupancy=0 (部分覆盖): 边界 (0)
        """
        for vi in range(self.n_voxels):
            attrs = self.voxel_attrs[vi]
            if not attrs:
                self.voxel_nature[vi] = -1  # 无特征 → 虚空
                continue
            
            # 按特征添加顺序处理（模拟 CSG 布尔运算）
            final_state = -1  # 初始为虚空
            
            for entry in attrs:
                if entry.nature > 0:  # 加材料 (union)
                    if entry.occupancy == 1:  # 完全覆盖
                        final_state = 1  # 固体
                    elif entry.occupancy == 0:  # 部分覆盖
                        if final_state == 1:
                            final_state = 0  # 从固体变为边界
                        elif final_state == -1:
                            final_state = 0  # 从虚空变为边界
                else:  # nature < 0, 减材料 (difference)
                    if entry.occupancy == 1:  # 完全覆盖
                        final_state = -1  # 虚空
                    elif entry.occupancy == 0:  # 部分覆盖
                        if final_state == -1:
                            final_state = 0  # 从虚空变为边界
                        elif final_state == 1:
                            final_state = 0  # 从固体变为边界
            
            self.voxel_nature[vi] = final_state

    # ---------- 查询 ----------
    def is_solid_voxel(self, vi):
        """体素是否为完全固体"""
        return self.voxel_nature[vi] == 1

    def is_void_voxel(self, vi):
        """体素是否为完全虚空"""
        return self.voxel_nature[vi] == -1

    def is_boundary_voxel(self, vi):
        """体素是否为边界（部分占据）"""
        return self.voxel_nature[vi] == 0

    def get_voxel_attrs_sorted(self, vi):
        """按特征添加顺序返回体素的属性列表"""
        return self.voxel_attrs[vi]

    # ---------- 网格转换 ----------
    def get_fem_mesh_info(self):
        """
        返回用于 FEM 求解的网格信息
        返回: (n_voxels, element_E_array, element_type_array)
        element_type: 0=空, 1=实, 2=边界
        """
        elem_E = np.ones(self.n_voxels, dtype=np.float64)
        elem_type = np.zeros(self.n_voxels, dtype=np.int8)
        for vi in range(self.n_voxels):
            if self.is_solid_voxel(vi):
                elem_type[vi] = 1
                elem_E[vi] = 1.0
            elif self.is_void_voxel(vi):
                elem_type[vi] = 0
                elem_E[vi] = 1e-8  # α 方法
            else:
                elem_type[vi] = 2  # 边界
                elem_E[vi] = 1.0  # 边界体素保持E，积分时处理
        return elem_E, elem_type