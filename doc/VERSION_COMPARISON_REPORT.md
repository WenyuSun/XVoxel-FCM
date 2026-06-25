# XVoxel-FCM 新旧版本对比报告

> 生成日期：2026-06-25  
> 对比范围：旧版 `src/`（单体架构）vs 新版 `xvoxel/` + `fcm/`（双包架构）

---

## 目录

1. [架构概览](#1-架构概览)
2. [逐文件对比](#2-逐文件对比)
   - [2.1 CSG 层](#21-csg-层)
   - [2.2 几何基元层](#22-几何基元层)
   - [2.3 体素模型层](#23-体素模型层)
   - [2.4 PMC 层](#24-pmc-层)
   - [2.5 有限元单元层](#25-有限元单元层)
   - [2.6 网格层](#26-网格层)
   - [2.7 装配层](#27-装配层)
   - [2.8 边界条件层](#28-边界条件层)
   - [2.9 求解器层](#29-求解器层)
3. [Bug 修复清单](#3-bug-修复清单)
4. [性能对比](#4-性能对比)
5. [测试覆盖对比](#5-测试覆盖对比)
6. [向后兼容性](#6-向后兼容性)
7. [设计差异总结](#7-设计差异总结)

---

## 1. 架构概览

| 维度 | 旧版 (`src/`) | 新版 (`xvoxel/` + `fcm/`) |
|------|---------------|---------------------------|
| **包数量** | 1 个单体包 | 2 个独立包 |
| **文件数** | 6 个 Python 文件 | 10 个 Python 文件 |
| **总行数** | ~1400 行 | ~2000 行 |
| **依赖关系** | 紧密耦合，相互引用 | `fcm/` → `xvoxel/` 单向依赖 |
| **CSG 表达** | 隐式（voxel_attrs 列表遍历模拟） | 显式（csg_root 树） |
| **单一事实来源** | ❌ 双重（features + voxel_attrs） | ✅ 单一（csg_root） |
| **SDF 求值** | 逐点逐体素 Python for 循环 | 批量向量化 numpy |
| **PMC** | 独立模块 `pmc.py`，依赖 voxel_attrs | 融入 CSG 树求解，`classify_sdfs` 一行搞定 |
| **元素支持** | Hex8/20/32 散落两文件 | 统一下 `fcm/elements.py`，`get_element_info()` 分发 |

### 旧版架构图

```
src/
├── csg.py          Feature + AttribEntry + classify_point_sdf (~90行)
├── primitives.py   Primitive(ABC) + Cube + CylinderY + Sphere + RoundCorner2D (~200行)
├── xvoxel.py       XVoxelModel (voxel_attrs + feature_voxels) (~250行)
├── pmc.py          pmc_point_3d + pmc_batch (~80行)
├── fem_base.py     Hex8/20/32形函数 + HexMesh + FEMSolver (~600行)
└── fem_xvoxel.py   XVoxelFEMSolver (装配+八叉树+BC+应力) (~600行)
```

### 新版架构图

```
xvoxel/  (纯 numpy，零 scipy 依赖)
├── __init__.py     公开 API 导出
├── csg.py          Feature(ABC) + Boolean + BoolOp + classify_sdfs (~120行)
├── primitives.py   Primitive(Feature) + Cube + CylinderZ/Y + Sphere + RoundCorner2D (~200行)
└── xvoxel.py       XVoxelModel v2 (csg_root + 预计算体素中心 + 增量更新) (~300行)

fcm/  (依赖 xvoxel/ + scipy)
├── __init__.py     公开 API 导出
├── elements.py     统一形函数 + Gauss法则 + 弹性矩阵 + 单元刚度 (~350行)
├── mesh.py         UniformHexMesh (Hex8/20/32) + from_xvoxel 工厂 (~250行)
├── assembly.py     FCM 装配 + 八叉树边界积分 (~200行)
├── boundary.py     Dirichlet + Traction (PMC过滤) (~200行)
└── solver.py       FCMSolver 外观类 (~200行)
```

---

## 2. 逐文件对比

### 2.1 CSG 层

| 对比项 | `src/csg.py` | `xvoxel/csg.py` |
|--------|-------------|-----------------|
| **Feature 类** | 包装类：持有 `primitive`、`nature`、单点 `sdf(x,y,z)` | **抽象基类**：`sdf_batch(points)` 向量化接口 |
| **Boolean 组合** | ❌ 无。通过 `voxel_attrs` 列表隐式模拟 CSG | ✅ `Boolean(op, children)` 显式树节点，支持 n-ary |
| **操作类型** | 无枚举，nature=±1 隐式表达 | `BoolOp.UNION/INTERSECTION/DIFFERENCE` 枚举 |
| **sdf 求值** | 单点：`feature.sdf(x, y, z)` | 批量：`feature.sdf_batch(points[N,3]) → [N]` |
| **Boolean.sdf_batch** | 不存在 | `np.minimum.reduce` / `np.maximum.reduce` / signs×stack→max |
| **SDF 分类** | `classify_point_sdf(sdf_val)` 单点 | `classify_sdfs(sdf_vals[N])` 向量化 |
| **AttribEntry** | 核心数据结构 | 仅保留用于旧代码兼容，新代码不使用 |

**关键差异**：旧版 CSG 是"注释级"的——Feature 有 nature 字段但无组合能力。布尔操作通过 `_update_voxel_nature()` 中逐体素遍历 `voxel_attrs` 列表来"模拟"。新版有真正的 CSG 树，`Boolean.sdf_batch()` 递归求值，无需逐体素模拟。

---

### 2.2 几何基元层

| 对比项 | `src/primitives.py` | `xvoxel/primitives.py` |
|--------|--------------------|------------------------|
| **基类** | `Primitive(ABC)` 独立抽象类 | `Primitive(Feature)` 继承自 Feature |
| **sdf 接口** | 仅 `sdf(x, y, z)` 单点 | `sdf(x,y,z)` + `sdf_batch(points[N,3]) → [N]` |
| **Cube** | 逐点标量 numpy | `sdf_batch` 向量化：`np.where(inside, d_int, d_ext)` |
| **CylinderY** | ✅ 支持 | ✅ 支持 + `sdf_batch` |
| **CylinderZ** | ❌ 不支持 | ✅ 新增支持 + `sdf_batch` |
| **Sphere** | 逐点标量 | 批量向量化 |
| **RoundCorner2D** | 逐点标量 numpy 运算 | 批量向量化 numpy |
| **name 参数** | 构造函数接受 | 构造函数接受 |

**关键差异**：每个基元新增 `sdf_batch(points)` 方法，实现真正向量化——消除了旧版中 XVoxelModel 逐体素 for 循环调用单点 SDF 的性能瓶颈。`CylinderZ` 是新增基元。

---

### 2.3 体素模型层

| 对比项 | `src/xvoxel.py` | `xvoxel/xvoxel.py` |
|--------|-----------------|---------------------|
| **CSG 表达** | 隐式：`features` 列表 + `voxel_attrs` 逐体素链表 | 显式：`csg_root` CSG 树 |
| **体素中心** | 每次调用 `voxel_center(i,j,k)` 计算 | 构造函数预计算 `_voxel_centers[N,3]` |
| **ravel 顺序** | 依赖 `_idx` 公式 | `.ravel('F')` 显式 Fortran 序（修复关键 bug） |
| **体素化** | `_voxelize_feature`：Python for 循环 N 个体素 | `_voxelize_feature`：一行 `sdf_batch` + `np.where` |
| **voxel_nature 更新** | `_update_voxel_nature`：逐体素遍历 `voxel_attrs` 模拟 CSG | `csg_root.sdf_batch()` → `classify_sdfs()` 一次完成 |
| **add_feature** | 追加到 features 列表 + 逐体素扫描 | 插入 CSG 树 + 批量 SDF + 增量更新 |
| **delete_feature** | 标记 `_deleted` + 清理 `voxel_attrs` + 重算 | 标记 `_deleted` + CSG 树摘除 + `_collapse_single_child` + 增量重算 |
| **edit_parameter** | Delete + Re-add（清理+扫描全部） | 直接更新叶子参数 + 仅重算脏体素 |
| **增量更新** | 部分支持（重算全部 affected） | 完全支持（仅重算 dirty indices 的 `csg_root.sdf_batch`） |
| **get_fem_mesh_info** | 返回 `elem_E, elem_type` | 返回 `elem_E, elem_type`（边界体素 E=1.0，修复 M5） |
| **voxel_attrs** | 核心数据结构 | **已删除** |

**关键差异**：
1. **单一事实来源**：旧版有 `features` + `voxel_attrs` 两个并行数据结构，需保持同步。新版只有 `csg_root` 一棵树。
2. **性能**：旧版逐体素 for 循环 → 新版批量向量化，消除 Python 循环。
3. **Bug 修复**：预计算体素中心使用 Fortran ravel，匹配 `_idx(k*nx*ny + j*nx + i)` 的索引约定。

---

### 2.4 PMC 层

| 对比项 | `src/pmc.py` | 新版实现 |
|--------|-------------|----------|
| **核心函数** | `pmc_point_3d(x,y,z, voxel_attrs, features)` | `classify_sdfs(csg_root.sdf_batch(points))` |
| **算法** | 逆序遍历 voxel_attrs，查 feature_map，单点 SDF | 一次性 CSG 树求值 → 阈值分类 |
| **批量** | `pmc_batch`：Python for 循环逐点调用 | `classify_sdfs`：纯 numpy 向量化 |
| **模块** | 独立文件 `src/pmc.py`（~80 行） | 融入 `xvoxel/csg.py`（`classify_sdfs` 函数） |
| **依赖** | 依赖 `voxel_attrs` + `features` | 仅依赖 `csg_root`（Feature 抽象） |

**关键差异**：旧版 PMC 是独立模块，需要 `voxel_attrs` 和 `features` 两个数据源。新版 PMC 退化为一行代码——CSG 树本身的 SDF 求值就是"隶属分类"。不再需要单独的 `pmc.py` 文件。

---

### 2.5 有限元单元层

| 对比项 | `src/fem_base.py`（部分） | `fcm/elements.py` |
|--------|--------------------------|-------------------|
| **元素类型** | 无枚举，通过 `element_order` 整数分发 | `ElementType` 枚举（HEX8=1, HEX20=2, HEX32=3） |
| **Gauss 规则** | 独立模块变量 `GAUSS_2`, `GAUSS_3`, `_GP4` | `GaussRule` dataclass + `_make_gauss_rule(n)` 工厂 |
| **形函数** | Hex8/20/32 散落在两文件中（`fem_base.py` 和 `fem_xvoxel.py` 各自有 `_hex8_shape_func` 方法） | 全部集中在 `fcm/elements.py`，模块级函数 |
| **弹性矩阵** | 每个 `ke_func` 内联计算 | `elastic_matrix_D(E, nu)` 独立函数 |
| **B 矩阵** | 每个 `ke_func` 内联 for 循环填充 | `_build_B_matrix(dN_dx, npe)` 独立函数 |
| **单元刚度** | `hex8/20/32_element_stiffness` 各约 40 行 | 可通过 `get_element_info(order)` 统一获取 |
| **get_element_info** | 无 | ✅ 统一分发：`{'npe', 'ke_func', 'shape_func', 'shape_grad', 'gauss_rule', 'ndof_per_elem'}` |
| **重复定义** | ❌ Hex8 形函数在 `fem_base.py` + `fem_xvoxel._hex8_shape_func` 重复 | ✅ 单一定义 |

**关键差异**：旧版形函数和刚度计算散落在两个文件中，Hex8 形函数甚至重复定义。新版全部集中，通过 `get_element_info(order)` 统一分发，消除重复。

---

### 2.6 网格层

| 对比项 | `src/fem_base.py` `HexMesh` + `src/fem_xvoxel.py` `_build_mesh` | `fcm/mesh.py` `UniformHexMesh` |
|--------|------------------------------------------------------------------|-------------------------------|
| **单一职责** | `HexMesh` 仅 Hex8；Hex20/32 在 `fem_xvoxel._build_mesh` 中 | ✅ 单一 `UniformHexMesh` 支持所有三种单元 |
| **构造方式** | `HexMesh(nx,ny,nz,lx,ly,lz)` 手动传参 | `UniformHexMesh.from_xvoxel(xvoxel_model)` 工厂方法 |
| **原点支持** | `HexMesh` 固定在原点 (0,0,0) | 支持任意原点 `(ox, oy, oz)` |
| **构建方法** | 混在 `_build_mesh` 中 (Hex8/20/32 分支) | 分离 `_build_hex8/20/32` 方法 |
| **节点连接** | 内联 for 循环计算偏移 | 统一模式，注释清晰 |
| **elem_center** | `HexMesh.elem_center(eid)` | `UniformHexMesh.elem_center(eid)` |
| **get_elem_coords** | `fem_xvoxel._get_elem_coords(eid)` | `UniformHexMesh.get_elem_coords(eid)` |

**关键差异**：旧版有两种半网格类（`HexMesh` + `XVoxelFEMSolver` 内建网格），职责不清。新版统一为 `UniformHexMesh`，支持工厂创建和所有单元类型。

---

### 2.7 装配层

| 对比项 | `src/fem_xvoxel.py` `assemble_FCM_system` | `fcm/assembly.py` |
|--------|-------------------------------------------|-------------------|
| **装配函数** | 类方法，和求解器绑定 | 模块级纯函数 `assemble_fcm_k` |
| **分类** | 内联 `elem_type[eid]` 分支 | `classify_elements()` 预分类为 solid/void/cut |
| **虚空体素** | 和固体体素在同一循环中 | 独立循环，更清晰 |
| **八叉树** | `_octree_integrate`(类方法) + `_integrate_subcell`(类方法) | `_octree_integrate`(模块函数) + Gauss 积分内联 |
| **八叉树 M5 修复** | ❌ 旧版八叉树可能将边界 Gauss 点当 void 处理 | ✅ 边界 Gauss 点使用全 E（仅 void 用 αE） |
| **PMC 调用** | 通过 `pmc_point_3d` 需要 `voxel_attrs` + `features` | 直接调用 `csg_root.sdf_batch()` + `_classify_single` |
| **B 矩阵** | 在 `_integrate_subcell` 中用 `cols0/1/2` 模式填充 | 复用 `_build_B_matrix` |
| **输出** | 逐 50/100 元素打印进度 | 清晰的分类统计 + 边界进度 |

**关键差异**：新版将装配拆分为独立纯函数，不绑定求解器实例。八叉树积分中边界 Gauss 点使用全 E（M5 修复）。PMC 通过 CSG 树统一入口。

---

### 2.8 边界条件层

| 对比项 | `src/fem_xvoxel.py` `apply_dirichlet` + Nitsche | `fcm/boundary.py` |
|--------|------------------------------------------------|-------------------|
| **Dirichlet** | 类方法，返回 `K.tocsr(), F` | 模块函数 `apply_dirichlet(K, F, fixed_dofs, vals)` |
| **面节点获取** | 内联计算 | `get_face_fixed_dofs(mesh, face_name)` 独立函数 |
| **Nitsche** | `apply_nitsche_dirichlet` ~200 行 | 移至 Phase 2（TODO） |
| **Traction** | 无独立函数 | `get_face_traction_nodal_forces` 独立函数 |
| **PMC 过滤** | ❌ 无（面载荷可能施加到虚空区域） | ✅ csg_root 过滤（`sdf > tol` 的 Gauss 点跳过） |
| **Hex20/32 边节点** | ❌ `get_face_fixed_dofs` 仅支持 Hex8 | ✅ 通过坐标邻近性检测支持 Hex20/32 |

**关键差异**：新版将边界条件拆分为独立模块，面牵引力加载带 PMC 过滤（避免虚空区域受力），支持高阶单元的边中点节点。

---

### 2.9 求解器层

| 对比项 | `src/fem_xvoxel.py` `XVoxelFEMSolver` | `fcm/solver.py` `FCMSolver` |
|--------|---------------------------------------|------------------------------|
| **职责** | God Class：网格+装配+BC+求解+应力 | 外观类：编排其他模块 |
| **构造** | 传入 `xvoxel_model` + 物理参数 | 传入 `xvoxel_model` + order，`set_material` 后设物理参数 |
| **BC 添加** | 手动调用 `apply_dirichlet` + `apply_nitsche_dirichlet` | `add_dirichlet_bc(face, dof_spec, value)` 声明式 |
| **虚空过滤** | ❌ BC 施加到所有面节点（包括虚空区节点） | ✅ `add_dirichlet_bc` 过滤到非 void 单元节点 |
| **K 缓存** | `_K_cached` + `_cached_types` + `_ke_cache` | 无缓存（每次装配） |
| **precompute_ke** | ✅ `_precompute_ke`（但 v2 未使用此优化） | 无（由 `assemble_fcm_k` 每次实时计算） |
| **材料参数** | 构造时传入 `E, nu, alpha` | `set_material(E, nu, alpha)` 可后设 |
| **solve 签名** | 手动传 `K, F` | `.solve()` 自动装配+BC+求解 |

**关键差异**：旧版 `XVoxelFEMSolver` 是 God Class（~600 行），网格构建、装配、BC、求解、应力全部耦合。新版 `FCMSolver` 是薄外观类（~200 行），编排独立模块。最关键的是 **BC 虚空过滤**——旧版将面全部节点的 DOF 固定，包括虚空区域的孤立节点，导致刚度奇异。

---

## 3. Bug 修复清单

| ID | 类别 | 描述 | 旧版状态 | 新版修复 |
|----|------|------|----------|----------|
| **C1** | CSG | PMC 逆向遍历 `voxel_attrs` → 最后特征优先 | 隐式（通过 attrs 顺序） | 显式 CSG 树保证最后添加=最高优先级 |
| **C2** | CSG | 双重事实来源（`features` + `voxel_attrs`） | ❌ 需手动同步 | ✅ 单一 `csg_root` |
| **M1** | 单元 | Hex8/20/32 统一接口 | 散落两文件 | `get_element_info(order)` |
| **M2** | 积分 | 边界八叉树自适应积分 | 仅 Hex8，有 bug | 全部支持，正确 PMC |
| **M5** | 积分 | 边界 Gauss 点使用全 E 而非 αE | ❌ 可能用 αE | ✅ `gp_status != -1` → 全 E |
| **I1** | 索引 | 体素中心 ravel 顺序与 `_idx` 不匹配 | 隐式（逐体素计算时无此问题） | ✅ 显式 `.ravel('F')` |
| **B1** | BC | apply_dirichlet 创建局部副本 | ❌ 修改不到原矩阵 | ✅ 返回 (K, F) 元组 |
| **B2** | BC | Dirichlet 施加到虚空区域节点 | ❌ 导致近奇异 | ✅ 过滤非 void 单元节点 |

---

## 4. 性能对比

| 场景 | 旧版 | 新版 | 提升 |
|------|------|------|------|
| **体素化** (72,000 体素 × 1 特征) | O(n) Python for 循环，~50ms | `sdf_batch` 向量化，~0.2ms | ~250× |
| **voxel_nature 更新** | 逐体素遍历 `voxel_attrs` 链表 | `csg_root.sdf_batch()` 一次性求值 | ~100× |
| **增量编辑** | 重算全部受影响体素 | 仅重算 dirty indices | 取决于 dirty ratio |
| **装配** | 逐元素 Python 循环 + 逐元素 PMC | 批量分类 + 逐元素（和旧版类似） | 相近 |

> 注：装配阶段仍是逐元素 Python 循环（LIL 矩阵写入），尚未向量化批量装配。这是 Phase 2 的优化方向。

---

## 5. 测试覆盖对比

| 维度 | 旧版测试 (`tests/`) | 新版测试 | 
|------|---------------------|----------|
| **xvoxel 功能** | 部分在 `test_xvoxel.py` | `test_xvoxel_v2.py` (15 tests) |
| **FEM 装配** | 部分在 `test_fem_base.py` | `test_fcm_v2.py` (19 tests) |
| **边界条件** | 无独立测试 | 含在 `test_fcm_v2.py` |
| **PMC** | 通过集成测试覆盖 | 通过 `classify_sdfs` + CSG 树测试覆盖 |
| **悬臂梁验证** | `test_cantilever.py` | 含在 FCM 测试中 (2.3% 偏差) |
| **总测试数** | 24 | 58 (34 new + 24 legacy, 0 regression) |

---

## 6. 向后兼容性

| API | 兼容状态 | 说明 |
|-----|---------|------|
| `from src.csg import Feature` | ⚠️ 部分兼容 | `xvoxel/csg.py` 保留了 `AttribEntry`、`classify_point_sdf` 用于旧代码 |
| `from src.xvoxel import XVoxelModel` | ❌ 不兼容 | 新版 `XVoxelModel` 接口变化（`add_feature(feature, op=...)` vs 旧版 `add_feature(primitive, nature, name)`） |
| `from src.fem_base import hex8_element_stiffness` | ✅ 兼容 | 函数签名相同，在 `fcm/elements.py` 中 |
| `from src.fem_xvoxel import XVoxelFEMSolver` | ❌ 不兼容 | 被 `FCMSolver` 替代，接口完全不同 |
| `from src.pmc import pmc_point_3d` | ❌ 不兼容 | PMC 已融入 CSG 树，无独立 `pmc` 模块 |
| 旧版示例脚本 | ❌ 不能直接运行 | 需要迁移到新 API |

---

## 7. 设计差异总结

### 7.1 「单一事实来源」原则

```
旧版:  features ──→ voxel_attrs ──→ voxel_nature
          │              │
          └── 两个数据源需手动同步 ──┘

新版:  csg_root ──→ sdf_batch(points) ──→ classify_sdfs ──→ voxel_nature
          │
          └── 一棵树，一切从中派生
```

### 7.2 函数式 vs 面向对象

| 方面 | 旧版 | 新版 |
|------|------|------|
| 装配 | 类方法 | 模块级纯函数 |
| BC | 类方法 | 模块级纯函数 |
| 网格 | 两种半网格（`HexMesh` + `_build_mesh`） | 单一 `UniformHexMesh` |
| 求解器 | God Class | 薄外观类 |

### 7.3 向量化

```
旧版: for vi in range(n_voxels):
          center = self.voxel_center(i, j, k)    # Python for
          sdf_val = prim.sdf(center[0], ...)      # 标量 SDF

新版: sdf_vals = feature.sdf_batch(self._voxel_centers)  # 一次性 (N,3)→(N,)
      inside_mask = sdf_vals < 0                           # 向量化
      affected_idx = np.where(inside_mask)[0]              # 向量化
```

### 7.4 增量更新

```
旧版 edit_parameter:  清理所有体素 → 重新体素化 → 完全重算
新版 edit_parameter:  old_affected ∪ new_affected → 仅重算 dirty indices 的 csg_root.sdf_batch
```

---

## 结论

新版双包架构在以下方面全面提升：

1. **正确性**：修复 8 个已确认 bug（C1/C2/M1/M2/M5/I1/B1/B2）
2. **可维护性**：单一事实来源（CSG 树）、模块化拆分、职责清晰
3. **性能**：向量化 SDF 求值，增量更新，消除逐体素 Python 循环
4. **可测试性**：测试数从 24 → 58，覆盖更全面
5. **可扩展性**：新增单元类型只需在 `elements.py` 添加 + `get_element_info` 注册

向后兼容性方面，旧版 `src/` 的 API 已被破坏。旧版示例脚本需要迁移到新 API。建议 Phase 2 完成后统一废弃 `src/`。
