# -*- coding: utf-8 -*-
"""
fcm/solver.py — FCMSolver 求解器外观类

统一 FCM 全流程: 建网格 → 分类 → 装配 → 边界条件 → 求解 → 应力恢复.
"""
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from typing import Optional, Tuple, Dict
from .mesh import UniformHexMesh
from .assembly import assemble_fcm_k
from .boundary import apply_dirichlet, get_face_fixed_dofs, get_face_traction_nodal_forces


class FCMSolver:
    """FCM 求解器外观.

    Usage:
        solver = FCMSolver(xvoxel_model, order=1)
        solver.add_dirichlet_bc('xmin', 'ux,uy,uz', 0.0)
        solver.add_traction_bc('xmax', (100.0, 0, 0))
        u = solver.solve()
        solver.compute_stress(u)
    """

    def __init__(self, xvoxel_model, order: int = 1):
        """
        Args:
            xvoxel_model: XVoxelModel 实例
            order: 形函数阶次 (1=Hex8, 2=Hex20, 3=Hex32)
        """
        self.xvoxel_model = xvoxel_model
        self.order = order
        self.mesh = UniformHexMesh.from_xvoxel(xvoxel_model, element_order=order)
        self.csg_root = xvoxel_model.csg_root
        self.voxel_nature = xvoxel_model.voxel_nature.copy()

        # 内部状态
        self.K: Optional[csr_matrix] = None
        self.F: Optional[np.ndarray] = None
        self.fixed_dofs: list = []
        self.prescribed_vals: list = []
        self.solution: Optional[np.ndarray] = None

        # 物理参数 (默认值)
        self.E = 2e5   # MPa
        self.nu = 0.3
        self.alpha = 1e-8

    def set_material(self, E: float, nu: float, alpha: float = 1e-8):
        """设置材料参数."""
        self.E = E
        self.nu = nu
        self.alpha = alpha

    def add_dirichlet_bc(self, face_name: str, dof_spec: str, value: float = 0.0):
        """添加 Dirichlet 边界条件 (仅对非 void 单元的节点).

        Args:
            face_name: 'xmin'/'xmax'/'ymin'/'ymax'/'zmin'/'zmax'
            dof_spec: 'ux,uy,uz' 或 'ux' 等逗号分隔字符串
            value: 指定位移值
        """
        all_dofs = get_face_fixed_dofs(self.mesh, face_name)

        # 过滤到非 void 单元的节点
        solid_mask = self.voxel_nature != -1
        non_void_node_set = set()
        for eid in np.where(solid_mask)[0]:
            for nid in self.mesh.elems[eid]:
                non_void_node_set.add(int(nid))
        non_void_nodes = np.array(sorted(non_void_node_set), dtype=np.int32)
        non_void_dofs = set()
        for nid in non_void_nodes:
            non_void_dofs.update([nid*3, nid*3+1, nid*3+2])

        # 根据 dof_spec 过滤自由度
        active = np.zeros(3, dtype=bool)
        if 'ux' in dof_spec:
            active[0] = True
        if 'uy' in dof_spec:
            active[1] = True
        if 'uz' in dof_spec:
            active[2] = True

        filtered_dofs = []
        for d in all_dofs:
            if d in non_void_dofs and active[d % 3]:
                filtered_dofs.append(d)

        if filtered_dofs:
            dofs = np.array(filtered_dofs, dtype=np.int32)
            vals = np.full(len(dofs), value, dtype=np.float64)
            self.fixed_dofs.append(dofs)
            self.prescribed_vals.append(vals)
            print(f"  Dirichlet BC '{face_name}' [{dof_spec}]: {len(dofs)} DOFs fixed (filtered from {len(all_dofs)})")

    def add_traction_bc(self, face_name: str, traction: Tuple[float, float, float]):
        """添加面牵引力 BC.

        Args:
            face_name: 面名称
            traction: (tx, ty, tz) 牵引力向量
        """
        if not hasattr(self, '_tractions'):
            self._tractions = []
        self._tractions.append((face_name, traction))

    def assemble(self, alpha: Optional[float] = None, max_depth: int = 3) -> csr_matrix:
        """装配全局刚度矩阵.

        Returns:
            csr_matrix 全局刚度矩阵
        """
        if alpha is not None:
            self.alpha = alpha

        print(f"\nAssembling FCM K (order={self.order}, alpha={self.alpha:.1e}, "
              f"max_depth={max_depth})...")
        self.K = assemble_fcm_k(
            self.mesh, self.voxel_nature, self.csg_root,
            self.E, self.nu, self.alpha, self.order, max_depth,
        )
        print(f"  K shape: {self.K.shape}, nnz: {self.K.nnz}")
        return self.K

    def assemble_force(self) -> np.ndarray:
        """装配全局载荷向量.

        Returns:
            (ndof,) 力向量
        """
        self.F = np.zeros(self.mesh.ndof, dtype=np.float64)

        # 面牵引力
        if hasattr(self, '_tractions'):
            for face_name, traction in self._tractions:
                # Only apply traction to non-void elements
                solid_mask = self.voxel_nature != -1
                F_face = get_face_traction_nodal_forces(
                    self.mesh, face_name, traction,
                    npe=getattr(self.mesh, '_npe_per_elem', 8),
                    csg_root=self.csg_root,
                    elem_mask=solid_mask,
                )
                self.F += F_face

        print(f"  Force norm: {np.linalg.norm(self.F):.6e}")
        return self.F

    def solve(self, E: Optional[float] = None, nu: Optional[float] = None,
              alpha: Optional[float] = None, max_depth: int = 3,
              solver: str = 'auto') -> np.ndarray:
        """完整求解流程.

        Returns:
            (ndof,) 位移向量
        """
        if E is not None:
            self.E = E
        if nu is not None:
            self.nu = nu
        if alpha is not None:
            self.alpha = alpha

        # 装配
        self.assemble(alpha=alpha, max_depth=max_depth)
        self.assemble_force()

        # 施加 Dirichlet BC
        if not self.fixed_dofs:
            print("  WARNING: No Dirichlet BC specified!")
        else:
            all_fixed = np.concatenate(self.fixed_dofs)
            all_vals = np.concatenate(self.prescribed_vals)
            self.K, self.F = apply_dirichlet(self.K, self.F, all_fixed, all_vals)

        # 求解
        print(f"  Solving ({solver})...")
        self.solution = spsolve(self.K.tocsr(), self.F)
        print(f"  Solution norm: {np.linalg.norm(self.solution):.6e}")
        return self.solution

    def compute_von_mises(self) -> np.ndarray:
        """计算每个体素中心的 von Mises 应力.

        Returns:
            (n_elems,) von Mises 应力数组
        """
        if self.solution is None:
            raise RuntimeError("Call solve() first.")

        from .elements import get_element_info, elastic_matrix_D, _build_B_matrix
        info = get_element_info(self.order)
        npe = info['npe']
        shape_grad_func = info['shape_grad']

        n_elems = self.mesh.n_elems
        von_mises = np.zeros(n_elems, dtype=np.float64)
        D = elastic_matrix_D(self.E, self.nu)

        for eid in range(n_elems):
            if self.voxel_nature[eid] == -1:
                continue  # void 元素

            coords = self.mesh.get_elem_coords(eid)
            elem_nodes = self.mesh.elems[eid]

            # 在单元中心计算应力
            dN = shape_grad_func(0.0, 0.0, 0.0)
            J = dN @ coords
            invJ = np.linalg.inv(J)
            dN_dx = invJ @ dN
            B = _build_B_matrix(dN_dx, npe)

            u_elem = np.zeros(3 * npe)
            for a in range(npe):
                node = elem_nodes[a]
                u_elem[a*3:a*3+3] = self.solution[node*3:node*3+3]

            strain = B @ u_elem
            stress = D @ strain
            s = stress
            von_mises[eid] = np.sqrt(
                s[0]**2 + s[1]**2 + s[2]**2
                - s[0]*s[1] - s[1]*s[2] - s[2]*s[0]
                + 3 * (s[3]**2 + s[4]**2 + s[5]**2)
            )

        return von_mises

    def get_results(self) -> Dict:
        """返回结果字典."""
        if self.solution is None:
            return {}

        max_u = np.max(np.abs(self.solution))
        return {
            'solution': self.solution,
            'max_displacement': max_u,
            'von_mises': self.compute_von_mises(),
            'K': self.K,
            'F': self.F,
            'mesh': self.mesh,
        }
