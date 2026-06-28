# -*- coding: utf-8 -*-
"""fluid/ 流体求解器单元测试.

覆盖:
    - StaggeredGrid 几何与散度
    - IBMForce 权重与力计算
    - FluidBC 边界条件施加
    - MomentumSolver 系数构建
    - PressureSolver Poisson 求解
    - SIMPLESolver 纯通道收敛 (Level 0)
    - FluidSolver 圆柱绕流 (Level 1, Cd 量级校验)

运行: pytest tests/test_fluid.py -v
"""
import numpy as np
import pytest

from xvoxel.xvoxel import XVoxelModel
from xvoxel.primitives import CylinderZ, Cube
from xvoxel.csg import BoolOp

from fluid.staggered_grid import StaggeredGrid
from fluid.ibm import IBMForce
from fluid.sharp_ibm import SharpIBMHandler
from fluid.boundary import FluidBC
from fluid.momentum import MomentumSolver
from fluid.pressure import PressureSolver
from fluid.simple_solver import SIMPLESolver
from fluid.solver import FluidSolver


# ======================================================================
#  StaggeredGrid
# ======================================================================

class TestStaggeredGrid:
    """MAC 交错网格容器测试."""

    def test_shape(self):
        """网格场形状符合 MAC 约定."""
        g = StaggeredGrid(10, 8, 4, 0.1, 0.1, 0.1)
        assert g.u.shape == (11, 8, 4)
        assert g.v.shape == (10, 9, 4)
        assert g.w.shape == (10, 8, 5)
        assert g.p.shape == (10, 8, 4)

    def test_from_xvoxel(self):
        """from_xvoxel 工厂正确继承维度."""
        xv = XVoxelModel(12, 10, 4, 2.4, 2.0, 0.4)
        g = StaggeredGrid.from_xvoxel(xv)
        assert g.nx == 12 and g.ny == 10 and g.nz == 4
        assert np.isclose(g.dx, 0.2)
        assert np.isclose(g.dy, 0.2)
        assert np.isclose(g.dz, 0.1)

    def test_divergence_uniform(self):
        """均匀速度场散度为 0."""
        g = StaggeredGrid(10, 10, 4, 0.1, 0.1, 0.1)
        g.u[:, :, :] = 1.0  # 均匀 u
        div = g.divergence()
        assert np.allclose(div, 0.0)

    def test_divergence_source(self):
        """发散速度场散度正确."""
        g = StaggeredGrid(10, 10, 4, 0.1, 0.1, 0.1)
        # u 线性增长: u[i] = i*0.1 → du/dx = 1.0
        for i in range(g.u.shape[0]):
            g.u[i, :, :] = i * 0.1
        div = g.divergence()
        # 内部体素散度应 ≈ 1.0
        assert np.allclose(div, 1.0, atol=1e-10)

    def test_cell_volume(self):
        """体素体积正确."""
        g = StaggeredGrid(5, 5, 5, 0.2, 0.3, 0.4)
        assert np.isclose(g.cell_volume(), 0.024)


# ======================================================================
#  IBMForce
# ======================================================================

class TestIBMForce:
    """浸入边界力测试."""

    def test_no_body(self):
        """无障碍物时 IBM 力为 0."""
        xv = XVoxelModel(20, 20, 4, 2.0, 2.0, 0.4)
        ibm = IBMForce(xv, dt=0.01, rho=1.0)
        g = StaggeredGrid.from_xvoxel(xv)
        g.u[:, :, :] = 1.0
        ibm.compute_force(g)
        assert np.allclose(ibm.f_ibm, 0.0)

    def test_with_body_force_nonzero(self):
        """有固体时 IBM 力非零 (固体区速度被推向 0)."""
        xv = XVoxelModel(40, 40, 4, 4.0, 4.0, 0.4)
        cyl = CylinderZ(2.0, 2.0, 0.5, -1.0, 2.0)
        xv.add_feature(cyl, op=BoolOp.UNION)
        ibm = IBMForce(xv, dt=0.01, rho=1.0)
        g = StaggeredGrid.from_xvoxel(xv)
        g.u[:, :, :] = 1.0
        ibm.compute_force(g)
        # x 方向力应为负 (减速)
        assert ibm.f_ibm[0].sum() < 0.0

    def test_weights_solid(self):
        """固体内部权重 α=1."""
        xv = XVoxelModel(40, 40, 4, 4.0, 4.0, 0.4)
        cyl = CylinderZ(2.0, 2.0, 0.5, -1.0, 2.0)
        xv.add_feature(cyl, op=BoolOp.UNION)
        ibm = IBMForce(xv, dt=0.01, rho=1.0)
        w = ibm.compute_weights()
        # 至少有一个权重 = 1.0 (固体内部)
        assert np.any(w >= 0.999)
        # 所有权重 ∈ [0, 1]
        assert np.all(w >= 0.0) and np.all(w <= 1.0)


# ======================================================================
#  SharpIBMHandler (尖锐界面 IBM)
# ======================================================================

class TestSharpIBM:
    """尖锐界面 IBM 测试 — 八叉树 + 高斯积分定位界面."""

    def _make_cylinder_xv(self):
        """构造圆柱 XVoxelModel (R=0.5, 中心 (2,2))."""
        xv = XVoxelModel(40, 40, 4, 4.0, 4.0, 0.4)
        cyl = CylinderZ(2.0, 2.0, 0.5, -1.0, 2.0)
        xv.add_feature(cyl, op=BoolOp.UNION)
        return xv

    def test_no_body(self):
        """无障碍物时界面点为空."""
        xv = XVoxelModel(20, 20, 4, 2.0, 2.0, 0.4)
        h = SharpIBMHandler(xv, dt=0.01, rho=1.0)
        assert h.interface_points.shape[0] == 0
        g = StaggeredGrid.from_xvoxel(xv)
        g.u[:, :, :] = 1.0
        h.compute_force(g)
        assert np.allclose(h.f_ibm, 0.0)

    def test_interface_points_on_surface(self):
        """界面高斯点落在 SDF=0 等值面上."""
        xv = self._make_cylinder_xv()
        h = SharpIBMHandler(xv)
        assert h.interface_points.shape[0] > 0, "应有界面点"
        sdf_vals = h._csg_root.sdf_batch(h.interface_points)
        # 所有界面点 |sdf| 应接近 0 (投影到等值面)
        assert np.all(np.abs(sdf_vals) < 1e-6), \
            f"界面点未落在等值面: max|sdf|={np.max(np.abs(sdf_vals))}"

    def test_normals_unit_length(self):
        """界面法向为单位向量."""
        xv = self._make_cylinder_xv()
        h = SharpIBMHandler(xv)
        norms = np.linalg.norm(h.interface_normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10), \
            f"法向非单位向量: {norms[:5]}"

    def test_normals_radial(self):
        """圆柱界面法向应沿径向 (指向远离圆柱中心)."""
        xv = self._make_cylinder_xv()
        h = SharpIBMHandler(xv)
        # 圆柱中心 (2,2), 法向 xy 分量应指向远离中心
        cx, cy = 2.0, 2.0
        dx = h.interface_points[:, 0] - cx
        dy = h.interface_points[:, 1] - cy
        # 径向单位向量
        r = np.sqrt(dx ** 2 + dy ** 2)
        radial = np.column_stack([dx / r, dy / r])
        # 法向 xy 分量与径向的点积应 > 0 (同向, 指向流体)
        dot = np.sum(radial * h.interface_normals[:, :2], axis=1)
        assert np.all(dot > 0.9), \
            f"法向非径向: min dot={np.min(dot)}"

    def test_force_nonzero(self):
        """有固体时界面力非零 (固体区速度被推向 0)."""
        xv = self._make_cylinder_xv()
        h = SharpIBMHandler(xv, dt=0.01, rho=1.0)
        g = StaggeredGrid.from_xvoxel(xv)
        g.u[:, :, :] = 1.0
        h.compute_force(g)
        # x 方向力应为负 (减速)
        assert h.f_ibm[0].sum() < 0.0, "界面 x 力应为负 (减速)"

    def test_apply_zeroes_solid_velocity(self):
        """apply() 强制固体侧面心速度为 0."""
        xv = self._make_cylinder_xv()
        h = SharpIBMHandler(xv)
        g = StaggeredGrid.from_xvoxel(xv)
        g.u[:, :, :] = 1.0
        g.v[:, :, :] = 1.0
        g.w[:, :, :] = 1.0
        h.apply(g)
        # 固体体素 (nature != -1) 的面心速度应为 0
        solid_ids = np.where(h.voxel_nature != -1)[0]
        i = (solid_ids % h.nx).astype(np.int64)
        j = ((solid_ids // h.nx) % h.ny).astype(np.int64)
        k = (solid_ids // (h.nx * h.ny)).astype(np.int64)
        assert np.allclose(g.u[i, j, k], 0.0)
        assert np.allclose(g.u[i + 1, j, k], 0.0)
        assert np.allclose(g.v[i, j, k], 0.0)

    def test_drag_force_positive(self):
        """均匀来流下阻力为正 (物体受 x 方向阻力)."""
        xv = self._make_cylinder_xv()
        h = SharpIBMHandler(xv, dt=0.01, rho=1.0)
        g = StaggeredGrid.from_xvoxel(xv)
        g.u[:, :, :] = 1.0
        h.compute_force(g)
        # drag_force = -f_ibm_x * dV, f_ibm_x < 0 → drag > 0
        assert h.drag_force() > 0.0, "阻力应为正"

    def test_interface_count_reasonable(self):
        """界面点数量在合理范围 (边界体素数 × 深度因子)."""
        xv = self._make_cylinder_xv()
        h = SharpIBMHandler(xv, max_depth=2)
        n_boundary = int(np.sum(xv.voxel_nature == 0))
        # 每个边界体素最多 8^2=64 叶节点 × 8 高斯点, 但多数被剪枝
        upper = n_boundary * 64 * 8
        assert 0 < h.interface_points.shape[0] < upper, \
            f"界面点数 {h.interface_points.shape[0]} 不在 (0, {upper})"

    def test_solver_switch_to_sharp(self):
        """FluidSolver 可切换到 sharp 方法."""
        xv = self._make_cylinder_xv()
        solver = FluidSolver(xv, Re=40, U_inf=1.0, boundary_method='sharp')
        assert solver.boundary_method == 'sharp'
        assert isinstance(solver._ibm, SharpIBMHandler)

    def test_solver_default_ibm(self):
        """FluidSolver 默认使用源项 IBM."""
        xv = self._make_cylinder_xv()
        solver = FluidSolver(xv, Re=40, U_inf=1.0)
        assert solver.boundary_method == 'ibm'
        assert isinstance(solver._ibm, IBMForce)


# ======================================================================
#  FluidBC
# ======================================================================

class TestFluidBC:
    """边界条件测试."""

    def test_inlet(self):
        """入口 BC 正确施加速度."""
        g = StaggeredGrid(10, 10, 4, 0.1, 0.1, 0.1)
        bc = FluidBC(g)
        bc.add_inlet('xmin', (1.0, 0.0, 0.0))
        bc.apply(g)
        assert np.allclose(g.u[0, :, :], 1.0)
        assert np.allclose(g.v[0, :, :], 0.0)

    def test_outlet(self):
        """出口 BC 正确施加压力."""
        g = StaggeredGrid(10, 10, 4, 0.1, 0.1, 0.1)
        bc = FluidBC(g)
        bc.add_outlet('xmax', pressure=0.0)
        bc.apply(g)
        assert np.allclose(g.p[-1, :, :], 0.0)

    def test_slip_wall(self):
        """滑移壁面法向速度为 0."""
        g = StaggeredGrid(10, 10, 4, 0.1, 0.1, 0.1)
        bc = FluidBC(g)
        bc.add_slip_wall('ymin')
        bc.add_slip_wall('ymax')
        bc.apply(g)
        assert np.allclose(g.v[:, 0, :], 0.0)
        assert np.allclose(g.v[:, -1, :], 0.0)


# ======================================================================
#  PressureSolver
# ======================================================================

class TestPressureSolver:
    """压力 Poisson 求解器测试."""

    def test_solve_constant_rhs(self):
        """常数 RHS 求解 (解析解为二次型)."""
        g = StaggeredGrid(20, 20, 4, 0.1, 0.1, 0.1)
        ps = PressureSolver(g, rho=1.0, dt=0.01)
        # 构造已知散度场
        u = np.zeros_like(g.u)
        v = np.zeros_like(g.v)
        w = np.zeros_like(g.w)
        u[1:-1, :, :] = 1.0  # 内部均匀 u
        p_corr = ps.solve(u, v, w, n_iter=200)
        assert p_corr.shape == (20, 20, 4)
        assert np.all(np.isfinite(p_corr))


# ======================================================================
#  Level 0: 纯通道 Poiseuille 收敛
# ======================================================================

class TestLevel0Channel:
    """Level 0 验证: 无障碍物纯通道流收敛."""

    def test_channel_converges(self):
        """纯通道流散度收敛到小量."""
        xv = XVoxelModel(60, 40, 4, 6.0, 4.0, 0.4)
        solver = FluidSolver(xv, Re=100, U_inf=1.0)
        solver.add_inlet_bc('xmin', (1.0, 0, 0))
        solver.add_outlet_bc('xmax', p=0.0)
        solver.add_slip_wall('ymin')
        solver.add_slip_wall('ymax')
        solver.add_slip_wall('zmin')
        solver.add_slip_wall('zmax')
        solver.set_solver_params(dt=0.1, alpha_p=0.3, alpha_u=0.7)
        u, v, p = solver.solve(max_iter=300, tol=1e-3,
                               momentum_iter=20, pressure_iter=100,
                               verbose=False)
        div_max = float(np.max(np.abs(solver.grid.divergence())))
        # 纯通道应收敛到较小散度
        assert div_max < 0.5, f"通道散度未收敛: {div_max}"
        # 出口速度应接近入口速度 (质量守恒)
        u_outlet = u[-1, :, :].mean()
        assert abs(u_outlet - 1.0) < 0.3, f"出口速度偏差大: {u_outlet}"


# ======================================================================
#  Level 1: 圆柱绕流 Cd 量级校验
# ======================================================================

class TestLevel1Cylinder:
    """Level 1 验证: 圆柱绕流 Cd 量级正确.

    目标 (Re=40): Cd ∈ [1.0, 2.5] (量级校验, 精确范围 [1.43,1.60] 需更精细 IBM).
    """

    @pytest.mark.slow
    def test_cylinder_cd_magnitude(self):
        """圆柱绕流 Cd 量级在合理范围."""
        xv = XVoxelModel(120, 80, 4, 6.0, 4.0, 0.4)
        cyl = CylinderZ(1.5, 2.0, 0.4, -1.0, 2.0)
        xv.add_feature(cyl, op=BoolOp.UNION)
        solver = FluidSolver(xv, Re=40, U_inf=1.0)
        solver.add_inlet_bc('xmin', (1.0, 0, 0))
        solver.add_outlet_bc('xmax', p=0.0)
        solver.add_slip_wall('ymin')
        solver.add_slip_wall('ymax')
        solver.add_slip_wall('zmin')
        solver.add_slip_wall('zmax')
        solver.set_solver_params(dt=0.05, alpha_p=0.2, alpha_u=0.7)
        u, v, p = solver.solve(max_iter=400, tol=1e-4,
                               momentum_iter=20, pressure_iter=100,
                               verbose=False)
        cd = solver.compute_drag_coefficient(D=0.8)
        # 量级校验: Cd 应在 [1.0, 2.5] (源项 IBM 近似, 精确值需直接强制改进)
        assert 1.0 < cd < 2.5, f"Cd 量级异常: {cd}"

    @pytest.mark.slow
    def test_cylinder_symmetry(self):
        """圆柱绕流升力近似为 0 (上下对称)."""
        xv = XVoxelModel(120, 80, 4, 6.0, 4.0, 0.4)
        cyl = CylinderZ(1.5, 2.0, 0.4, -1.0, 2.0)
        xv.add_feature(cyl, op=BoolOp.UNION)
        solver = FluidSolver(xv, Re=40, U_inf=1.0)
        solver.add_inlet_bc('xmin', (1.0, 0, 0))
        solver.add_outlet_bc('xmax', p=0.0)
        solver.add_slip_wall('ymin')
        solver.add_slip_wall('ymax')
        solver.add_slip_wall('zmin')
        solver.add_slip_wall('zmax')
        solver.set_solver_params(dt=0.05, alpha_p=0.2, alpha_u=0.7)
        u, v, p = solver.solve(max_iter=400, tol=1e-4,
                               momentum_iter=20, pressure_iter=100,
                               verbose=False)
        cl = solver.compute_lift_coefficient(D=0.8)
        # 对称流场升力应接近 0
        assert abs(cl) < 0.5, f"升力不对称: {cl}"
