# -*- coding: utf-8 -*-
"""
solver.py — FluidSolver 流体求解器外观类

对齐 FCMSolver 的声明式 API:
    solver = FluidSolver(xv, Re=40, U_inf=1.0)
    solver.set_fluid(rho=1.0, nu=0.025)
    solver.add_inlet_bc('xmin', (1.0, 0, 0))
    solver.add_outlet_bc('xmax', p=0.0)
    solver.add_slip_wall('ymin')
    solver.add_slip_wall('ymax')
    u, v, p = solver.solve(max_iter=5000, tol=1e-6)
    Cd = solver.compute_drag_coefficient()
    results = solver.get_results()
"""
import numpy as np
from typing import Tuple, Dict, Optional
from .staggered_grid import StaggeredGrid
from .ibm import IBMForce
from .sharp_ibm import SharpIBMHandler
from .boundary import FluidBC
from .simple_solver import SIMPLESolver


class FluidSolver:
    """FVM 流体求解器外观 — 对齐 FCMSolver 的 API 风格.

    Attributes:
        Re: 雷诺数
        U_inf: 来流速度
        rho: 流体密度
        nu: 运动粘度
        grid: StaggeredGrid 实例
        boundary_method: 'ibm' (源项) 或 'sharp' (尖锐界面)
    """

    def __init__(self, xvoxel_model, Re: float, U_inf: float = 1.0,
                 boundary_method: str = 'ibm'):
        """初始化流体求解器.

        Args:
            xvoxel_model: XVoxelModel 实例 (只读消费).
            Re: 雷诺数 (基于特征长度 D=1.0).
            U_inf: 来流速度.
            boundary_method: 边界处理方法 — 'ibm' (源项体积力, 默认)
                或 'sharp' (尖锐界面, 八叉树+高斯积分定位界面).
        """
        # 只读消费 xvoxel 数据
        self._xv = xvoxel_model

        # 物理参数
        self.Re = float(Re)
        self.U_inf = float(U_inf)
        self.rho = 1.0
        self.nu = self.U_inf / self.Re  # 特征长度 D=1.0 已隐含

        # 边界处理方法
        self.boundary_method = str(boundary_method)

        # 网格 (工厂模式)
        self.grid = StaggeredGrid.from_xvoxel(xvoxel_model)

        # 子模块 — 根据边界方法选择处理器
        if self.boundary_method == 'sharp':
            self._ibm = SharpIBMHandler(xvoxel_model, dt=0.01, rho=self.rho)
        else:
            self._ibm = IBMForce(xvoxel_model, dt=0.01, rho=self.rho)
        self._bc = FluidBC(self.grid)

        # SIMPLE 参数 (默认值)
        self._dt = 0.01
        self._alpha_p = 0.3
        self._alpha_u = 0.7

        # 结果
        self._solution: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._simple: Optional[SIMPLESolver] = None

    # ------------------------------------------------------------------
    # 物性设置 (对齐 set_material)
    # ------------------------------------------------------------------
    def set_fluid(self, rho: float, nu: float) -> None:
        """设置流体物性.

        Args:
            rho: 密度.
            nu: 运动粘度.
        """
        self.rho = float(rho)
        self.nu = float(nu)

    # ------------------------------------------------------------------
    # BC 注册 (惰性存储)
    # ------------------------------------------------------------------
    def add_inlet_bc(self, face: str, velocity: tuple) -> None:
        """注册速度入口 BC.

        Args:
            face: 'xmin'/'xmax'/'ymin'/'ymax'/'zmin'/'zmax'.
            velocity: (u, v, w).
        """
        self._bc.add_inlet(face, velocity)

    def add_outlet_bc(self, face: str, p: float = 0.0) -> None:
        """注册压力出口 BC."""
        self._bc.add_outlet(face, p)

    def add_slip_wall(self, face: str) -> None:
        """注册自由滑移壁面."""
        self._bc.add_slip_wall(face)

    def add_noslip_wall(self, face: str) -> None:
        """注册无滑移壁面."""
        self._bc.add_noslip_wall(face)

    # ------------------------------------------------------------------
    # 求解器参数
    # ------------------------------------------------------------------
    def set_solver_params(self, dt: float = 0.01,
                          alpha_p: float = 0.3,
                          alpha_u: float = 0.7) -> None:
        """设置 SIMPLE 求解参数."""
        self._dt = float(dt)
        self._alpha_p = float(alpha_p)
        self._alpha_u = float(alpha_u)

    # ------------------------------------------------------------------
    # 装配 (对齐 assemble)
    # ------------------------------------------------------------------
    def assemble(self) -> None:
        """初始化网格和子求解器 (在首次 solve 前调用)."""
        # 初始化速度场 (来流)
        self.grid.u[:, :, :] = self.U_inf
        self.grid.v[:, :, :] = 0.0
        self.grid.w[:, :, :] = 0.0
        self.grid.p[:, :, :] = 0.0

    # ------------------------------------------------------------------
    # 求解 (对齐 solve)
    # ------------------------------------------------------------------
    def solve(self, max_iter: int = 5000,
              tol: float = 1e-6,
              momentum_iter: int = 20,
              pressure_iter: int = 50,
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """求解稳态流场.

        Args:
            max_iter: 最大 SIMPLE 外迭代次数.
            tol: 散度收敛容差.
            momentum_iter: 动量方程内迭代次数.
            pressure_iter: 压力修正内迭代次数.
            verbose: 打印收敛历史.

        Returns:
            (u, v, p) 速度场与压力场.
        """
        self.assemble()

        self._simple = SIMPLESolver(
            self.grid, self._xv,
            rho=self.rho, nu=self.nu,
            ibm=self._ibm, bc=self._bc,
            dt=self._dt, alpha_p=self._alpha_p, alpha_u=self._alpha_u,
        )
        u, v, p = self._simple.solve_steady(
            max_iter=max_iter, tol=tol,
            momentum_iter=momentum_iter,
            pressure_iter=pressure_iter,
            verbose=verbose,
        )
        self._solution = (u, v, p)
        return u, v, p

    # ------------------------------------------------------------------
    # 后处理 (对齐 compute_von_mises)
    # ------------------------------------------------------------------
    def compute_drag_coefficient(self, D: float = 1.0) -> float:
        """计算阻力系数 C_d.

        C_d = F_drag / (0.5 * ρ * U_inf² * D * W)

        Args:
            D: 特征长度 (圆柱直径).

        Returns:
            阻力系数.
        """
        if self._solution is None:
            raise RuntimeError("Call solve() first.")
        from .postprocess import compute_drag_coefficient
        return compute_drag_coefficient(self._ibm, self.rho, self.U_inf,
                                        D=D, W=self._xv.lz,
                                        dx_min=self._ibm.dx_min)

    def compute_lift_coefficient(self, D: float = 1.0) -> float:
        """计算升力系数 C_l (用于确认对称性)."""
        if self._solution is None:
            raise RuntimeError("Call solve() first.")
        from .postprocess import compute_lift_coefficient
        return compute_lift_coefficient(self._ibm, self.rho, self.U_inf,
                                        D=D, W=self._xv.lz,
                                        dx_min=self._ibm.dx_min)

    # ------------------------------------------------------------------
    # 结果获取 (对齐 get_results)
    # ------------------------------------------------------------------
    def get_results(self) -> Dict:
        """返回结果字典."""
        if self._solution is None:
            return {}
        u, v, p = self._solution
        return {
            'u': u, 'v': v, 'p': p,
            'drag_coefficient': self.compute_drag_coefficient(),
            'lift_coefficient': self.compute_lift_coefficient(),
            'grid': self.grid,
            'div_history': self._simple.div_history if self._simple else [],
        }
