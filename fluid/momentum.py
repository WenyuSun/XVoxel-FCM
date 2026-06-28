# -*- coding: utf-8 -*-
"""
momentum.py — 动量方程控制体积分离散

u-动量方程在 x-面心 (i+1/2, j, k) 的控制体:
    a_P * u_P = a_E * u_E + a_W * u_W + a_N * u_N + a_S * u_S
              + a_T * u_T + a_B * u_B - (dp/dx) * ΔV + f_ibm * ΔV

系数 (迎风对流 + 中心扩散):
    F_e = ρ * û_e * A_x,  D_e = μ * A_x / dx
    a_E = D_e + max(-F_e, 0)
    a_W = D_w + max( F_w, 0)
    a_N = D_n + max(-F_n, 0)
    a_S = D_s + max( F_s, 0)
    a_T = D_t + max(-F_t, 0)
    a_B = D_b + max( F_b, 0)
    a_P = a_E + a_W + a_N + a_S + a_T + a_B  (连续性满足时 F 项相消)

向量化策略 (借鉴 fcm/assembly.py):
    - 系数构建 100% 向量化 (numpy 切片 + broadcasting)
    - 线性求解用 Red-Black SOR (可向量化, 比 Gauss-Seidel 更适合 numpy)
    - 零 Python for 循环在系数构建路径
"""
import numpy as np


class MomentumSolver:
    """动量方程求解器 (向量化系数构建 + Red-Black SOR).

    Attributes:
        grid: StaggeredGrid 实例
        rho: 流体密度
        nu: 运动粘度 (μ = ρ*ν)
        dt: 伪瞬态时间步长
        alpha_u: 动量欠松弛因子
    """

    def __init__(self, grid, rho: float, nu: float,
                 dt: float = 0.01, alpha_u: float = 0.7):
        self.grid = grid
        self.rho = float(rho)
        self.nu = float(nu)
        self.mu = self.rho * self.nu
        self.dt = float(dt)
        self.alpha_u = float(alpha_u)

    # ------------------------------------------------------------------
    # 系数构建 (全向量化) — u 动量
    # ------------------------------------------------------------------
    def build_u_coefficients(self, u: np.ndarray, v: np.ndarray,
                             w: np.ndarray):
        """全向量化 u-动量系数构建.

        u 存储在 x-面心 (nx+1, ny, nz). 内部 u 面: i ∈ [1, nx-1].

        Returns:
            (aE, aW, aN, aS, aT, aB, aP, Su) 各为 (nx-1, ny, nz) 形状.
            Su = 源项 (含压力梯度 + 体积力 + 伪瞬态).
        """
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        rho, mu = self.rho, self.mu

        # 面截面积
        Ax = dy * dz   # x-面 (u 控制体的东/西面)
        Ay = dx * dz   # y-面
        Az = dx * dy   # z-面

        # 内部 u 面: u[1:-1, :, :]  形状 (nx-1, ny, nz)
        # 邻居: E=u[2:, :, :], W=u[:-2, :, :], N=u[1:-1, 1:, :], S=u[1:-1, :-1, :]
        #       T=u[1:-1, :, 1:], B=u[1:-1, :, :-1]
        u_P = u[1:-1, :, :]
        u_E = u[2:,   :, :]
        u_W = u[:-2,  :, :]

        # u 控制体界面速度 (插值)
        # 东面 (i+1): û_e = 0.5*(u_P + u_E)
        ue = 0.5 * (u_P + u_E)
        uw = 0.5 * (u_W + u_P)

        # v 在 u 控制体南北界面的插值
        # v 形状 (nx, ny+1, nz). u 控制体南面在 (i-1/2 ~ i+1/2, j-1/2)
        # v_s = 0.25*(v[i,j-1]+v[i,j]+v[i-1,j-1]+v[i-1,j]) → 用切片
        # v[i, j, k] 对应 u 控制体 (i+1/2, j-1/2) 的南面
        vn = 0.25 * (v[1:, 1:, :] + v[1:, :-1, :] +
                     v[:-1, 1:, :] + v[:-1, :-1, :])   # 北面 (j+1/2)
        vs = 0.25 * (v[1:, :-1, :] + v[1:, 1:, :] +
                     v[:-1, :-1, :] + v[:-1, 1:, :])
        # 修正: 南面应为 j-1/2, 即 v[:, j-1, :] 和 v[:, j, :] 的平均
        # 重新定义 (清晰起见):
        # u 控制体中心在 (i+1/2, j+1/2, k+1/2) 体素? 不, u 在面心 (i+1/2, j, k)
        # u 控制体: x∈[i, i+1], y∈[j-1/2, j+1/2], z∈[k-1/2, k+1/2]
        # 南面 y=j-1/2: v 在 (i+1/2, j-1/2) = avg(v[i,j-1], v[i+1,j-1])
        # 北面 y=j+1/2: v 在 (i+1/2, j+1/2) = avg(v[i,j], v[i+1,j])
        # 但 v 索引: v[i,j,k] 在 (i+1/2, j, k+1/2)? 不, v 在 y-面心 (i+1/2, j+1/2, k+1/2)体素?
        # 约定: v[i,j,k] 物理位置 (x=(i+0.5)dx, y=j*dy, z=(k+0.5)dz)
        # u[i,j,k] 物理位置 (x=i*dx, y=(j+0.5)dy, z=(k+0.5)dz)
        # u 控制体 (中心 u[i,j,k], i∈[1,nx-1]): x∈[(i-0.5)dx,(i+0.5)dx], y∈[j*dy,(j+1)dy], z∈[k*dz,(k+1)dz]
        # 东面 x=(i+0.5)dx: û_e = 0.5*(u[i]+u[i+1]) ✓
        # 北面 y=(j+1)dy: v 在 (x=(i+0.5)dx, y=(j+1)dy) = 0.5*(v[i,j+1]+v[i+1,j+1])... 
        #   v[i,j,k] 在 (x=(i+0.5)dx, y=j*dy, z=(k+0.5)dz)
        #   北面 y=(j+1)dy 对应 v 索引 j+1: v_n = 0.5*(v[i,j+1,k]+v[i+1,j+1,k])
        # 南面 y=j*dy 对应 v 索引 j: v_s = 0.5*(v[i,j,k]+v[i+1,j,k])
        # u 控制体内部 u[i,j,k] (i∈[1,nx-1]) 对应 v 切片 [i-1:i+1] 即 v[i-1] 和 v[i]
        # 为对齐形状 (nx-1, ny, nz): u 内部 i∈[1,nx-1] → v 索引 i-1, i (共 nx-1 个对)
        vn = 0.5 * (v[0:-1, 1:, :] + v[1:, 1:, :])    # 北面 v (j+1)
        vs = 0.5 * (v[0:-1, 0:-1, :] + v[1:, 0:-1, :])  # 南面 v (j)

        # w 在 u 控制体上下界面的插值
        # w[i,j,k] 在 (x=(i+0.5)dx, y=(j+0.5)dy, z=k*dz)
        # u 控制体上界面 z=(k+1)dz: w_t = 0.5*(w[i-1,j,k+1]+w[i,j,k+1])
        wt = 0.5 * (w[0:-1, :, 1:] + w[1:, :, 1:])    # 上界面 w (k+1)
        wb = 0.5 * (w[0:-1, :, 0:-1] + w[1:, :, 0:-1])  # 下界面 w (k)

        # 质量通量 F = ρ * u_face * Area
        Fe = rho * ue * Ax
        Fw = rho * uw * Ax
        Fn = rho * vn * Ay
        Fs = rho * vs * Ay
        Ft = rho * wt * Az
        Fb = rho * wb * Az

        # 扩散系数 D = μ * Area / d
        De = mu * Ax / dx
        Dw = De
        Dn = mu * Ay / dy
        Ds = Dn
        Dt = mu * Az / dz
        Db = Dt

        # 系数 = 扩散 + 迎风偏置 (全向量化 max)
        aE = De + np.maximum(-Fe, 0.0)
        aW = Dw + np.maximum( Fw, 0.0)
        aN = Dn + np.maximum(-Fn, 0.0)
        aS = Ds + np.maximum( Fs, 0.0)
        aT = Dt + np.maximum(-Ft, 0.0)
        aB = Db + np.maximum( Fb, 0.0)

        # 伪瞬态源项: a_P0 = ρ*ΔV/Δt
        dV = dx * dy * dz
        aP0 = rho * dV / self.dt

        aP = aE + aW + aN + aS + aT + aB + aP0

        return aE, aW, aN, aS, aT, aB, aP, aP0

    # ------------------------------------------------------------------
    # 系数构建 (全向量化) — v 动量
    # ------------------------------------------------------------------
    def build_v_coefficients(self, u: np.ndarray, v: np.ndarray,
                             w: np.ndarray):
        """全向量化 v-动量系数构建.

        v 存储在 y-面心 (nx, ny+1, nz). 内部 v 面: j ∈ [1, ny-1].

        Returns:
            (aE, aW, aN, aS, aT, aB, aP, aP0) 各为 (nx, ny-1, nz) 形状.
        """
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        rho, mu = self.rho, self.mu

        Ax = dy * dz
        Ay = dx * dz
        Az = dx * dy

        v_P = v[:, 1:-1, :]
        v_N = v[:, 2:,   :]
        v_S = v[:, :-2,  :]

        # v 控制体: 中心 v[i,j,k] (j∈[1,ny-1])
        #   x∈[(i+0.5-0.5)dx,(i+0.5+0.5)dx]=[i*dx,(i+1)dx]
        #   y∈[(j-0.5)dy,(j+0.5)dy]
        #   z∈[k*dz,(k+1)dz]
        # 北面 y=(j+0.5)dy: v_n = 0.5*(v[j]+v[j+1]) ✓
        vn = 0.5 * (v_P + v_N)
        vs = 0.5 * (v_S + v_P)

        # u 在 v 控制体东西界面的插值
        # u[i,j,k] 在 (x=i*dx, y=(j+0.5)dy, z=(k+0.5)dz)
        # v 控制体东面 x=(i+1)dx: u_e = 0.5*(u[i+1,j-1]+u[i+1,j])
        ue = 0.5 * (u[1:, 0:-1, :] + u[1:, 1:, :])    # 东面 u (i+1)
        uw = 0.5 * (u[0:-1, 0:-1, :] + u[0:-1, 1:, :])  # 西面 u (i)

        # w 在 v 控制体上下界面
        wt = 0.5 * (w[:, 0:-1, 1:] + w[:, 1:, 1:])
        wb = 0.5 * (w[:, 0:-1, 0:-1] + w[:, 1:, 0:-1])

        Fe = rho * ue * Ax
        Fw = rho * uw * Ax
        Fn = rho * vn * Ay
        Fs = rho * vs * Ay
        Ft = rho * wt * Az
        Fb = rho * wb * Az

        De = mu * Ax / dx
        Dw = De
        Dn = mu * Ay / dy
        Ds = Dn
        Dt = mu * Az / dz
        Db = Dt

        aE = De + np.maximum(-Fe, 0.0)
        aW = Dw + np.maximum( Fw, 0.0)
        aN = Dn + np.maximum(-Fn, 0.0)
        aS = Ds + np.maximum( Fs, 0.0)
        aT = Dt + np.maximum(-Ft, 0.0)
        aB = Db + np.maximum( Fb, 0.0)

        dV = dx * dy * dz
        aP0 = rho * dV / self.dt
        aP = aE + aW + aN + aS + aT + aB + aP0

        return aE, aW, aN, aS, aT, aB, aP, aP0

    # ------------------------------------------------------------------
    # 系数构建 (全向量化) — w 动量
    # ------------------------------------------------------------------
    def build_w_coefficients(self, u: np.ndarray, v: np.ndarray,
                             w: np.ndarray):
        """全向量化 w-动量系数构建.

        w 存储在 z-面心 (nx, ny, nz+1). 内部 w 面: k ∈ [1, nz-1].

        Returns:
            (aE, aW, aN, aS, aT, aB, aP, aP0) 各为 (nx, ny, nz-1) 形状.
        """
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        rho, mu = self.rho, self.mu

        Ax = dy * dz
        Ay = dx * dz
        Az = dx * dy

        w_P = w[:, :, 1:-1]
        w_T = w[:, :, 2:]
        w_B = w[:, :, :-2]

        wt = 0.5 * (w_P + w_T)
        wb = 0.5 * (w_B + w_P)

        # u 在 w 控制体东西界面
        ue = 0.5 * (u[1:, :, 0:-1] + u[1:, :, 1:])
        uw = 0.5 * (u[0:-1, :, 0:-1] + u[0:-1, :, 1:])

        # v 在 w 控制体南北界面
        vn = 0.5 * (v[:, 1:, 0:-1] + v[:, 1:, 1:])
        vs = 0.5 * (v[:, 0:-1, 0:-1] + v[:, 0:-1, 1:])

        Fe = rho * ue * Ax
        Fw = rho * uw * Ax
        Fn = rho * vn * Ay
        Fs = rho * vs * Ay
        Ft = rho * wt * Az
        Fb = rho * wb * Az

        De = mu * Ax / dx
        Dw = De
        Dn = mu * Ay / dy
        Ds = Dn
        Dt = mu * Az / dz
        Db = Dt

        aE = De + np.maximum(-Fe, 0.0)
        aW = Dw + np.maximum( Fw, 0.0)
        aN = Dn + np.maximum(-Fn, 0.0)
        aS = Ds + np.maximum( Fs, 0.0)
        aT = Dt + np.maximum(-Ft, 0.0)
        aB = Db + np.maximum( Fb, 0.0)

        dV = dx * dy * dz
        aP0 = rho * dV / self.dt
        aP = aE + aW + aN + aS + aT + aB + aP0

        return aE, aW, aN, aS, aT, aB, aP, aP0

    # ------------------------------------------------------------------
    # Jacobi 迭代求解 (向量化, Neumann 边界)
    # ------------------------------------------------------------------
    def _jacobi_solve(self, field: np.ndarray,
                      aE, aW, aN, aS, aT, aB, aP, rhs: np.ndarray,
                      n_iter: int, axis: int,
                      force_mask: np.ndarray = None,
                      force_val: np.ndarray = None) -> np.ndarray:
        """通用 Jacobi 迭代求解器 (向量化, 零 Python for 循环在内部点).

        求解 a_P * φ_P = a_E*φ_E + a_W*φ_W + ... + rhs
        内部点沿 `axis` 方向为 [1:-1], 边界采用 Neumann (复制相邻内部值).
        若提供 force_mask, 则掩码点强制为 force_val (IBM 直接强制).

        Args:
            field: 当前场 (含边界). u→(nx+1,ny,nz) axis=0; v→(nx,ny+1,nz) axis=1; w→(nx,ny,nz+1) axis=2.
            aE..aB, aP: 系数 (内部点形状).
            rhs: 右端项 (内部点形状).
            n_iter: Jacobi 迭代次数.
            axis: 内部方向 (0=x for u, 1=y for v, 2=z for w).
            force_mask: (内部点形状) bool, True=该点强制为 force_val.
            force_val: (内部点形状) float, 强制值 (通常 0 for 无滑移).

        Returns:
            更新后的场 (含边界, Neumann).
        """
        f = field.copy()
        for _ in range(n_iter):
            if axis == 0:
                # u: 内部 [1:-1, :, :]
                f_in = f[1:-1, :, :]
                f_E = f[2:,   :, :]
                f_W = f[:-2,  :, :]
                # Neumann: 南北上下边界复制相邻内部
                f_N = np.empty_like(f_in)
                f_N[:, :-1, :] = f[1:-1, 1:, :]
                f_N[:, -1,  :] = f[1:-1, -1, :]
                f_S = np.empty_like(f_in)
                f_S[:, 1:, :] = f[1:-1, :-1, :]
                f_S[:, 0,  :] = f[1:-1, 0, :]
                f_T = np.empty_like(f_in)
                f_T[:, :, :-1] = f[1:-1, :, 1:]
                f_T[:, :, -1]  = f[1:-1, :, -1]
                f_B = np.empty_like(f_in)
                f_B[:, :, 1:] = f[1:-1, :, :-1]
                f_B[:, :, 0]  = f[1:-1, :, 0]
                f_new_in = (aE * f_E + aW * f_W + aN * f_N + aS * f_S +
                            aT * f_T + aB * f_B + rhs) / aP
                if force_mask is not None:
                    f_new_in = np.where(force_mask, force_val, f_new_in)
                f[1:-1, :, :] = f_new_in
                # Neumann 边界 (x 方向两端)
                f[0,  :, :] = f[1,  :, :]
                f[-1, :, :] = f[-2, :, :]
            elif axis == 1:
                # v: 内部 [:, 1:-1, :]
                f_in = f[:, 1:-1, :]
                f_N = f[:, 2:,   :]
                f_S = f[:, :-2,  :]
                f_E = np.empty_like(f_in)
                f_E[:-1, :, :] = f[1:, 1:-1, :]
                f_E[-1,  :, :] = f[-1, 1:-1, :]
                f_W = np.empty_like(f_in)
                f_W[1:, :, :] = f[:-1, 1:-1, :]
                f_W[0,  :, :] = f[0, 1:-1, :]
                f_T = np.empty_like(f_in)
                f_T[:, :, :-1] = f[:, 1:-1, 1:]
                f_T[:, :, -1]  = f[:, 1:-1, -1]
                f_B = np.empty_like(f_in)
                f_B[:, :, 1:] = f[:, 1:-1, :-1]
                f_B[:, :, 0]  = f[:, 1:-1, 0]
                f_new_in = (aE * f_E + aW * f_W + aN * f_N + aS * f_S +
                            aT * f_T + aB * f_B + rhs) / aP
                if force_mask is not None:
                    f_new_in = np.where(force_mask, force_val, f_new_in)
                f[:, 1:-1, :] = f_new_in
                f[:, 0,  :] = f[:, 1,  :]
                f[:, -1, :] = f[:, -2, :]
            else:
                # w: 内部 [:, :, 1:-1]
                f_in = f[:, :, 1:-1]
                f_T = f[:, :, 2:]
                f_B = f[:, :, :-2]
                f_E = np.empty_like(f_in)
                f_E[:-1, :, :] = f[1:, :, 1:-1]
                f_E[-1,  :, :] = f[-1, :, 1:-1]
                f_W = np.empty_like(f_in)
                f_W[1:, :, :] = f[:-1, :, 1:-1]
                f_W[0,  :, :] = f[0, :, 1:-1]
                f_N = np.empty_like(f_in)
                f_N[:, :-1, :] = f[:, 1:, 1:-1]
                f_N[:, -1,  :] = f[:, -1, 1:-1]
                f_S = np.empty_like(f_in)
                f_S[:, 1:, :] = f[:, :-1, 1:-1]
                f_S[:, 0,  :] = f[:, 0, 1:-1]
                f_new_in = (aE * f_E + aW * f_W + aN * f_N + aS * f_S +
                            aT * f_T + aB * f_B + rhs) / aP
                if force_mask is not None:
                    f_new_in = np.where(force_mask, force_val, f_new_in)
                f[:, :, 1:-1] = f_new_in
                f[:, :, 0]  = f[:, :, 1]
                f[:, :, -1] = f[:, :, -2]
        return f

    def solve_u(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                p: np.ndarray, f_ibm_x: np.ndarray,
                n_iter: int = 20,
                ibm_mask: np.ndarray = None) -> np.ndarray:
        """求解 u-动量方程 (Jacobi 迭代 + 欠松弛 + IBM 直接强制).

        a_P * u_P = a_E*u_E + a_W*u_W + ... - (dp/dx)*ΔV + f_ibm*ΔV + a_P0*u_old
        IBM 直接强制: ibm_mask 为 True 的 u 面强制为 0 (无滑移).

        Args:
            u, v, w, p: 当前速度场与压力.
            f_ibm_x: (nx, ny, nz) IBM 体积力 x 分量 (体素中心).
            n_iter: Jacobi 迭代次数.
            ibm_mask: (nx, ny, nz) bool, True=固体体素. u 面掩码由相邻体素 OR 得到.

        Returns:
            更新后的 u (nx+1, ny, nz).
        """
        aE, aW, aN, aS, aT, aB, aP, aP0 = self.build_u_coefficients(u, v, w)
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        dV = dx * dy * dz

        # 压力梯度源项 (体素中心 → u 面心): dp/dx 在 u 面 = (p[i] - p[i-1])/dx
        dpdx = (p[1:, :, :] - p[:-1, :, :]) / dx

        # IBM 体积力源项 (体素中心 → u 面心平均)
        f_src = 0.5 * (f_ibm_x[1:, :, :] + f_ibm_x[:-1, :, :])

        # 伪瞬态源: a_P0 * u_old
        u_old = u[1:-1, :, :].copy()

        # RHS = -dpdx*dV + f_src*dV + aP0*u_old
        rhs = -dpdx * dV + f_src * dV + aP0 * u_old

        # IBM 直接强制掩码 (u 面心): 相邻两体素任一为固体 → 强制 u=0
        force_mask = None
        force_val = None
        if ibm_mask is not None:
            # u 内部面 i∈[1,nx-1] 对应体素 i-1, i
            face_mask = ibm_mask[:-1, :, :] | ibm_mask[1:, :, :]
            force_mask = face_mask
            force_val = np.zeros_like(face_mask, dtype=np.float64)

        u_new = self._jacobi_solve(u, aE, aW, aN, aS, aT, aB, aP, rhs,
                                   n_iter, axis=0,
                                   force_mask=force_mask, force_val=force_val)

        # 欠松弛
        return u + self.alpha_u * (u_new - u)

    def solve_v(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                p: np.ndarray, f_ibm_y: np.ndarray,
                n_iter: int = 20,
                ibm_mask: np.ndarray = None) -> np.ndarray:
        """求解 v-动量方程 (Jacobi 迭代 + 欠松弛 + IBM 直接强制)."""
        aE, aW, aN, aS, aT, aB, aP, aP0 = self.build_v_coefficients(u, v, w)
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        dV = dx * dy * dz

        dpdy = (p[:, 1:, :] - p[:, :-1, :]) / dy
        f_src = 0.5 * (f_ibm_y[:, 1:, :] + f_ibm_y[:, :-1, :])
        v_old = v[:, 1:-1, :].copy()
        rhs = -dpdy * dV + f_src * dV + aP0 * v_old

        # IBM 直接强制掩码 (v 面心): 相邻两体素任一为固体 → 强制 v=0
        force_mask = None
        force_val = None
        if ibm_mask is not None:
            face_mask = ibm_mask[:, :-1, :] | ibm_mask[:, 1:, :]
            force_mask = face_mask
            force_val = np.zeros_like(face_mask, dtype=np.float64)

        v_new = self._jacobi_solve(v, aE, aW, aN, aS, aT, aB, aP, rhs,
                                   n_iter, axis=1,
                                   force_mask=force_mask, force_val=force_val)
        return v + self.alpha_u * (v_new - v)

    def solve_w(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                p: np.ndarray, f_ibm_z: np.ndarray,
                n_iter: int = 20,
                ibm_mask: np.ndarray = None) -> np.ndarray:
        """求解 w-动量方程 (Jacobi 迭代 + 欠松弛 + IBM 直接强制)."""
        aE, aW, aN, aS, aT, aB, aP, aP0 = self.build_w_coefficients(u, v, w)
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        dV = dx * dy * dz

        dpdz = (p[:, :, 1:] - p[:, :, :-1]) / dz
        f_src = 0.5 * (f_ibm_z[:, :, 1:] + f_ibm_z[:, :, :-1])
        w_old = w[:, :, 1:-1].copy()
        rhs = -dpdz * dV + f_src * dV + aP0 * w_old

        # IBM 直接强制掩码 (w 面心): 相邻两体素任一为固体 → 强制 w=0
        force_mask = None
        force_val = None
        if ibm_mask is not None:
            face_mask = ibm_mask[:, :, :-1] | ibm_mask[:, :, 1:]
            force_mask = face_mask
            force_val = np.zeros_like(face_mask, dtype=np.float64)

        w_new = self._jacobi_solve(w, aE, aW, aN, aS, aT, aB, aP, rhs,
                                   n_iter, axis=2,
                                   force_mask=force_mask, force_val=force_val)
        return w + self.alpha_u * (w_new - w)
