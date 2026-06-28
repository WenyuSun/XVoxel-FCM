# -*- coding: utf-8 -*-
"""
test_regression_fig7.py — Fig 7 L-shape 数值保真回归测试

原则 P1 (数值保真红线):
    重构/向量化不得改变任何算法逻辑, 仿真结果必须与黄金基准一致.
    本测试是自动化门禁 — 任何对 fcm/assembly.py、xvoxel/、fcm/ 的修改
    若使结果超出容差, 本测试将红灯, 阻止合并.

基准文件: tests/baselines/fig7_golden.json
容差配置: tests/baselines/tolerance.yaml

重跑基准 (仅算法有意变更时):
    python tests/baselines/generate_fig7_baseline.py
"""
import json
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 加载黄金基准与容差
# ---------------------------------------------------------------------------
_BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'baselines')
_GOLDEN_PATH = os.path.join(_BASELINE_DIR, 'fig7_golden.json')

if not os.path.exists(_GOLDEN_PATH):
    pytest.skip(
        f"Golden baseline not found at {_GOLDEN_PATH}. "
        f"Run `python tests/baselines/generate_fig7_baseline.py` first.",
        allow_module_level=True,
    )

with open(_GOLDEN_PATH, encoding='utf-8') as _f:
    GOLDEN = json.load(_f)

# 容差 (硬编码 fallback, 优先读 tolerance.yaml)
_DISP_ABS = 1.0e-9
_DISP_REL_PCT = 0.001
_STRESS_ABS = 1.0
_STRESS_REL_PCT = 0.5
_U_NORM_REL_PCT = 0.001

try:
    import yaml
    with open(os.path.join(_BASELINE_DIR, 'tolerance.yaml'), encoding='utf-8') as _f:
        _tol = yaml.safe_load(_f)
    _DISP_ABS = _tol['displacement']['abs_tol']
    _DISP_REL_PCT = _tol['displacement']['rel_tol_pct']
    _STRESS_ABS = _tol['stress']['abs_tol']
    _STRESS_REL_PCT = _tol['stress']['rel_tol_pct']
    _U_NORM_REL_PCT = _tol['u_norm']['rel_tol_pct']
except Exception:  # noqa: BLE001 — yaml 缺失时用默认容差
    pass


# ---------------------------------------------------------------------------
# 运行新版 Fig 7 (与 generate_fig7_baseline.py 相同逻辑)
# ---------------------------------------------------------------------------
def _run_fig7_new():
    """运行新版 Fig 7 全部 5 个 R, 返回结果列表."""
    import sys
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    from xvoxel import XVoxelModel, Cube, RoundCorner2D
    from fcm import FCMSolver
    from fcm.boundary import apply_dirichlet
    from fcm.elements import hex8_shape_grad
    from scipy.sparse.linalg import spsolve
    from tests.baselines._fig7_traction import compute_traction_force

    cfg = GOLDEN['config']
    radii = cfg['RADII']
    results = []

    for step, r in enumerate(radii):
        if step == 0:
            xv = XVoxelModel(cfg['NX'], cfg['NY'], cfg['NZ'],
                             cfg['LX'], cfg['LY'], cfg['LZ'],
                             origin=(0.0, 0.0, -cfg['LZ'] / 2))
            xv.add_feature(Cube(cx=1.5, cy=7.5, cz=0.0, sx=3.0, sy=15.0, sz=3.0, name="vert"))
            xv.add_feature(Cube(cx=7.5, cy=1.5, cz=0.0, sx=15.0, sy=3.0, sz=3.0, name="horiz"))
            corner = RoundCorner2D(cx=3.0, cy=3.0, r=r,
                                   zmin=-cfg['LZ'] / 2, zmax=cfg['LZ'] / 2,
                                   sign_x=+1, sign_y=+1, name="corner")
            xv.add_feature(corner)
            corner_fid = 2
        else:
            xv.edit_parameter(corner_fid, 'r', r)

        solver = FCMSolver(xv, order=cfg['ORDER'])
        solver.set_material(cfg['E'], cfg['NU'], cfg['ALPHA'])
        solver.add_dirichlet_bc('ymax', 'ux,uy,uz', 0.0)
        K = solver.assemble(alpha=cfg['ALPHA'], max_depth=cfg['MAX_DEPTH'])

        csg = xv.csg_root

        def _pmc(x, y, z, eid):
            sdf = float(csg.sdf_batch(np.array([[x, y, z]]))[0])
            return -1 if sdf > 0 else (0 if sdf == 0 else 1)

        F = compute_traction_force(
            solver.mesh.get_elem_coords, solver.mesh.elems, solver.mesh.ndof,
            xv.n_voxels, (0.0, 0.0, -cfg['LZ'] / 2), cfg['LX'],
            tuple(cfg['TRACTION']), _pmc,
        )

        if solver.fixed_dofs:
            K, F = apply_dirichlet(K, F, np.concatenate(solver.fixed_dofs),
                                   np.concatenate(solver.prescribed_vals))

        u = spsolve(K.tocsr() if hasattr(K, 'tocsr') else K, F)

        # Von Mises (中心 Gauss 点)
        mesh = solver.mesh
        c_mat = cfg['E'] / ((1 + cfg['NU']) * (1 - 2 * cfg['NU']))
        D = np.array([
            [1 - cfg['NU'], cfg['NU'], cfg['NU'], 0, 0, 0],
            [cfg['NU'], 1 - cfg['NU'], cfg['NU'], 0, 0, 0],
            [cfg['NU'], cfg['NU'], 1 - cfg['NU'], 0, 0, 0],
            [0, 0, 0, (1 - 2 * cfg['NU']) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * cfg['NU']) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * cfg['NU']) / 2],
        ]) * c_mat
        vm = np.zeros(xv.n_voxels)
        for eid in range(xv.n_voxels):
            if xv.voxel_nature[eid] == -1:
                continue
            coords = mesh.get_elem_coords(eid)
            elem_nodes = mesh.elems[eid]
            u_e = np.zeros(24)
            for a in range(8):
                for d in range(3):
                    u_e[a * 3 + d] = u[elem_nodes[a] * 3 + d]
            dN = hex8_shape_grad(0, 0, 0)
            J = dN @ coords
            dN_dx = np.linalg.inv(J) @ dN
            B = np.zeros((6, 24))
            for a in range(8):
                col = a * 3
                B[0, col] = dN_dx[0, a]
                B[1, col + 1] = dN_dx[1, a]
                B[2, col + 2] = dN_dx[2, a]
                B[3, col] = dN_dx[1, a]; B[3, col + 1] = dN_dx[0, a]
                B[4, col + 1] = dN_dx[2, a]; B[4, col + 2] = dN_dx[1, a]
                B[5, col] = dN_dx[2, a]; B[5, col + 2] = dN_dx[0, a]
            strain = B @ u_e
            stress = D @ strain
            sxx, syy, szz, sxy, syz, sxz = stress
            vm[eid] = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2
                                     + (szz - sxx) ** 2
                                     + 6 * (sxy ** 2 + syz ** 2 + sxz ** 2)))

        results.append({
            'radius': r,
            'voxel_nature': xv.voxel_nature.copy(),
            'max_u': float(np.max(np.abs(u))),
            'max_vm': float(np.max(vm[xv.voxel_nature != -1]))
                      if np.any(xv.voxel_nature != -1) else 0.0,
            'u_norm': float(np.linalg.norm(u)),
        })
    return results


# 模块级缓存, 避免重复运行 (Fig 7 全量约 0.3s)
_CACHED_RESULTS = None


def _get_results():
    global _CACHED_RESULTS
    if _CACHED_RESULTS is None:
        _CACHED_RESULTS = _run_fig7_new()
    return _CACHED_RESULTS


# ---------------------------------------------------------------------------
# 回归测试
# ---------------------------------------------------------------------------
class TestFig7Regression:
    """Fig 7 数值保真回归测试 — 原则 P1 的自动化门禁."""

    @pytest.mark.parametrize('idx', range(len(GOLDEN['results'])))
    def test_voxel_classification_identical(self, idx):
        """体素分类必须与基准完全一致 (零容差)."""
        new = _get_results()[idx]
        golden = GOLDEN['results'][idx]
        golden_vn = np.array(golden['voxel_nature'], dtype=int)
        assert np.array_equal(new['voxel_nature'], golden_vn), (
            f"R={golden['radius']}: voxel classification differs — "
            f"{int(np.sum(new['voxel_nature'] != golden_vn))} voxels mismatched. "
            f"重构不得改变体素分类 (原则 P1)."
        )

    @pytest.mark.parametrize('idx', range(len(GOLDEN['results'])))
    def test_max_displacement(self, idx):
        """最大位移必须在容差内."""
        new = _get_results()[idx]
        golden = GOLDEN['results'][idx]
        diff = abs(new['max_u'] - golden['max_u'])
        rel = diff / max(abs(golden['max_u']), 1e-16) * 100
        assert diff <= _DISP_ABS or rel <= _DISP_REL_PCT, (
            f"R={golden['radius']}: max|u| diff={diff:.4e} "
            f"(abs_tol={_DISP_ABS:.1e}, rel={rel:.4f}% > {_DISP_REL_PCT}%). "
            f"重构不得改变位移结果 (原则 P1)."
        )

    @pytest.mark.parametrize('idx', range(len(GOLDEN['results'])))
    def test_max_von_mises(self, idx):
        """最大 von Mises 应力必须在容差内."""
        new = _get_results()[idx]
        golden = GOLDEN['results'][idx]
        diff = abs(new['max_vm'] - golden['max_vm'])
        rel = diff / max(abs(golden['max_vm']), 1e-16) * 100
        assert diff <= _STRESS_ABS or rel <= _STRESS_REL_PCT, (
            f"R={golden['radius']}: max σ_vm diff={diff:.4e} "
            f"(abs_tol={_STRESS_ABS}, rel={rel:.4f}% > {_STRESS_REL_PCT}%). "
            f"重构不得改变应力结果 (原则 P1)."
        )

    @pytest.mark.parametrize('idx', range(len(GOLDEN['results'])))
    def test_displacement_norm(self, idx):
        """位移范数必须在容差内."""
        new = _get_results()[idx]
        golden = GOLDEN['results'][idx]
        rel = abs(new['u_norm'] - golden['u_norm']) / max(abs(golden['u_norm']), 1e-16) * 100
        assert rel <= _U_NORM_REL_PCT, (
            f"R={golden['radius']}: ||u|| rel diff={rel:.4f}% > {_U_NORM_REL_PCT}%. "
            f"重构不得改变位移结果 (原则 P1)."
        )
