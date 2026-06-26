# -*- coding: utf-8 -*-
"""
generate_fig7_baseline.py — 生成 Fig 7 L-shape 黄金基准 (golden baseline)

用途:
    运行新版 (xvoxel/ + fcm/) Fig 7 样例, 将每个 R 的关键数值结果
    (体素分类、最大位移、最大 von Mises 应力、位移范数) 写入
    tests/baselines/fig7_golden.json, 作为回归测试的基准.

何时重跑:
    - 仅当算法逻辑发生**有意**变更 (如修正已知 bug、升级积分阶次) 时重跑.
    - 纯重构、向量化、性能优化**不得**重跑 — 应通过回归测试验证基准不变.

运行:
    python tests/baselines/generate_fig7_baseline.py
"""
import json
import os
import sys
import time

import numpy as np

# Repo root on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 统一参数 (与 examples/compare_fig7.py 保持一致)
NX, NY, NZ = 15, 15, 3
LX, LY, LZ = 15.0, 15.0, 3.0
ORIGIN = (0.0, 0.0, -LZ / 2)
E = 2e5
NU = 0.3
ALPHA = 1e-8
ORDER = 1  # Hex8
MAX_DEPTH = 4
TRACTION = (0.0, -100.0, 0.0)  # xmax downward
RADII = [6.0, 5.0, 4.0, 3.0, 2.0]


def run_fig7_new() -> list:
    """运行新版 Fig 7, 返回每个 R 的关键结果列表."""
    from xvoxel import XVoxelModel, Cube, RoundCorner2D
    from fcm import FCMSolver
    from fcm.boundary import apply_dirichlet
    from scipy.sparse.linalg import spsolve

    results = []
    for step, r in enumerate(RADII):
        if step == 0:
            xv = XVoxelModel(NX, NY, NZ, LX, LY, LZ, origin=ORIGIN)
            vert = Cube(cx=1.5, cy=7.5, cz=0.0, sx=3.0, sy=15.0, sz=3.0, name="vert")
            horiz = Cube(cx=7.5, cy=1.5, cz=0.0, sx=15.0, sy=3.0, sz=3.0, name="horiz")
            corner = RoundCorner2D(cx=3.0, cy=3.0, r=r,
                                   zmin=-LZ / 2, zmax=LZ / 2,
                                   sign_x=+1, sign_y=+1, name="corner")
            xv.add_feature(vert)
            xv.add_feature(horiz)
            corner_fid = xv.add_feature(corner)
        else:
            xv.edit_parameter(corner_fid, 'r', r)

        solver = FCMSolver(xv, order=ORDER)
        solver.set_material(E, NU, ALPHA)
        solver.add_dirichlet_bc('ymax', 'ux,uy,uz', 0.0)

        K = solver.assemble(alpha=ALPHA, max_depth=MAX_DEPTH)

        # Traction (与 compare_fig7.py 共享手动计算)
        from examples.compare_fig7 import compute_traction_force
        csg = xv.csg_root

        def _pmc_new(x, y, z, eid):
            sdf = float(csg.sdf_batch(np.array([[x, y, z]]))[0])
            if sdf > 0:
                return -1
            elif sdf == 0:
                return 0
            else:
                return 1

        F = compute_traction_force(
            solver.mesh.get_elem_coords, solver.mesh.elems, solver.mesh.ndof,
            xv.n_voxels, ORIGIN, LX, TRACTION, _pmc_new,
        )

        if solver.fixed_dofs:
            all_fixed = np.concatenate(solver.fixed_dofs)
            all_vals = np.concatenate(solver.prescribed_vals)
            K, F = apply_dirichlet(K, F, all_fixed, all_vals)

        u = spsolve(K.tocsr() if hasattr(K, 'tocsr') else K, F)

        # Von Mises (中心 Gauss 点, 与 compare_fig7.py 一致)
        from src.fem_base import hex8_shape_grad
        mesh = solver.mesh
        c_mat = E / ((1 + NU) * (1 - 2 * NU))
        D = np.array([
            [1 - NU, NU, NU, 0, 0, 0],
            [NU, 1 - NU, NU, 0, 0, 0],
            [NU, NU, 1 - NU, 0, 0, 0],
            [0, 0, 0, (1 - 2 * NU) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * NU) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * NU) / 2],
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
                B[3, col] = dN_dx[1, a]
                B[3, col + 1] = dN_dx[0, a]
                B[4, col + 1] = dN_dx[2, a]
                B[4, col + 2] = dN_dx[1, a]
                B[5, col] = dN_dx[2, a]
                B[5, col + 2] = dN_dx[0, a]
            strain = B @ u_e
            stress = D @ strain
            sxx, syy, szz, sxy, syz, sxz = stress
            vm[eid] = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2
                                     + (szz - sxx) ** 2
                                     + 6 * (sxy ** 2 + syz ** 2 + sxz ** 2)))

        max_u = float(np.max(np.abs(u)))
        max_vm = float(np.max(vm[xv.voxel_nature != -1])) if np.any(xv.voxel_nature != -1) else 0.0
        u_norm = float(np.linalg.norm(u))

        results.append({
            'radius': r,
            'voxel_nature': xv.voxel_nature.astype(int).tolist(),
            'solid': int(np.sum(xv.voxel_nature == 1)),
            'boundary': int(np.sum(xv.voxel_nature == 0)),
            'void': int(np.sum(xv.voxel_nature == -1)),
            'max_u': max_u,
            'max_vm': max_vm,
            'u_norm': u_norm,
        })
        print(f"  R={r:.1f}: solid={results[-1]['solid']} bnd={results[-1]['boundary']} "
              f"void={results[-1]['void']} | max|u|={max_u:.6f} max_vm={max_vm:.2f}")

    return results


def main() -> None:
    """生成黄金基准并写入 JSON."""
    print("=" * 70)
    print("  Generating Fig 7 golden baseline (new version)")
    print("=" * 70)
    t0 = time.time()
    results = run_fig7_new()
    elapsed = time.time() - t0

    baseline = {
        'description': 'Fig 7 L-shape golden baseline (new xvoxel/ + fcm/ version)',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'NX': NX, 'NY': NY, 'NZ': NZ,
            'LX': LX, 'LY': LY, 'LZ': LZ,
            'E': E, 'NU': NU, 'ALPHA': ALPHA,
            'ORDER': ORDER, 'MAX_DEPTH': MAX_DEPTH,
            'TRACTION': list(TRACTION),
            'RADII': RADII,
        },
        'elapsed_seconds': elapsed,
        'results': results,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fig7_golden.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"\nGolden baseline written to: {out_path}")
    print(f"Total elapsed: {elapsed:.3f}s")


if __name__ == '__main__':
    main()
