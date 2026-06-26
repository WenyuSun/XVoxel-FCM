# -*- coding: utf-8 -*-
"""
compare_fig7.py — 新旧版本 Fig 7 L-shape 执行结果逐项对比

统一参数:
    - 网格 15×15×3
    - Hex8 线性单元
    - 强 Dirichlet BC (ymax 面 ux=uy=uz=0)
    - Traction (xmax 面, downward -100 N/mm²)
    - E=2e5, ν=0.3, α=1e-8
    - 5 步: R=6→5→4→3→2
    - 八叉树 max_depth=3

对比内容:
    1. voxel_nature (体素分类一致)
    2. 最大位移
    3. 最大 von Mises 应力
    4. 各位移分量统计
    5. 耗时
"""
import sys
import os
import time
import numpy as np

# 把 repo root 加入 sys.path (旧版需要)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 统一参数
# ============================================================
NX, NY, NZ = 15, 15, 3
LX, LY, LZ = 15.0, 15.0, 3.0
ORIGIN = (0.0, 0.0, -LZ/2)
E = 2e5
NU = 0.3
ALPHA = 1e-8
ORDER = 1  # Hex8
MAX_DEPTH = 4  # Match old hardcoded value in _octree_integrate
TRACTION = (0.0, -100.0, 0.0)  # xmax downward


# ============================================================
# 共享: 手动 traction 计算 (旧版算法, 新版用 CSG 树等价实现)
# ============================================================
def compute_traction_force(get_coords, elems, ndof, n_voxels, origin, lx, traction, pmc_fn):
    """手动计算 xmax 面牵引力等效节点力 — 新旧共用, 保证完全一致.

    Args:
        get_coords: callable(eid) -> (npe, 3) 单元节点坐标
        elems: (n_elems, npe) 单元-节点表
        ndof: 总自由度数
        n_voxels: 体素总数
        origin: (ox, oy, oz)
        lx: X 方向长度
        traction: (tx, ty, tz)
        pmc_fn: callable(x, y, z, eid) -> int status (-1/0/+1)
    """
    F_face = np.zeros(ndof)
    tx, ty, tz = traction
    for eid in range(n_voxels):
        coords = get_coords(eid)
        if abs(coords[1, 0] - (origin[0] + lx)) > 1e-10:
            continue
        elem_nodes = elems[eid]
        for gp_xi in [-1/np.sqrt(3), 1/np.sqrt(3)]:
            for gp_eta in [-1/np.sqrt(3), 1/np.sqrt(3)]:
                N4 = np.array([
                    0.25*(1-gp_xi)*(1-gp_eta),
                    0.25*(1+gp_xi)*(1-gp_eta),
                    0.25*(1+gp_xi)*(1+gp_eta),
                    0.25*(1-gp_xi)*(1+gp_eta),
                ])
                face_nodes = [1, 2, 6, 5]
                fc = coords[face_nodes]
                dNdxi = np.array([
                    -0.25*(1-gp_eta), 0.25*(1-gp_eta),
                    0.25*(1+gp_eta), -0.25*(1+gp_eta),
                ])
                dNdeta = np.array([
                    -0.25*(1-gp_xi), -0.25*(1+gp_xi),
                    0.25*(1+gp_xi), 0.25*(1-gp_xi),
                ])
                J_face = np.array([
                    [dNdxi @ fc[:, 1], dNdeta @ fc[:, 1]],
                    [dNdxi @ fc[:, 2], dNdeta @ fc[:, 2]],
                ])
                dS = abs(np.linalg.det(J_face))
                gp_xyz = N4 @ fc
                status = pmc_fn(gp_xyz[0], gp_xyz[1], gp_xyz[2], eid)
                if status == -1:
                    continue
                for a, ni in enumerate(face_nodes):
                    nid = elem_nodes[ni]
                    force = 1.0 * N4[a] * dS
                    F_face[nid*3]     += force * tx
                    F_face[nid*3 + 1] += force * ty
                    F_face[nid*3 + 2] += force * tz
    return F_face

# ============================================================
# 旧版运行
# ============================================================
def run_old():
    print("\n" + "=" * 70)
    print("  OLD VERSION (src/)")
    print("=" * 70)
    from src.primitives import Cube, RoundCorner2D
    from src.xvoxel import XVoxelModel
    from src.fem_xvoxel import XVoxelFEMSolver

    results = []
    radii = [6.0, 5.0, 4.0, 3.0, 2.0]
    t_total = time.time()

    for step, r in enumerate(radii):
        t_step = time.time()

        if step == 0:
            # Build initial model
            xv = XVoxelModel(NX, NY, NZ, LX, LY, LZ, origin=ORIGIN)
            vert = Cube(cx=1.5, cy=7.5, cz=0.0, sx=3.0, sy=15.0, sz=3.0)
            horiz = Cube(cx=7.5, cy=1.5, cz=0.0, sx=15.0, sy=3.0, sz=3.0)
            corner = RoundCorner2D(cx=3.0, cy=3.0, r=r,
                                   zmin=-LZ/2, zmax=LZ/2,
                                   sign_x=+1, sign_y=+1)
            xv.add_feature(vert, nature=1, name="vert")
            xv.add_feature(horiz, nature=1, name="horiz")
            corner_fid = xv.add_feature(corner, nature=+1, name="corner")
        else:
            xv.edit_parameter(corner_fid, 'r', r)

        t_voxelize = time.time() - t_step

        # Solver
        solver = XVoxelFEMSolver(xv, E=E, nu=NU, alpha=ALPHA, element_order=ORDER)
        t_mesh = time.time() - t_step - t_voxelize  # approx

        t_asm = time.time()
        K, F = solver.assemble_FCM_system()
        t_asm = time.time() - t_asm

        # Dirichlet BC on ymax — OLD: all face nodes (includes void!)
        fixed_dofs = []
        for nid in range(solver.n_nodes):
            y = solver.nodes[nid, 1]
            if abs(y - (ORIGIN[1] + LY)) < 1e-10:
                fixed_dofs.extend([nid*3, nid*3+1, nid*3+2])
        fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=np.int32)

        # Traction on xmax (shared manual code — 新旧版本完全一致)
        from src.pmc import pmc_point_3d
        F += compute_traction_force(
            solver._get_elem_coords, solver.elems, solver.ndof,
            xv.n_voxels, ORIGIN, LX, TRACTION,
            lambda x, y, z, eid: pmc_point_3d(x, y, z,
                                              xv.voxel_attrs[eid], xv.features),
        )

        K, F = solver.apply_dirichlet(K, F, fixed_dofs)
        t_bc = time.time() - t_asm - (time.time() - t_step - t_voxelize - t_mesh)  # rough

        t_solve = time.time()
        from scipy.sparse.linalg import spsolve
        u = spsolve(K, F)
        t_solve = time.time() - t_solve

        # Von Mises
        t_stress = time.time()
        c = E / ((1+NU)*(1-2*NU))
        D = np.array([
            [1-NU, NU, NU, 0, 0, 0],
            [NU, 1-NU, NU, 0, 0, 0],
            [NU, NU, 1-NU, 0, 0, 0],
            [0, 0, 0, (1-2*NU)/2, 0, 0],
            [0, 0, 0, 0, (1-2*NU)/2, 0],
            [0, 0, 0, 0, 0, (1-2*NU)/2],
        ]) * c
        vm = np.zeros(xv.n_voxels)
        for eid in range(xv.n_voxels):
            if xv.voxel_nature[eid] == -1:
                continue
            coords = solver._get_elem_coords(eid)
            elem_nodes = solver.elems[eid]
            u_e = np.zeros(24)
            for a in range(8):
                for d in range(3):
                    u_e[a*3 + d] = u[elem_nodes[a]*3 + d]
            # center Gauss point
            from src.fem_base import hex8_shape_grad
            dN = hex8_shape_grad(0, 0, 0)
            J = dN @ coords
            dN_dx = np.linalg.inv(J) @ dN
            B = np.zeros((6, 24))
            for a in range(8):
                col = a * 3
                B[0, col]   = dN_dx[0, a]
                B[1, col+1] = dN_dx[1, a]
                B[2, col+2] = dN_dx[2, a]
                B[3, col]   = dN_dx[1, a]
                B[3, col+1] = dN_dx[0, a]
                B[4, col+1] = dN_dx[2, a]
                B[4, col+2] = dN_dx[1, a]
                B[5, col]   = dN_dx[2, a]
                B[5, col+2] = dN_dx[0, a]
            strain = B @ u_e
            stress = D @ strain
            sxx, syy, szz, sxy, syz, sxz = stress
            vm[eid] = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2
                                      + 6*(sxy**2 + syz**2 + sxz**2)))
        t_stress = time.time() - t_stress

        t_step_elapsed = time.time() - t_step
        solid = int(np.sum(xv.voxel_nature == 1))
        bnd = int(np.sum(xv.voxel_nature == 0))
        void = int(np.sum(xv.voxel_nature == -1))
        max_u = np.max(np.abs(u))
        max_vm = np.max(vm[xv.voxel_nature != -1]) if np.any(xv.voxel_nature != -1) else 0

        results.append({
            'radius': r,
            'voxel_nature': xv.voxel_nature.copy(),
            'solid': solid,
            'boundary': bnd,
            'void': void,
            'max_u': max_u,
            'max_vm': max_vm,
            'u_norm': np.linalg.norm(u),
            't_voxelize': t_voxelize,
            't_asm': t_asm,
            't_solve': t_solve,
            't_step': t_step_elapsed,
        })
        print(f"  R={r:.1f}: solid={solid} bnd={bnd} void={void} | "
              f"max|u|={max_u:.6f} max_vm={max_vm:.2f} | "
              f"vox={t_voxelize:.3f}s asm={t_asm:.3f}s solve={t_solve:.3f}s")

    t_total = time.time() - t_total
    return results, t_total


# ============================================================
# 新版运行
# ============================================================
def run_new():
    print("\n" + "=" * 70)
    print("  NEW VERSION (xvoxel/ + fcm/)")
    print("=" * 70)
    from xvoxel import XVoxelModel, Cube, RoundCorner2D
    from fcm import FCMSolver

    results = []
    radii = [6.0, 5.0, 4.0, 3.0, 2.0]
    t_total = time.time()

    for step, r in enumerate(radii):
        t_step = time.time()

        if step == 0:
            xv = XVoxelModel(NX, NY, NZ, LX, LY, LZ, origin=ORIGIN)
            vert = Cube(cx=1.5, cy=7.5, cz=0.0, sx=3.0, sy=15.0, sz=3.0, name="vert")
            horiz = Cube(cx=7.5, cy=1.5, cz=0.0, sx=15.0, sy=3.0, sz=3.0, name="horiz")
            corner = RoundCorner2D(cx=3.0, cy=3.0, r=r,
                                   zmin=-LZ/2, zmax=LZ/2,
                                   sign_x=+1, sign_y=+1, name="corner")
            xv.add_feature(vert)
            xv.add_feature(horiz)
            corner_fid = xv.add_feature(corner)
        else:
            xv.edit_parameter(corner_fid, 'r', r)

        t_voxelize = time.time() - t_step

        solver = FCMSolver(xv, order=ORDER)
        solver.set_material(E, NU, ALPHA)
        solver.add_dirichlet_bc('ymax', 'ux,uy,uz', 0.0)

        t_asm = time.time()
        K = solver.assemble(alpha=ALPHA, max_depth=MAX_DEPTH)
        t_asm = time.time() - t_asm

        # Traction: 使用与旧版相同的共享手动计算 (保证完全一致)
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

        # Apply Dirichlet BC
        if solver.fixed_dofs:
            all_fixed = np.concatenate(solver.fixed_dofs)
            all_vals = np.concatenate(solver.prescribed_vals)
            from fcm.boundary import apply_dirichlet
            K, F = apply_dirichlet(K, F, all_fixed, all_vals)

        t_solve = time.time()
        from scipy.sparse.linalg import spsolve
        u = spsolve(K.tocsr() if hasattr(K, 'tocsr') else K, F)
        t_solve = time.time() - t_solve

        # Von Mises — use same manual code as old for fair comparison
        t_stress = time.time()
        mesh = solver.mesh
        c_mat = E / ((1+NU)*(1-2*NU))
        D = np.array([
            [1-NU, NU, NU, 0, 0, 0],
            [NU, 1-NU, NU, 0, 0, 0],
            [NU, NU, 1-NU, 0, 0, 0],
            [0, 0, 0, (1-2*NU)/2, 0, 0],
            [0, 0, 0, 0, (1-2*NU)/2, 0],
            [0, 0, 0, 0, 0, (1-2*NU)/2],
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
                    u_e[a*3 + d] = u[elem_nodes[a]*3 + d]
            from src.fem_base import hex8_shape_grad
            dN = hex8_shape_grad(0, 0, 0)
            J = dN @ coords
            dN_dx = np.linalg.inv(J) @ dN
            B = np.zeros((6, 24))
            for a in range(8):
                col = a * 3
                B[0, col]   = dN_dx[0, a]
                B[1, col+1] = dN_dx[1, a]
                B[2, col+2] = dN_dx[2, a]
                B[3, col]   = dN_dx[1, a]
                B[3, col+1] = dN_dx[0, a]
                B[4, col+1] = dN_dx[2, a]
                B[4, col+2] = dN_dx[1, a]
                B[5, col]   = dN_dx[2, a]
                B[5, col+2] = dN_dx[0, a]
            strain = B @ u_e
            stress = D @ strain
            sxx, syy, szz, sxy, syz, sxz = stress
            vm[eid] = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2
                                      + 6*(sxy**2 + syz**2 + sxz**2)))
        t_stress = time.time() - t_stress

        t_step_elapsed = time.time() - t_step
        solid = int(np.sum(xv.voxel_nature == 1))
        bnd = int(np.sum(xv.voxel_nature == 0))
        void = int(np.sum(xv.voxel_nature == -1))
        max_u = np.max(np.abs(u))
        max_vm = np.max(vm[xv.voxel_nature != -1]) if np.any(xv.voxel_nature != -1) else 0

        results.append({
            'radius': r,
            'voxel_nature': xv.voxel_nature.copy(),
            'solid': solid,
            'boundary': bnd,
            'void': void,
            'max_u': max_u,
            'max_vm': max_vm,
            'u_norm': np.linalg.norm(u),
            't_voxelize': t_voxelize,
            't_asm': t_asm,
            't_solve': t_solve,
            't_step': t_step_elapsed,
        })
        print(f"  R={r:.1f}: solid={solid} bnd={bnd} void={void} | "
              f"max|u|={max_u:.6f} max_vm={max_vm:.2f} | "
              f"vox={t_voxelize:.3f}s asm={t_asm:.3f}s solve={t_solve:.3f}s")

    t_total = time.time() - t_total
    return results, t_total


# ============================================================
# 对比报告
# ============================================================
def compare(old_results, old_total, new_results, new_total):
    print("\n")
    print("=" * 80)
    print("  COMPARISON REPORT: FIG 7 L-SHAPE — OLD vs NEW")
    print("=" * 80)

    # 1. Voxel classification
    print("\n--- 1. Voxel Classification ---")
    for i, (old, new) in enumerate(zip(old_results, new_results)):
        r = old['radius']
        match = np.array_equal(old['voxel_nature'], new['voxel_nature'])
        mismatch = np.sum(old['voxel_nature'] != new['voxel_nature'])
        status = "✓ IDENTICAL" if match else f"✗ {mismatch} voxels differ"
        print(f"  R={r:.1f}: old(s={old['solid']},b={old['boundary']},v={old['void']}) "
              f"new(s={new['solid']},b={new['boundary']},v={new['void']}) — {status}")

    # 2. Displacement
    print("\n--- 2. Displacement ---")
    print(f"  {'R':>6s}  {'Old max|u|':>14s}  {'New max|u|':>14s}  {'Diff':>12s}  {'Rel diff':>10s}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*10}")
    for old, new in zip(old_results, new_results):
        r = old['radius']
        diff = abs(old['max_u'] - new['max_u'])
        rel = diff / max(abs(old['max_u']), abs(new['max_u']), 1e-16) * 100
        print(f"  {r:6.1f}  {old['max_u']:14.6e}  {new['max_u']:14.6e}  {diff:12.4e}  {rel:9.4f}%")

    old_norms = [f"{r['u_norm']:.6e}" for r in old_results]
    new_norms = [f"{r['u_norm']:.6e}" for r in new_results]
    print(f"\n  Old u norms: {old_norms}")
    print(f"  New u norms: {new_norms}")

    # 3. Von Mises
    print("\n--- 3. Max von Mises Stress ---")
    print(f"  {'R':>6s}  {'Old max σ_vm':>14s}  {'New max σ_vm':>14s}  {'Diff':>12s}  {'Rel diff':>10s}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*10}")
    for old, new in zip(old_results, new_results):
        r = old['radius']
        diff = abs(old['max_vm'] - new['max_vm'])
        rel = diff / max(abs(old['max_vm']), abs(new['max_vm']), 1e-16) * 100
        print(f"  {r:6.1f}  {old['max_vm']:14.4f}  {new['max_vm']:14.4f}  {diff:12.4f}  {rel:9.4f}%")

    # 4. Timing
    print("\n--- 4. Timing ---")
    print(f"  {'R':>6s}  {'Old vox(s)':>10s}  {'New vox(s)':>10s}  {'Old asm(s)':>10s}  {'New asm(s)':>10s}  {'Old solve(s)':>12s}  {'New solve(s)':>12s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}")
    for old, new in zip(old_results, new_results):
        r = old['radius']
        print(f"  {r:6.1f}  {old['t_voxelize']:10.4f}  {new['t_voxelize']:10.4f}  "
              f"{old['t_asm']:10.4f}  {new['t_asm']:10.4f}  "
              f"{old['t_solve']:12.4f}  {new['t_solve']:12.4f}")

    old_total_vox = sum(r['t_voxelize'] for r in old_results)
    new_total_vox = sum(r['t_voxelize'] for r in new_results)
    old_total_asm = sum(r['t_asm'] for r in old_results)
    new_total_asm = sum(r['t_asm'] for r in new_results)
    old_total_solve = sum(r['t_solve'] for r in old_results)
    new_total_solve = sum(r['t_solve'] for r in new_results)

    print(f"\n  Summary:")
    print(f"    Old total: {old_total:.2f}s  (vox={old_total_vox:.3f}s asm={old_total_asm:.3f}s solve={old_total_solve:.3f}s)")
    print(f"    New total: {new_total:.2f}s  (vox={new_total_vox:.3f}s asm={new_total_asm:.3f}s solve={new_total_solve:.3f}s)")
    if old_total > 0:
        speedup = old_total / new_total if new_total > 0 else float('inf')
        print(f"    Speedup: {speedup:.2f}× {'(new faster)' if speedup > 1 else '(old faster)'}")

    # 5. Key difference explanations
    print("\n--- 5. Key Differences Explained ---")
    print("""
  ⚠ BC Handling Difference:
    Old version: Dirichlet BC applied to ALL nodes on ymax face
                 (including void-region nodes → near-singular stiffness)
    New version: Dirichlet BC filtered to NON-VOID element nodes only
                 (fewer DOFs fixed, structurally correct)
    
    This is the BUG B2 fix — old version was fixing nodes that had
    no structural connection, leading to inflated displacements.
    
  ⚠ Voxel Classification:
    Should be IDENTICAL if CSG tree construction matches.
    Any differences indicate a bug in one version's voxelization.
    
  ⚠ Performance Expectation:
    New version's _voxelize_feature uses vectorized sdf_batch →
    should be ~100-250× faster in voxelization phase.
    Assembly and solve should be comparable (both use scipy.sparse).
""")

    print("=" * 80)
    print("  END OF COMPARISON")
    print("=" * 80)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 80)
    print("  FIG 7 COMPARISON: Old (src/) vs New (xvoxel/ + fcm/)")
    print(f"  Grid: {NX}×{NY}×{NZ}, Hex8, Dirichlet BC, max_depth={MAX_DEPTH}")
    print("=" * 80)

    old_results, old_total = run_old()
    new_results, new_total = run_new()
    compare(old_results, old_total, new_results, new_total)
