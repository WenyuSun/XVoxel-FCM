# -*- coding: utf-8 -*-
"""profile_assembly.py — 分析新版 fcm/assembly.py 的装配耗时瓶颈"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NX, NY, NZ = 15, 15, 3
LX, LY, LZ = 15.0, 15.0, 3.0
ORIGIN = (0.0, 0.0, -LZ/2)
E = 2e5; NU = 0.3; ALPHA = 1e-8; MAX_DEPTH = 4

from xvoxel import XVoxelModel, Cube, RoundCorner2D
from fcm import FCMSolver
from fcm.mesh import UniformHexMesh
from fcm.assembly import (assemble_fcm_k, classify_elements, _add_ke_to_lil,
                           _get_elem_dofs, assemble_boundary_ke, _octree_integrate,
                           _ref_to_phys, _classify_single)
from fcm.elements import get_element_info

xv = XVoxelModel(NX, NY, NZ, LX, LY, LZ, origin=ORIGIN)
vert = Cube(cx=1.5, cy=7.5, cz=0.0, sx=3.0, sy=15.0, sz=3.0, name="vert")
horiz = Cube(cx=7.5, cy=1.5, cz=0.0, sx=15.0, sy=3.0, sz=3.0, name="horiz")
corner = RoundCorner2D(cx=3.0, cy=3.0, r=6.0, zmin=-LZ/2, zmax=LZ/2, sign_x=+1, sign_y=+1, name="corner")
xv.add_feature(vert); xv.add_feature(horiz); xv.add_feature(corner)

solver = FCMSolver(xv, order=1)
solver.set_material(E, NU, ALPHA)
mesh = solver.mesh
csg_root = xv.csg_root
voxel_nature = xv.voxel_nature
info = get_element_info(1)
npe = info['npe']; ndof_per_elem = info['ndof_per_elem']
ke_func = info['ke_func']; gauss_rule = info['gauss_rule']
solid_eids, void_eids, cut_eids = classify_elements(voxel_nature)

print(f"Solids: {len(solid_eids)}, Voids: {len(void_eids)}, Cut: {len(cut_eids)}")

# ---- Profile: _add_ke_to_lil ----
from scipy.sparse import lil_matrix, coo_matrix
ndof = mesh.ndof
K = lil_matrix((ndof, ndof), dtype=np.float64)

# Pre-build a sample ke
eid0 = solid_eids[0]
coords0 = mesh.get_elem_coords(eid0)
ke0 = ke_func(coords0, E, NU)
dofs0 = _get_elem_dofs(mesh.elems[eid0])

# Time: _get_elem_dofs (called 675 times)
t0 = time.perf_counter()
for _ in range(675):
    _get_elem_dofs(mesh.elems[0])
t_getdofs = time.perf_counter() - t0
print(f"\n_get_elem_dofs × 675: {t_getdofs*1000:.1f} ms")

# Time: _add_ke_to_lil (called 675 times)
t0 = time.perf_counter()
for _ in range(675):
    _add_ke_to_lil(K, ke0, dofs0, ndof_per_elem)
t_add_lil = time.perf_counter() - t0
print(f"_add_ke_to_lil × 675: {t_add_lil*1000:.1f} ms  ← 主瓶颈!")

# Time: ke_func per element
t0 = time.perf_counter()
for i in range(min(258, len(solid_eids))):
    eid = solid_eids[i]
    ke_func(mesh.get_elem_coords(eid), E, NU)
t_ke = time.perf_counter() - t0
print(f"ke_func × 258 solid: {t_ke*1000:.1f} ms")

# Compare: COO-style batched add
rows, cols, data = [], [], []
t0 = time.perf_counter()
for _ in range(675):
    for a in range(ndof_per_elem):
        for b in range(ndof_per_elem):
            rows.append(dofs0[a])
            cols.append(dofs0[b])
            data.append(ke0[a, b])
coo = coo_matrix((data, (rows, cols)), shape=(ndof, ndof))
t_coo_batch = time.perf_counter() - t0
print(f"COO batch (collect then build) × 675: {t_coo_batch*1000:.1f} ms")

# Time: Octree integration per cut element
if len(cut_eids) > 0:
    eid_cut = cut_eids[0]
    coords_cut = mesh.get_elem_coords(eid_cut)
    t0 = time.perf_counter()
    for _ in range(9):
        ke = assemble_boundary_ke(coords_cut, csg_root, E, NU, ALPHA, 1, gauss_rule, MAX_DEPTH)
    t_cut = time.perf_counter() - t0
    print(f"assemble_boundary_ke × 9 (cut elements): {t_cut*1000:.1f} ms")
    print(f"  per cut element: {t_cut/9*1000:.1f} ms")

# Time: sdf_batch overhead (called very frequently in octree)
t0 = time.perf_counter()
pt = np.array([[5.0, 7.5, 0.0]])
for _ in range(5000):
    csg_root.sdf_batch(pt)
t_sdf = time.perf_counter() - t0
print(f"sdf_batch × 5000: {t_sdf*1000:.1f} ms ({t_sdf/5000*1e6:.1f} μs/call)")

print(f"\n--- Bottleneck Summary ---")
print(f"lil add (675 elems):  {t_add_lil*1000:.0f} ms  ← 最大瓶颈")
print(f"ke_func (258 solid):  {t_ke*1000:.0f} ms")
print(f"cut elements (9):     {t_cut*1000:.0f} ms")
print(f"void elements (408):  ~{t_add_lil*408/675*1000:.0f} ms (lil add overhead)")
print(f"\n预期改进: lil→coo batch 可减少 ~{t_add_lil*1000:.0f}ms")
