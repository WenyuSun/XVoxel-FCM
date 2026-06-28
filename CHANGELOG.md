# Changelog

All notable changes to XVoxel-FCM will be documented in this file.

## [Unreleased]

### Added
- Git repository initialization with `.gitignore`.
- `README.md` with installation instructions and quick-start examples.
- `LICENSE` (MIT).
- `doc/` folder for design documents and review reports.
- XVoxel-FCM Codex skills: `xvoxel-python-style`, `xvoxel-fem-solver`,
  `xvoxel-paper-reproduction`, `xvoxel-testing`, `xvoxel-versioning`,
  `xvoxel-task-workflow`.
- `fluid/` package: FVM + IBM incompressible steady-flow solver
  (MAC staggered grid + SIMPLE algorithm + source-term immersed boundary
  method). Consumes `xvoxel/` geometry read-only.
  - `fluid/staggered_grid.py`: MAC staggered grid container with
    `from_xvoxel` factory and `divergence()`.
  - `fluid/ibm.py`: Immersed boundary force computation (source-term,
    penalty=1, SDF-based weights).
  - `fluid/boundary.py`: Vectorized boundary conditions (inlet/outlet/
    slip-wall/no-slip-wall).
  - `fluid/momentum.py`: FVM momentum discretization (upwind convection
    + central diffusion) with Jacobi linear solver.
  - `fluid/pressure.py`: Pressure Poisson solver (red-black SOR,
    omega=1.7).
  - `fluid/simple_solver.py`: SIMPLE algorithm main loop
    (pseudo-transient steady-state).
  - `fluid/solver.py`: `FluidSolver` facade with declarative API
    aligned to `FCMSolver`.
  - `fluid/postprocess.py`: Drag/lift coefficient computation.
- `tests/test_fluid.py`: 15 tests (13 fast + 2 `@slow` cylinder
  cross-flow validation). All pass.
- `examples/cylinder_flow_validation.py`: Level 0 (channel Poiseuille)
  + Level 1 (cylinder Re=20, Re=40) validation script with plotting.
- `doc/FLUID_MVP_REPORT.md`: FVM+IBM MVP validation report.
- `figures/channel_flow_validation.png`,
  `figures/cylinder_flow_Re40.png`, `figures/cylinder_flow_Re20.png`:
  validation figures.

### Changed
- Removed `trimesh` from `requirements.txt` (unused dependency).
- Moved design/review documents into `doc/`.
- `tests/conftest.py`: registered custom `slow` pytest mark.

### Removed
- StellaCAD-specific skills and workflows no longer applicable to this repo.

## [0.1.0] — 2025-06

### Added
- Initial prototype: XVoxel data structure, CSG feature management, PMC,
  FCM solver (Hex8), L-shape and connector benchmark examples.
