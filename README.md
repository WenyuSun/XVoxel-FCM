# XVoxel-FCM

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Reproduction of the JCAD 2024 paper:

> **"XVoxel-Based Parametric Design Optimization of Feature Models"**
> — *Li et al., Journal of Computational and Applied Design, 2024*

XVoxel-FCM implements the **XVoxel extended voxel data structure** and the
**Finite Cell Method (FCM)** for efficient, re-meshing-free parametric
design optimization of feature-based CAD models.

---

## Features

| Module | Description |
|--------|-------------|
| **XVoxel Data Structure** | Extended voxel grid with feature-attribute linked lists for full CSG history tracking |
| **Point Membership Classification (PMC)** | Fast, history-aware material classification respecting feature edit order |
| **Finite Cell Method (FCM)** | High-order embedded-domain FEM on fixed Cartesian grids — no re-meshing |
| **Parametric Feature Editing** | Edit feature parameters (dimensions, positions, radii) with local stiffness updates |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/WenyuSun/XVoxel-FCM.git
cd XVoxel-FCM

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Example 1 — L-Shape Bracket (Paper Fig 7–9)

```bash
python examples/fig7_lshape.py
```

Simulates an L-shaped bracket ($15 \times 15 \times 3$ mm) under downward
traction load, with a 5-step fillet radius editing sequence ($R:6 \rightarrow
5 \rightarrow 4 \rightarrow 3 \rightarrow 2$ mm).

### Example 2 — Mechanical Connector (Paper Fig 10–12)

```bash
python examples/fig10_connector_strict.py
```

Simulates a connecting rod ($55 \times 16 \times 9$ mm) with inner holes and
grooves, edited through a 6-step feature-modification sequence.

---

## Project Structure

```
src/
├── primitives.py      # Geometric primitives (Cube, Cylinder) with SDF
├── csg.py             # CSG operations, Feature & AttribEntry classes
├── xvoxel.py          # XVoxel extended voxel data structure
├── pmc.py             # Point Membership Classification
├── fem_base.py        # Base Hex8 / Hex20 FEM solver
└── fem_xvoxel.py      # XVoxel-aware FCM solver with local update
examples/
├── fig7_lshape.py              # L-shape model (strict reproduction)
├── fig10_connector_strict.py   # Connector model (strict)
└── fig12_connector.py          # Connector model (alternative params)
tests/
└── ...                 # Unit and integration tests
doc/
└── ...                 # Design docs, review reports (local only)
data/
└── ...                 # Serialized simulation results (*.pkl)
output/
└── ...                 # Generated outputs
```

---

## Dependencies

- Python ≥ 3.9
- [NumPy](https://numpy.org/) ≥ 1.22
- [SciPy](https://scipy.org/) ≥ 1.8
- [Matplotlib](https://matplotlib.org/) ≥ 3.5

---

## Testing

```bash
pytest tests/ -v
```

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{li2024xvoxel,
  title   = {XVoxel-Based Parametric Design Optimization of Feature Models},
  author  = {Li, ...},
  journal = {Journal of Computational and Applied Design},
  year    = {2024}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
