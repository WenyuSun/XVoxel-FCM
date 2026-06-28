# -*- coding: utf-8 -*-
"""generate_comparison_figures.py — 生成两种 IBM 方法的对比图件.

对比源项 IBM (ibm) 与尖锐界面 IBM (sharp) 的:
    1. Cd 对比 (两 Re + 文献)
    2. 中心线速度 u/U_inf vs x/D (两 Re, 两方法叠加)
    3. 横向速度剖面 u/U_inf vs y/D (Re=40, 三位置, 两方法叠加)
    4. 散度收敛历史对比 (两 Re, 两方法叠加)
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_REPO_ROOT, 'figures')
OUT_DIR = os.path.join(_REPO_ROOT, 'output')

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# 文献 Cd 范围
LIT_CD = {
    20: (1.96, 2.10),
    40: (1.43, 1.60),
}
LIT_CD_POINT = {20: 2.01, 40: 1.48}  # Tritton 实验


def main():
    ibm = json.load(open(os.path.join(OUT_DIR, 'cylinder_analysis_ibm.json'),
                         encoding='utf-8'))
    sharp = json.load(open(os.path.join(OUT_DIR,
                          'cylinder_analysis_sharp.json'), encoding='utf-8'))

    # ------------------------------------------------------------------
    # 图 1: Cd 对比 (两 Re + 文献)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    Re_vals = [20, 40]
    cd_ibm = [ibm[f'Re{r}']['Cd'] for r in Re_vals]
    cd_sharp = [sharp[f'Re{r}']['Cd'] for r in Re_vals]
    ax.plot(Re_vals, cd_ibm, 'bs-', markersize=10, linewidth=2,
            label='Source-term IBM (ibm)')
    ax.plot(Re_vals, cd_sharp, 'r^-', markersize=10, linewidth=2,
            label='Sharp-interface IBM (sharp)')
    # 文献范围
    for r in Re_vals:
        lo, hi = LIT_CD[r]
        ax.plot([r, r], [lo, hi], 'g-', linewidth=4, alpha=0.4,
                label='Literature range' if r == 20 else '')
        ax.plot(r, LIT_CD_POINT[r], 'g*', markersize=14, alpha=0.9,
                label='Tritton exp.' if r == 20 else '')
    ax.set_xlabel('Reynolds number Re')
    ax.set_ylabel(r'Drag coefficient $C_d$')
    ax.set_title('Drag coefficient: source-term vs sharp-interface IBM')
    ax.legend()
    ax.set_xticks(Re_vals)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'compare_cd.png'))
    plt.close(fig)
    print('-> compare_cd.png')

    # ------------------------------------------------------------------
    # 图 2: 中心线速度对比 (两 Re)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for idx, Re in enumerate([40, 20]):
        ax = axes[idx]
        ci = ibm[f'Re{Re}']['centerline']
        cs = sharp[f'Re{Re}']['centerline']
        ax.plot(ci['x_over_D'], ci['u_over_Uinf'], 'b-', linewidth=2,
                label='Source-term IBM')
        ax.plot(cs['x_over_D'], cs['u_over_Uinf'], 'r--', linewidth=2,
                label='Sharp-interface IBM')
        ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
        ax.axvline(0, color='k', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.set_xlabel(r'$x/D$')
        ax.set_ylabel(r'$u/U_\infty$')
        ax.set_title(f'Centerline velocity (Re={Re})')
        ax.legend()
        ax.set_xlim([-2, 8])
    fig.suptitle('Wake centerline velocity: method comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'compare_centerline.png'))
    plt.close(fig)
    print('-> compare_centerline.png')

    # ------------------------------------------------------------------
    # 图 3: 横向速度剖面对比 (Re=40, 三位置)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, xd in enumerate(['1.0', '2.0', '4.0']):
        ax = axes[idx]
        ti = ibm['Re40']['transverse'][f'x/D={xd}']
        ts = sharp['Re40']['transverse'][f'x/D={xd}']
        ax.plot(ti['u_over_Uinf'], ti['y_over_D'], 'b-', linewidth=2,
                label='Source-term IBM')
        ax.plot(ts['u_over_Uinf'], ts['y_over_D'], 'r--', linewidth=2,
                label='Sharp-interface IBM')
        ax.axvline(0, color='k', linewidth=0.5, linestyle=':')
        ax.set_xlabel(r'$u/U_\infty$')
        ax.set_ylabel(r'$y/D$')
        ax.set_title(f'x/D = {xd}')
        ax.legend()
    fig.suptitle('Transverse velocity profiles (Re=40): method comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'compare_transverse.png'))
    plt.close(fig)
    print('-> compare_transverse.png')

    # ------------------------------------------------------------------
    # 图 4: 散度收敛历史对比 (两 Re)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for idx, Re in enumerate([40, 20]):
        ax = axes[idx]
        di = ibm[f'Re{Re}']['div_history']
        ds = sharp[f'Re{Re}']['div_history']
        ax.semilogy(range(len(di)), di, 'b-', linewidth=2,
                    label='Source-term IBM')
        ax.semilogy(range(len(ds)), ds, 'r--', linewidth=2,
                    label='Sharp-interface IBM')
        ax.set_xlabel('SIMPLE outer iteration')
        ax.set_ylabel(r'$\max|\nabla\cdot u|$')
        ax.set_title(f'Divergence convergence (Re={Re})')
        ax.legend()
    fig.suptitle('Divergence convergence: method comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'compare_convergence.png'))
    plt.close(fig)
    print('-> compare_convergence.png')

    print('\nAll comparison figures generated.')


if __name__ == '__main__':
    main()
