# -*- coding: utf-8 -*-
"""
generate_sharp_ibm_schematic.py — 生成尖锐界面 IBM 理论示意图

生成 3 张论文级示意图:
  1. octree_subdivision.png  — 八叉树自适应细分 + 高斯点分类
  2. interface_projection.png — 界面投影 + 法向计算
  3. force_scatter.png        — 界面力散列回体素

约定: 蓝色=流体(void, SDF>0), 红色=固体(solid, SDF<0),
      绿色=界面(Φ=0), 黑点=高斯点, 箭头=法向.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from matplotlib.collections import PatchCollection

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'Microsoft YaHei',
    'axes.unicode_minus': False,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIG_DIR = os.path.join(_REPO_ROOT, 'figures')


def gauss_pts_2x2():
    """2×2 Gauss-Legendre 点 (1D: ±1/√3)."""
    g = 1.0 / np.sqrt(3.0)
    return np.array([-g, g])


def plot_octree_subdivision():
    """图1: 八叉树自适应细分 + 高斯点 SDF 分类 (2D 截面示意)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── 子图1: 原始边界体素 (depth=0) ──
    ax = axes[0]
    ax.set_title('(a) 边界体素 (depth=0)', fontsize=12, fontweight='bold')
    _draw_voxel_with_circle(ax, depth=0, show_gauss=True, show_sdf=True)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # ── 子图2: 一次细分 (depth=1) ──
    ax = axes[1]
    ax.set_title('(b) 八叉树细分 (depth=1)', fontsize=12, fontweight='bold')
    _draw_voxel_with_circle(ax, depth=1, show_gauss=True, show_sdf=True)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # ── 子图3: 三次细分 (depth=3, 叶节点) ──
    ax = axes[2]
    ax.set_title('(c) 叶节点高斯分类 (depth=3)', fontsize=12, fontweight='bold')
    _draw_voxel_with_circle(ax, depth=3, show_gauss=True, show_sdf=True,
                            show_interface=True)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='#4A90D9', alpha=0.3, label='流体 (SDF>0)'),
        mpatches.Patch(facecolor='#D94A4A', alpha=0.3, label='固体 (SDF<0)'),
        plt.Line2D([0], [0], color='#2E8B57', linewidth=2.5, label='界面 Φ=0'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k',
                   markersize=7, label='高斯点'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700',
                   markeredgecolor='k', markersize=8, label='界面高斯点'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(_FIG_DIR, 'octree_subdivision.png')
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


def _draw_voxel_with_circle(ax, depth, show_gauss, show_sdf, show_interface=False):
    """绘制含圆形界面的体素 + 八叉树细分 + 高斯点."""
    # 圆柱截面 (圆心 (0.5, 0.5), 半径 0.35)
    cx, cy, r = 0.5, 0.5, 0.35

    # 背景色: 流体蓝
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='#4A90D9', alpha=0.15,
                           edgecolor='k', linewidth=1.5))

    # 绘制细分网格线
    n_sub = 2 ** depth
    h = 1.0 / n_sub
    for i in range(1, n_sub):
        ax.axhline(i * h, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(i * h, color='gray', linewidth=0.5, alpha=0.5)

    # 绘制圆 (固体)
    circle = Circle((cx, cy), r, facecolor='#D94A4A', alpha=0.25,
                    edgecolor='#2E8B57', linewidth=2.5)
    ax.add_patch(circle)

    # 高斯点 (2×2 per leaf)
    if show_gauss:
        g = gauss_pts_2x2()
        for i in range(n_sub):
            for j in range(n_sub):
                lo_x, lo_y = i * h, j * h
                for gx in g:
                    for gy in g:
                        px = lo_x + 0.5 * (gx + 1) * h
                        py = lo_y + 0.5 * (gy + 1) * h
                        sdf = np.sqrt((px - cx) ** 2 + (py - cy) ** 2) - r
                        if show_interface and abs(sdf) < h * 0.5:
                            # 界面高斯点 (金色)
                            ax.plot(px, py, 'o', color='#FFD700',
                                    markeredgecolor='k', markersize=6, zorder=5)
                        elif sdf < 0:
                            # 固体内高斯点 (深红)
                            ax.plot(px, py, 'o', color='#8B0000',
                                    markersize=3, alpha=0.6, zorder=4)
                        else:
                            # 流体内高斯点 (深蓝)
                            ax.plot(px, py, 'o', color='#1A3A5C',
                                    markersize=3, alpha=0.6, zorder=4)


def plot_interface_projection():
    """图2: 界面投影 (Newton 步) + 法向计算."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title('界面投影与法向计算', fontsize=13, fontweight='bold')

    cx, cy, r = 0.5, 0.5, 0.35

    # 圆 (界面 Φ=0)
    circle = Circle((cx, cy), r, facecolor='#D94A4A', alpha=0.15,
                    edgecolor='#2E8B57', linewidth=2.5, label='界面 Φ=0')
    ax.add_patch(circle)

    # 几个高斯点 → 投影到界面
    np.random.seed(42)
    gauss_pts = np.array([
        [0.30, 0.45],
        [0.65, 0.30],
        [0.75, 0.60],
        [0.40, 0.72],
        [0.55, 0.25],
    ])

    for pt in gauss_pts:
        sdf = np.sqrt((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2) - r
        # 法向 = ∇Φ/|∇Φ| = 径向单位向量 (圆的 SDF 梯度指向外, 即流体侧)
        dx, dy = pt[0] - cx, pt[1] - cy
        dist = np.sqrt(dx ** 2 + dy ** 2)
        nx, ny = dx / dist, dy / dist
        # 投影: x_int = x - sdf * n
        int_pt = pt - sdf * np.array([nx, ny])

        # 绘制原始高斯点
        ax.plot(pt[0], pt[1], 'o', color='#1A3A5C', markersize=8, zorder=5)
        # 绘制投影后界面点
        ax.plot(int_pt[0], int_pt[1], 'o', color='#FFD700',
                markeredgecolor='k', markersize=9, zorder=6)
        # 投影路径 (虚线)
        ax.plot([pt[0], int_pt[0]], [pt[1], int_pt[1]],
                'k--', linewidth=1.2, alpha=0.7, zorder=3)
        # 法向箭头 (从界面点指向流体)
        arrow_len = 0.08
        ax.annotate('', xy=(int_pt[0] + arrow_len * nx, int_pt[1] + arrow_len * ny),
                    xytext=(int_pt[0], int_pt[1]),
                    arrowprops=dict(arrowstyle='->', color='#2E8B57',
                                    lw=2.0), zorder=7)

    # 标注
    ax.annotate('高斯点 $\\mathbf{x}$\n(SDF 分类)', xy=(0.30, 0.45),
                xytext=(0.05, 0.15), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#1A3A5C', lw=1.5))
    ax.annotate('界面点 $\\mathbf{x}_{int}$\n(投影到 Φ=0)', xy=(0.65, 0.30),
                xytext=(0.80, 0.10), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.5))
    ax.annotate('法向 $\\mathbf{n}=\\nabla\\Phi/|\\nabla\\Phi|$', xy=(0.75, 0.60),
                xytext=(0.82, 0.75), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=1.5))
    ax.annotate('投影: $\\mathbf{x}_{int}=\\mathbf{x}-\\mathrm{SDF}\\cdot\\mathbf{n}$',
                xy=(0.40, 0.72), xytext=(0.02, 0.85), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='k', lw=1.0, ls='--'))

    # 中心差分示意 (右下角小图)
    ax_inset = fig.add_axes([0.62, 0.62, 0.30, 0.28])
    ax_inset.set_title('中心差分求 $\\nabla\\Phi$', fontsize=9)
    # 界面局部放大
    theta = np.linspace(-0.3, 0.3, 50)
    ax_inset.plot(cx + r * np.cos(theta + 0.4), cy + r * np.sin(theta + 0.4),
                  '#2E8B57', linewidth=2)
    # 中心点 + 6 个差分点
    pc = np.array([cx + r * np.cos(0.4), cy + r * np.sin(0.4)])
    h = 0.04
    pts = [pc + [h, 0], pc - [h, 0], pc + [0, h], pc - [0, h],
           pc + [h * 0.7, h * 0.7], pc - [h * 0.7, h * 0.7]]
    for p in pts:
        ax_inset.plot(p[0], p[1], 'o', color='#1A3A5C', markersize=4)
    ax_inset.plot(pc[0], pc[1], 'o', color='#FFD700', markeredgecolor='k',
                  markersize=6, zorder=5)
    ax_inset.annotate('', xy=(pc[0] + 0.06, pc[1]), xytext=(pc[0], pc[1]),
                      arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=1.5))
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_aspect('equal')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    out = os.path.join(_FIG_DIR, 'interface_projection.png')
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


def plot_force_scatter():
    """图3: 界面力散列回体素 (np.add.at)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    cx, cy, r = 0.5, 0.5, 0.35

    # ── 子图1: 界面高斯点 + 力向量 ──
    ax = axes[0]
    ax.set_title('(a) 界面高斯点与界面力', fontsize=12, fontweight='bold')

    # 网格
    n = 6
    h = 1.0 / n
    for i in range(n + 1):
        ax.axhline(i * h, color='gray', linewidth=0.4, alpha=0.4)
        ax.axvline(i * h, color='gray', linewidth=0.4, alpha=0.4)

    # 圆
    circle = Circle((cx, cy), r, facecolor='#D94A4A', alpha=0.15,
                    edgecolor='#2E8B57', linewidth=2)
    ax.add_patch(circle)

    # 界面高斯点 (沿圆周分布)
    n_int = 16
    angles = np.linspace(0, 2 * np.pi, n_int, endpoint=False)
    int_pts = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    # 模拟速度场 (向右流), 界面力 = -ρ/Δt * u_int
    u_int = np.column_stack([np.ones(n_int), np.zeros(n_int)])
    force = -1.0 * u_int  # 简化系数

    for i in range(n_int):
        ax.plot(int_pts[i, 0], int_pts[i, 1], 'o', color='#FFD700',
                markeredgecolor='k', markersize=7, zorder=5)
        # 力向量 (蓝色, 指向 -x)
        scale = 0.06
        ax.annotate('', xy=(int_pts[i, 0] + scale * force[i, 0],
                            int_pts[i, 1] + scale * force[i, 1]),
                    xytext=(int_pts[i, 0], int_pts[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='#1A3A5C', lw=1.8),
                    zorder=6)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # ── 子图2: 散列回体素 f_ibm ──
    ax = axes[1]
    ax.set_title('(b) 散列回体素 $\\mathbf{f}_{ibm}$ (np.add.at)', fontsize=12,
                 fontweight='bold')

    # 计算每个体素的力累加
    f_voxel = np.zeros((n, n))
    for i in range(n_int):
        # 找所属体素
        vi = int(int_pts[i, 0] / h)
        vj = int(int_pts[i, 1] / h)
        vi = min(vi, n - 1)
        vj = min(vj, n - 1)
        f_voxel[vi, vj] += force[i, 0]  # x 分量

    # 绘制体素力 (颜色映射)
    vmax = max(abs(f_voxel.min()), abs(f_voxel.max())) + 1e-10
    im = ax.imshow(f_voxel.T, extent=[0, 1, 0, 1], origin='lower',
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax, alpha=0.7)

    # 网格
    for i in range(n + 1):
        ax.axhline(i * h, color='gray', linewidth=0.4, alpha=0.4)
        ax.axvline(i * h, color='gray', linewidth=0.4, alpha=0.4)

    # 圆
    circle = Circle((cx, cy), r, facecolor='none', edgecolor='#2E8B57',
                    linewidth=2, linestyle='--')
    ax.add_patch(circle)

    # 界面点
    ax.plot(int_pts[:, 0], int_pts[:, 1], 'o', color='#FFD700',
            markeredgecolor='k', markersize=5, zorder=5)

    plt.colorbar(im, ax=ax, label='$f_{ibm,x}$ (体素累加值)')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    out = os.path.join(_FIG_DIR, 'force_scatter.png')
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


def plot_gauss_quadrature():
    """图4: 2×2×2 高斯积分规则示意 (参考立方体 + 积分点)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── 子图1: 参考立方体 [-1,1]³ 中的 8 个高斯点 ──
    ax = axes[0]
    ax.set_title('(a) $2\\times2\\times2$ 高斯积分点 (参考立方体)', fontsize=12,
                 fontweight='bold')

    g = 1.0 / np.sqrt(3.0)
    pts = np.array([[i, j, k] for i in [-g, g] for j in [-g, g] for k in [-g, g]])

    # 3D 投影 (等距)
    from mpl_toolkits.mplot3d import Axes3D
    ax.remove()
    ax = fig.add_subplot(121, projection='3d')

    # 立方体边框
    import itertools
    for s, e in itertools.combinations(np.array(list(itertools.product([-1, 1], repeat=3))), 2):
        if np.sum(np.abs(s - e)) == 2:
            ax.plot3D(*zip(s, e), 'k-', alpha=0.5)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='#FFD700', edgecolors='k',
               s=80, zorder=5)
    # 标注权重
    for p in pts:
        ax.text(p[0] * 1.15, p[1] * 1.15, p[2] * 1.15, '$w{=}1$', fontsize=8)

    ax.set_xlabel('$\\xi$')
    ax.set_ylabel('$\\eta$')
    ax.set_zlabel('$\\zeta$')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    # ── 子图2: 1D Gauss-Legendre 公式 ──
    ax = axes[1]
    ax.set_title('(b) 一维 Gauss-Legendre 求积节点', fontsize=12, fontweight='bold')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.0)
    ax.axhline(0, color='k', linewidth=1)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

    g = 1.0 / np.sqrt(3.0)
    ax.plot([-g, g], [0, 0], 'o', color='#FFD700', markeredgecolor='k',
            markersize=12, zorder=5)
    ax.text(-g, 0.15, f'$\\xi_1=-1/\\sqrt{{3}}$\n$w_1=1$',
            ha='center', fontsize=10)
    ax.text(g, 0.15, f'$\\xi_2=+1/\\sqrt{{3}}$\n$w_2=1$',
            ha='center', fontsize=10)
    ax.text(-1, -0.3, '$-1$', ha='center', fontsize=10)
    ax.text(1, -0.3, '$+1$', ha='center', fontsize=10)
    ax.text(0, -0.4, '$\\int_{-1}^{1} f(\\xi)\\,d\\xi \\approx '
            '\\sum_{i=1}^{2} w_i\\, f(\\xi_i)$',
            ha='center', fontsize=11, bbox=dict(facecolor='#FFFFCC', alpha=0.8))
    ax.set_xticks([-1, -g, 0, g, 1])
    ax.set_xticklabels(['$-1$', '$-1/\\sqrt{3}$', '0', '$+1/\\sqrt{3}$', '$+1$'])
    ax.set_yticks([])

    plt.tight_layout()
    out = os.path.join(_FIG_DIR, 'gauss_quadrature.png')
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    os.makedirs(_FIG_DIR, exist_ok=True)
    plot_octree_subdivision()
    plot_interface_projection()
    plot_force_scatter()
    plot_gauss_quadrature()
    print('All sharp-interface IBM schematics generated.')
