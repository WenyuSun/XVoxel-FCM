# -*- coding: utf-8 -*-
"""
primitives.py — 几何基元 (叶子节点)

所有基元继承 Primitive, 实现 sdf_batch(points) 向量化 SDF 求值.

设计决策:
    - Primitive 继承 Feature (定义在 csg.py)
    - 向量化实现在 sdf_batch 方法中, 消除逐体素的 Python for 循环
    - 保留 sdf() 单点接口用于调试兼容
"""
import numpy as np
from abc import abstractmethod
from .csg import Feature


class Primitive(Feature):
    """几何基元抽象类 (叶子节点)."""

    def __init__(self, name: str = ""):
        super().__init__(feature_id=-1, name=name)

    @abstractmethod
    def sdf(self, x: float, y: float, z: float) -> float:
        """单点 SDF — 兼容旧 API, 调试用."""
        ...

    @abstractmethod
    def get_params(self):
        """返回参数字典."""
        ...

    @abstractmethod
    def set_param(self, name, val):
        """设置参数."""
        ...


class Cube(Primitive):
    """长方体基元: 中心(cx,cy,cz), 尺寸(sx,sy,sz).

    SDF 公式: max(|dx|-sx/2, |dy|-sy/2, |dz|-sz/2)
    内部: max 为负; 外部: max 为正或 sqrt(sum(max(0, di)^2)).
    """
    def __init__(self, cx=0.0, cy=0.0, cz=0.0,
                 sx=1.0, sy=1.0, sz=1.0, name=""):
        super().__init__(name)
        self.cx, self.cy, self.cz = float(cx), float(cy), float(cz)
        self.sx, self.sy, self.sz = float(sx), float(sy), float(sz)

    def sdf(self, x, y, z):
        dx = np.abs(x - self.cx) - self.sx / 2.0
        dy = np.abs(y - self.cy) - self.sy / 2.0
        dz = np.abs(z - self.cz) - self.sz / 2.0
        d = np.array([dx, dy, dz])
        dpos = np.max(d)
        if dpos < 0:
            return float(dpos)
        return float(np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2 + np.maximum(dz, 0)**2))

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        """向量化批量 SDF. points: (N, 3) → (N,)."""
        dx = np.abs(points[:, 0] - self.cx) - self.sx / 2.0
        dy = np.abs(points[:, 1] - self.cy) - self.sy / 2.0
        dz = np.abs(points[:, 2] - self.cz) - self.sz / 2.0

        # 外部距离 (Pythagorean)
        d_ext = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2 + np.maximum(dz, 0)**2)
        # 内部距离 (max of signed distances)
        d_int = np.maximum(np.maximum(dx, dy), dz)
        # 向量化分支: 全部在内部用 d_int, 否则用 d_ext
        inside = (dx < 0) & (dy < 0) & (dz < 0)
        return np.where(inside, d_int, d_ext)

    def get_params(self):
        return {'cx': self.cx, 'cy': self.cy, 'cz': self.cz,
                'sx': self.sx, 'sy': self.sy, 'sz': self.sz}

    def set_param(self, name, val):
        setattr(self, name, float(val))


class CylinderZ(Primitive):
    """沿 Z 轴圆柱体: 轴心(cx,cy), 半径 r, Z 范围[zmin, zmax].

    SDF: 组合径向距离和轴向距离.
    径向: sqrt((x-cx)² + (y-cy)²) - r
    轴向: max(z-zmax, zmin-z)
    内部: max(径向, 轴向)
    外部: sqrt(max(径向,0)² + max(轴向,0)²)
    """
    def __init__(self, cx=0.0, cy=0.0, r=1.0, zmin=-1.0, zmax=1.0, name=""):
        super().__init__(name)
        self.cx, self.cy = float(cx), float(cy)
        self.r = float(r)
        self.zmin, self.zmax = float(zmin), float(zmax)

    def sdf(self, x, y, z):
        dx = np.sqrt((x - self.cx)**2 + (y - self.cy)**2) - self.r
        dz = max(z - self.zmax, self.zmin - z)
        if dx >= 0 or dz >= 0:
            return float(np.sqrt(max(dx, 0)**2 + max(dz, 0)**2))
        else:
            return float(max(dx, dz))

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        """向量化批量 SDF."""
        # 径向 SDF
        d_radial = np.sqrt((points[:, 0] - self.cx)**2 + (points[:, 1] - self.cy)**2) - self.r
        # 轴向 SDF
        d_axial = np.maximum(points[:, 2] - self.zmax, self.zmin - points[:, 2])

        # 外部: sqrt(max(0,d_radial)² + max(0,d_axial)²)
        ext = np.sqrt(np.maximum(d_radial, 0)**2 + np.maximum(d_axial, 0)**2)
        # 内部: max(d_radial, d_axial)
        inside = (d_radial < 0) & (d_axial < 0)
        return np.where(inside, np.maximum(d_radial, d_axial), ext)

    def get_params(self):
        return {'cx': self.cx, 'cy': self.cy, 'r': self.r,
                'zmin': self.zmin, 'zmax': self.zmax}

    def set_param(self, name, val):
        setattr(self, name, float(val))


class CylinderY(Primitive):
    """沿 Y 轴圆柱体: 轴心(cx,cz), 半径 r, Y 范围[ymin, ymax]."""
    def __init__(self, cx=0.0, cz=0.0, r=1.0, ymin=-1.0, ymax=1.0, name=""):
        super().__init__(name)
        self.cx, self.cz = float(cx), float(cz)
        self.r = float(r)
        self.ymin, self.ymax = float(ymin), float(ymax)

    def sdf(self, x, y, z):
        dx = np.sqrt((x - self.cx)**2 + (z - self.cz)**2) - self.r
        dy = max(y - self.ymax, self.ymin - y)
        if dx >= 0 or dy >= 0:
            return float(np.sqrt(max(dx, 0)**2 + max(dy, 0)**2))
        else:
            return float(max(dx, dy))

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        d_radial = np.sqrt((points[:, 0] - self.cx)**2 + (points[:, 2] - self.cz)**2) - self.r
        d_axial = np.maximum(points[:, 1] - self.ymax, self.ymin - points[:, 1])
        ext = np.sqrt(np.maximum(d_radial, 0)**2 + np.maximum(d_axial, 0)**2)
        inside = (d_radial < 0) & (d_axial < 0)
        return np.where(inside, np.maximum(d_radial, d_axial), ext)

    def get_params(self):
        return {'cx': self.cx, 'cz': self.cz, 'r': self.r,
                'ymin': self.ymin, 'ymax': self.ymax}

    def set_param(self, name, val):
        setattr(self, name, float(val))


class Sphere(Primitive):
    """球体: 中心(cx,cy,cz), 半径 r."""
    def __init__(self, cx=0.0, cy=0.0, cz=0.0, r=1.0, name=""):
        super().__init__(name)
        self.cx, self.cy, self.cz = float(cx), float(cy), float(cz)
        self.r = float(r)

    def sdf(self, x, y, z):
        return float(np.sqrt((x - self.cx)**2 + (y - self.cy)**2 + (z - self.cz)**2) - self.r)

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        return np.sqrt((points[:, 0] - self.cx)**2 +
                       (points[:, 1] - self.cy)**2 +
                       (points[:, 2] - self.cz)**2) - self.r

    def get_params(self):
        return {'cx': self.cx, 'cy': self.cy, 'cz': self.cz, 'r': self.r}

    def set_param(self, name, val):
        setattr(self, name, float(val))


class RoundCorner2D(Primitive):
    """
    2D 凹圆角 / 内角倒圆（与两个表面相切的 fillet）.

    (cx, cy) = 角顶点坐标。圆柱圆心自动偏移到 (cx + sign_x*r, cy + sign_y*r)
    以保证圆柱与两个表面相切。

    sign_x=-1, sign_y=-1: 左下象限 (x<cx, y<cy), 圆心 (cx-r, cy-r)
    sign_x=+1, sign_y=+1: 右上象限 (x>cx, y>cy), 圆心 (cx+r, cy+r)
    """
    def __init__(self, cx=0.0, cy=0.0, r=1.0, zmin=-1.0, zmax=1.0,
                 sign_x=-1, sign_y=-1, name=""):
        super().__init__(name)
        self.cx, self.cy = float(cx), float(cy)
        self.r = float(r)
        self.zmin, self.zmax = float(zmin), float(zmax)
        self.sign_x = int(sign_x)
        self.sign_y = int(sign_y)

    def _center(self):
        return (self.cx + self.sign_x * self.r,
                self.cy + self.sign_y * self.r)

    def sdf(self, x, y, z):
        ocx, ocy = self._center()
        bcx = self.cx + self.sign_x * self.r / 2.0
        bcy = self.cy + self.sign_y * self.r / 2.0
        box_x = np.abs(x - bcx) - self.r / 2.0
        box_y = np.abs(y - bcy) - self.r / 2.0
        box_sdf = max(box_x, box_y)

        radial = np.sqrt((x - ocx)**2 + (y - ocy)**2) - self.r
        sx = self.sign_x * (x - self.cx)
        sy = self.sign_y * (y - self.cy)
        cyl_quad = max(max(-sx, sx - self.r), max(-sy, sy - self.r))
        cyl_sdf = max(radial, cyl_quad)

        fill_sdf = max(box_sdf, -cyl_sdf)
        dz = max(z - self.zmax, self.zmin - z)
        if dz >= 0:
            return float(np.sqrt(max(fill_sdf, 0)**2 + dz**2))
        else:
            return float(max(fill_sdf, dz))

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        """向量化批量 SDF."""
        ocx, ocy = self._center()
        bcx = self.cx + self.sign_x * self.r / 2.0
        bcy = self.cy + self.sign_y * self.r / 2.0

        # Box SDF
        box_x = np.abs(points[:, 0] - bcx) - self.r / 2.0
        box_y = np.abs(points[:, 1] - bcy) - self.r / 2.0
        box_sdf = np.maximum(box_x, box_y)

        # Cylinder SDF
        radial = np.sqrt((points[:, 0] - ocx)**2 + (points[:, 1] - ocy)**2) - self.r
        sx = self.sign_x * (points[:, 0] - self.cx)
        sy = self.sign_y * (points[:, 1] - self.cy)
        cyl_quad = np.maximum(np.maximum(-sx, sx - self.r),
                               np.maximum(-sy, sy - self.r))
        cyl_sdf = np.maximum(radial, cyl_quad)

        # Corner fill = Box - Cylinder
        fill_sdf = np.maximum(box_sdf, -cyl_sdf)

        # Z 方向
        dz = np.maximum(points[:, 2] - self.zmax, self.zmin - points[:, 2])
        inside = dz < 0
        ext = np.sqrt(np.maximum(fill_sdf, 0)**2 + np.maximum(dz, 0)**2)
        return np.where(inside, np.maximum(fill_sdf, dz), ext)

    def get_params(self):
        return {'cx': self.cx, 'cy': self.cy, 'r': self.r,
                'zmin': self.zmin, 'zmax': self.zmax}

    def set_param(self, name, val):
        setattr(self, name, float(val))
