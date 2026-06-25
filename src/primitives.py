# -*- coding: utf-8 -*-
"""
primitives.py — 基本几何体素及其有符号距离函数(SDF)
支持: 立方体、圆柱体、球体（沿Y轴方向）
"""
import numpy as np
from abc import ABC, abstractmethod


class Primitive(ABC):
    """基本体素抽象类"""
    def __init__(self, name=""):
        self.name = name

    @abstractmethod
    def sdf(self, x, y, z):
        """返回点(x,y,z)的有符号距离: 负=内部, 正=外部, 0=边界"""
        pass

    @abstractmethod
    def get_params(self):
        """返回参数字典"""
        pass

    @abstractmethod
    def set_param(self, name, val):
        """设置参数"""
        pass


class Cube(Primitive):
    """立方体（轴对齐）: 中心(cx,cy,cz), 尺寸(sx,sy,sz)"""
    def __init__(self, cx=0, cy=0, cz=0, sx=1, sy=1, sz=1, name=""):
        super().__init__(name)
        self.cx, self.cy, self.cz = cx, cy, cz
        self.sx, self.sy, self.sz = sx, sy, sz

    def sdf(self, x, y, z):
        dx = np.abs(x - self.cx) - self.sx/2
        dy = np.abs(y - self.cy) - self.sy/2
        dz = np.abs(z - self.cz) - self.sz/2
        d = np.array([dx, dy, dz])
        # SDF = max(d) 对于外部, 次大值负号 对于内部
        dpos = np.max(d)
        if dpos < 0:
            # 内部: 返回最大的负距离（最靠近边界）
            return dpos
        # 外部: 返回距离
        return dpos

    def get_params(self):
        return {'cx': self.cx, 'cy': self.cy, 'cz': self.cz,
                'sx': self.sx, 'sy': self.sy, 'sz': self.sz}

    def set_param(self, name, val):
        setattr(self, name, float(val))


class CylinderY(Primitive):
    """
    沿Y轴方向的圆柱体
    cx,cz: 轴心在XZ平面的位置
    r: 半径
    ymin, ymax: Y方向的起始和结束
    """
    def __init__(self, cx=0, cz=0, r=1, ymin=-1, ymax=1, name=""):
        super().__init__(name)
        self.cx, self.cz = cx, cz
        self.r = r
        self.ymin, self.ymax = ymin, ymax

    def sdf(self, x, y, z):
        dx = np.sqrt((x - self.cx)**2 + (z - self.cz)**2) - self.r
        dy = max(y - self.ymax, self.ymin - y)
        # 组合: 圆柱体 SDF
        if dx >= 0 or dy >= 0:
            return np.sqrt(max(dx, 0)**2 + max(dy, 0)**2)
        else:
            return max(dx, dy)

    def get_params(self):
        return {'cx': self.cx, 'cz': self.cz, 'r': self.r,
                'ymin': self.ymin, 'ymax': self.ymax}

    def set_param(self, name, val):
        setattr(self, name, float(val))


class CylinderZ(Primitive):
    """
    沿Z轴方向的圆柱体
    cx,cy: 轴心在XY平面的位置
    r: 半径
    zmin, zmax: Z方向的起始和结束
    """
    def __init__(self, cx=0, cy=0, r=1, zmin=-1, zmax=1, name=""):
        super().__init__(name)
        self.cx, self.cy = cx, cy
        self.r = r
        self.zmin, self.zmax = zmin, zmax

    def sdf(self, x, y, z):
        dx = np.sqrt((x - self.cx)**2 + (y - self.cy)**2) - self.r
        dz = max(z - self.zmax, self.zmin - z)
        if dx >= 0 or dz >= 0:
            return np.sqrt(max(dx, 0)**2 + max(dz, 0)**2)
        else:
            return max(dx, dz)

    def get_params(self):
        return {'cx': self.cx, 'cy': self.cy, 'r': self.r,
                'zmin': self.zmin, 'zmax': self.zmax}

    def set_param(self, name, val):
        setattr(self, name, float(val))


class Sphere(Primitive):
    """球体: 中心(cx,cy,cz), 半径r"""
    def __init__(self, cx=0, cy=0, cz=0, r=1, name=""):
        super().__init__(name)
        self.cx, self.cy, self.cz = cx, cy, cz
        self.r = r

    def sdf(self, x, y, z):
        return np.sqrt((x-self.cx)**2 + (y-self.cy)**2 + (z-self.cz)**2) - self.r

    def get_params(self):
        return {'cx': self.cx, 'cy': self.cy, 'cz': self.cz, 'r': self.r}

    def set_param(self, name, val):
        setattr(self, name, float(val))


class RoundCorner2D(Primitive):
    """
    2D凹圆角 / 内角倒圆（与两个表面相切的 fillet）
    
    (cx, cy) = 角顶点坐标。圆柱圆心自动偏移到 (cx + sign_x*r, cy + sign_y*r)
    以保证圆柱与两个表面相切。
    
    sign_x=-1, sign_y=-1: 左下象限 (x<cx, y<cy), 圆心 (cx-r, cy-r)
    sign_x=+1, sign_y=+1: 右上象限 (x>cx, y>cy), 圆心 (cx+r, cy+r)
    
    nature=+1（加材料）填充凹入角，形成 concave fillet。
    nature=-1（减材料）切掉外角，形成 convex round。
    """
    def __init__(self, cx=0, cy=0, r=1, zmin=-1, zmax=1,
                 sign_x=-1, sign_y=-1, name=""):
        super().__init__(name)
        self.cx, self.cy = cx, cy   # 角顶点
        self.r = r
        self.zmin, self.zmax = zmin, zmax
        self.sign_x = sign_x  # -1: x<cx; +1: x>cx
        self.sign_y = sign_y  # -1: y<cy; +1: y>cy

    def _center(self):
        """圆柱圆心: 从角顶点偏移 ±r 以保证与两表面相切"""
        return (self.cx + self.sign_x * self.r,
                self.cy + self.sign_y * self.r)

    def sdf(self, x, y, z):
        """
        Corner fill SDF: Box([cx, cx+sign_x*r], [cy, cy+sign_y*r]) - Cylinder
        
        填充区域 = 方框减去1/4圆柱（劣弧包围的区域）。
        
        SDF < 0 当且仅当:
          1. 0 < sign_x*(x-cx) < r   — 在方框内 (x方向)
          2. 0 < sign_y*(y-cy) < r   — 在方框内 (y方向)
          3. (x-(cx+sign_x*r))² + (y-(cy+sign_y*r))² > r²  — 在圆柱外
          4. zmin < z < zmax          — 在z范围内
        """
        ocx, ocy = self._center()
        # Box SDF: box = [cx, cx+sign_x*r] × [cy, cy+sign_y*r]
        # Box center = (cx + sign_x*r/2, cy + sign_y*r/2), half-size = r/2
        bcx = self.cx + self.sign_x * self.r / 2.0
        bcy = self.cy + self.sign_y * self.r / 2.0
        box_x = np.abs(x - bcx) - self.r / 2.0
        box_y = np.abs(y - bcy) - self.r / 2.0
        box_sdf = np.maximum(box_x, box_y)

        # Cylinder SDF (restricted to quadrant)
        radial = np.sqrt((x - ocx)**2 + (y - ocy)**2) - self.r
        sx = self.sign_x * (x - self.cx)
        sy = self.sign_y * (y - self.cy)
        cyl_quad = np.maximum(np.maximum(-sx, sx - self.r),
                               np.maximum(-sy, sy - self.r))
        cyl_sdf = np.maximum(radial, cyl_quad)

        # Corner fill = Box - Cylinder: SDF = max(box_sdf, -cyl_sdf)
        fill_sdf = np.maximum(box_sdf, -cyl_sdf)

        # Z方向
        dz = max(z - self.zmax, self.zmin - z)
        if dz >= 0:
            return np.sqrt(max(fill_sdf, 0)**2 + dz**2)
        else:
            return max(fill_sdf, dz)

    def get_params(self):
        return {'cx': self.cx, 'cy': self.cy, 'r': self.r,
                'zmin': self.zmin, 'zmax': self.zmax}

    def set_param(self, name, val):
        setattr(self, name, float(val))