"""
Graphics layer: 3D geometric objects.

Point3D, Line3D, Vector3D, Sphere — plain matplotlib artists wrapped as
Renderables. Compose with function plots on the same Canvas3D:

    canvas = Canvas3D()
    canvas.add(Surface(f))
    canvas.add(Point3D(0, 0, 0, color="red"))
    canvas.add(Vector3D((0, 0, 0), (1, 1, 1)))
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from .canvas import Renderable


# ==== Point3D ====
class Point3D(Renderable):
    """A single point (x, y, z) drawn as a marker."""

    def __init__(self, x, y, z, **style):
        self.x = x
        self.y = y
        self.z = z
        style.setdefault("color", "black")
        style.setdefault("marker", "o")
        style.setdefault("zorder", 5)
        self.style = style

    def render(self, ax, **kw):
        merged = {**self.style, **kw}
        return ax.scatter([self.x], [self.y], [self.z], **merged)


# ==== Line3D ====
class Line3D(Renderable):
    """Line segment between two points p and q in R^3."""

    def __init__(self, p, q, **style):
        self.p = p
        self.q = q
        style.setdefault("color", "black")
        style.setdefault("zorder", 4)
        self.style = style

    def render(self, ax, **kw):
        merged = {**self.style, **kw}
        xs = [self.p[0], self.q[0]]
        ys = [self.p[1], self.q[1]]
        zs = [self.p[2], self.q[2]]
        return ax.plot(xs, ys, zs, **merged)


# ==== Vector3D ====
class Vector3D(Renderable):
    """
    Arrow from origin to tip in R^3, drawn with FancyArrowPatch.

    Matplotlib's ax.quiver does not work reliably in 3D; the standard
    workaround is a FancyArrowPatch with its endpoints projected to
    2D display coordinates. This class encapsulates that.
    """

    def __init__(self, origin, tip, **style):
        self.origin = origin
        self.tip = tip
        style.setdefault("color", "black")
        style.setdefault("arrowstyle", "-|>")
        style.setdefault("mutation_scale", 15)
        style.setdefault("linewidth", 1.5)
        self.style = style

    def render(self, ax, **kw):
        merged = {**self.style, **kw}
        ox, oy, oz = self.origin
        tx, ty, tz = self.tip

        # Project 3D points to 2D display coordinates.
        x0, y0, _ = proj3d.proj_transform(ox, oy, oz, ax.get_proj())
        x1, y1, _ = proj3d.proj_transform(tx, ty, tz, ax.get_proj())

        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            transform=ax.transData, **merged,
        )
        ax.add_patch(arrow)
        return arrow

# ==== Plane ====
class Plane(Renderable):
    """
    Plane through point `p` with normal `normal`, drawn as a square
    patch of side `size` centered at `p`.

    Two spanning vectors u, v are computed from `normal` internally;
    the drawn patch is the set  p + s*u + t*v  for s, t in
    [-size/2, size/2].

    Parameters
    ----------
    p : (x, y, z)
    normal : (nx, ny, nz)
        Does not need to be normalized.
    size : float
    n : int
        Mesh resolution (2 is enough for a flat patch).
    cmap : str or Colormap, optional
    color : optional
    """

    def __init__(self, p, normal, size=1.0, n=2, cmap=None, color=None, alpha=None, **style):
        self.p = p
        self.normal = normal
        self.size = size
        self.n = n
        self.cmap = cmap
        self.color = color
        self.alpha = alpha
        style.setdefault("alpha", self.alpha)
        self.style = style

        px, py, pz = p
        nx, ny, nz = normal
        norm = np.sqrt(nx*nx + ny*ny + nz*nz)
        if norm == 0:
            raise ValueError("Plane normal must be nonzero")
        nx, ny, nz = nx/norm, ny/norm, nz/norm
        nvec = np.array([nx, ny, nz])

        # Pick a helper vector not parallel to the normal, then u = n x helper.
        if abs(nx) < 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        else:
            helper = np.array([0.0, 1.0, 0.0])
        u = np.cross(nvec, helper)
        u /= np.linalg.norm(u)
        v = np.cross(nvec, u)

        s = np.linspace(-size/2, size/2, max(n, 2))
        t = np.linspace(-size/2, size/2, max(n, 2))
        S, T = np.meshgrid(s, t)
        self.X = px + S*u[0] + T*v[0]
        self.Y = py + S*u[1] + T*v[1]
        self.Z = pz + S*u[2] + T*v[2]

    def render(self, ax, **kw):
        kw.setdefault("linewidth", 0)
        kw.setdefault("alpha", self.alpha)
        kw.setdefault("antialiased", True)
        if self.cmap is not None:
            kw.setdefault("cmap", self.cmap)
            kw.setdefault("vmin", self.Z.min())
            kw.setdefault("vmax", self.Z.max())
            kw.pop("color", None)
        elif self.color is not None:
            kw.setdefault("color", self.color)
        
        return ax.plot_surface(self.X, self.Y, self.Z, **kw)
