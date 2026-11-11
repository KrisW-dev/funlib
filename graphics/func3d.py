"""
Graphics layer: 3D function plots.

Surface             : graph  z = f(x, y)          from  f : R^2 -> R
ParametricCurve3D   : curve  (x(t), y(t), z(t))   from  (x, y, z) : R -> R^3
ParametricSurface3D : surface (x(u,v), y(u,v), z(u,v))  from  (x, y, z) : R^2 -> R^3

All three are Renderables (see canvas.py). Each is constructed from a
Function-like object f (duck-typed: needs ._args, .coord_matrix, .eval) and
renders into a matplotlib Axes3D via .render(ax, **kw).

No imports from `analysis`: the plot classes only use the Function
interface. This keeps the release merge (analysis.py, graphics.py)
free of an import cycle.
"""

import numpy as np
import matplotlib.pyplot as plt

from .canvas import Renderable
from .func2d import _check_function, _numeric, _linspace


# ==== Surface ====
class Surface(Renderable):
    """
    Graph of a scalar function of two variables:  z = f(x, y).

    Parameters
    ----------
    f : Function-like
        Must satisfy _kind() == "scalar" and dom() == 2.
    x_range, y_range : (float, float)
    n : int
        Grid resolution.
    cmap : str or Colormap, optional
    """

    def __init__(self, f, x_range=(-10, 10), y_range=(-10, 10),n=100, cmap=None, color=None, alpha=None):
        _check_function(f)
        if f._kind() != "scalar" or f.dom() != 2:
            raise ValueError("Surface needs a function R^2 -> R (scalar, dom 2)")
        self.f = f
        self.cmap = cmap
        self.x_range = x_range
        self.y_range = y_range
        self.n = n
        self.alpha = alpha
        self.color = color
        xs = _linspace(x_range, n)
        ys = _linspace(y_range, n)
        self.X, self.Y = np.meshgrid(xs, ys)
        funcs = _numeric(f)          # scalar f: returns a 1-element list
        self.Z = funcs(self.X, self.Y)[0]

    def render(self, ax, **kw):
        kw.setdefault("cmap", self.cmap)
        kw.setdefault("alpha", self.alpha)
        kw.setdefault("linewidth", 0)
        kw.setdefault("antialiased", True)

        if self.cmap is not None:
            kw.setdefault("cmap", self.cmap)
            kw.setdefault("vmin", self.Z.min())
            kw.setdefault("vmax", self.Z.max())
            kw.pop("color", None)          # cmap wins; drop any color kwarg
        elif self.color is not None:
            kw.setdefault("color", self.color)

        ax.auto_scale_xyz(self.X.ravel(), self.Y.ravel(), self.Z.ravel())
        self._surf = ax.plot_surface(self.X, self.Y, self.Z, **kw)
        ax.set_xlabel(str(self.f._args[0]))
        ax.set_ylabel(str(self.f._args[1]))
        ax.set_zlabel(self.f._name)
        return self._surf


# ==== ParametricCurve3D ====
class ParametricCurve3D(Renderable):
    """
    Parametric space curve:  (x(t), y(t), z(t)).

    Parameters
    ----------
    f : Function-like
        Must satisfy _kind() == "vector", dom() == 1, cod() == (3, 1).
    t_range : (float, float)
    n : int
    """

    def __init__(self, f, t_range=(0, 2 * np.pi), n=500, color=None):
        _check_function(f)
        if f._kind() != "vector" or f.dom() != 1 or f.cod() != (3, 1):
            raise ValueError("ParametricCurve3D needs a function R -> R^3 (3 x 1)")
        self.f = f
        self.color = color
        self.t = _linspace(t_range, n)
        funcs = _numeric(f)
        self.x, self.y, self.z = funcs(self.t)

    def render(self, ax, **kw):
        kw.setdefault("label", self.f._name)
        if self.color is not None:
            kw.setdefault("color", self.color)
        ax.auto_scale_xyz(self.x, self.y, self.z)
        ax.plot(self.x, self.y, self.z, **kw)
        ax.set_xlabel(str(self.f._args[0]))
        


# ==== ParametricSurface3D ====
class ParametricSurface3D(Renderable):
    """
    Parametric surface:  (x(u, v), y(u, v), z(u, v)).

    Parameters
    ----------
    f : Function-like
        Must satisfy _kind() == "vector", dom() == 2, cod() == (3, 1).
    u_range, v_range : (float, float)
    n : int
        Grid resolution.
    cmap : str or Colormap, optional
    """

    def __init__(self, f, u_range=(-10, 10), v_range=(-10, 10), n=100, cmap=None, color=None, alpha=None):
        _check_function(f)
        if f._kind() != "vector" or f.dom() != 2 or f.cod() != (3, 1):
            raise ValueError("ParametricSurface3D needs a function R^2 -> R^3 (3 x 1)")
        self.f = f
        self.cmap = cmap
        self.u_range = u_range
        self.v_range = v_range
        self.n = n
        self.alpha = alpha
        self.color = color
        us = _linspace(u_range, n)
        vs = _linspace(v_range, n)
        self.U, self.V = np.meshgrid(us, vs)
        funcs = _numeric(f)
        self.X, self.Y, self.Z = funcs(self.U, self.V)

    def render(self, ax, **kw):
        kw.setdefault("cmap", self.cmap)
        kw.setdefault("alpha", self.alpha)
        kw.setdefault("linewidth", 0)
        kw.setdefault("antialiased", True)
        
        if self.cmap is not None:
            kw.setdefault("cmap", self.cmap)
            kw.setdefault("vmin", self.Z.min())
            kw.setdefault("vmax", self.Z.max())
            kw.pop("color", None)          # cmap wins; drop any color kwarg
        elif self.color is not None:
            kw.setdefault("color", self.color)

        ax.auto_scale_xyz(self.X.ravel(), self.Y.ravel(), self.Z.ravel())
        self._surf = ax.plot_surface(self.X, self.Y, self.Z, **kw)
        ax.set_xlabel(str(self.f._args[0]))
        ax.set_ylabel(str(self.f._args[1]))
        return self._surf