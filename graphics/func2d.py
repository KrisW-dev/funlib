"""
Graphics layer: 2D function plots.

Curve2D            : graph  y = f(x)      from  f : R -> R
ParametricCurve2D  : curve  (x(t), y(t))  from  (x, y) : R -> R^2
Implicit2D         : level sets f(x, y) = c  from  f : R^2 -> R

All three are Renderables (see canvas.py). Each is constructed from a
Function-like object f (duck-typed: needs ._args, .coord_matrix, .eval) and
renders into a matplotlib Axes via .render(ax, **kw).

No imports from `analysis`: the plot classes only use the Function
interface. This keeps the release merge (analysis.py, graphics.py)
free of an import cycle.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from .canvas import Renderable


def _check_function(f):
    """Duck-typed guard; gives students a clear error, not an AttributeError."""
    if not (hasattr(f, "_args") and hasattr(f, "coord_matrix") and hasattr(f, "eval")):
        raise TypeError("expected a Function-like object with ._args, .coord_matrix, .eval")

def _numeric(f):
    funcs = sp.lambdify(f._args, list(f.coord_matrix), "numpy")

    def wrapped(*arrays):
        shape = np.shape(arrays[0])
        result = funcs(*arrays)
        return [np.broadcast_to(np.asarray(r, dtype=float), shape) for r in result]

    return wrapped # This is a closure, it has funcs in its environment

def _linspace(rng, n):
    """np.linspace over a range whose endpoints may be SymPy numbers."""
    return np.linspace(float(rng[0]), float(rng[1]), n)


# ==== Curve2D ====
class Curve2D(Renderable):
    """
    Graph of a scalar function of one variable:  y = f(x).

    Parameters
    ----------
    f : Function-like
        Must satisfy _kind() == "scalar" and dom() == 1.
    x_range : (float, float)
    n : int
        Number of sample points.
    """

    def __init__(self, f, x_range=(-10, 10), y_range=None, n=500, color=None):
        _check_function(f)
        if f._kind() != "scalar" or f.dom() != 1:
            raise ValueError("Curve2D needs a function R -> R (scalar, dom 1)")
        self.f = f
        self.color = color
        self.x = _linspace(x_range, n)
        self.y = _numeric(f)(self.x)[0]
        

    def render(self, ax, **kw):
        kw.setdefault("label", self.f._name)
        if self.color is not None:
            kw.setdefault("color", self.color)
        ax.plot(self.x, self.y, **kw)
        ax.set_xlabel(str(self.f._args[0]))
        ax.set_ylabel(self.f._name)


# ==== ParametricCurve2D ====
class ParametricCurve2D(Renderable):
    """
    Parametric plane curve:  (x(t), y(t)).

    Parameters
    ----------
    f : Function-like
        Must satisfy _kind() == "vector", dom() == 1, cod() == (2, 1).
        The coord_matrix is 2 x 1; the two entries are x(t), y(t).
    t_range : (float, float)
    n : int
    """

    def __init__(self, f, t_range=(0, 2 * np.pi), x_range=None, y_range=None, n=500, color=None):
        _check_function(f)
        if f._kind() != "vector" or f.dom() != 1 or f.cod() != (2, 1):
            raise ValueError("ParametricCurve2D needs a function R -> R^2 (2 x 1)")
        self.f = f
        self.color = color
        self.t = _linspace(t_range, n)
        funcs = _numeric(f)
        self.x, self.y = funcs(self.t)
        

    def render(self, ax, **kw):
        kw.setdefault("label", self.f._name)
        if self.color is not None:
            kw.setdefault("color", self.color)
        ax.plot(self.x, self.y, **kw)
        ax.set_xlabel(str(self.f._args[0]))


# ==== Implicit2D ====
class Implicit2D(Renderable):
    """
    Level sets of a scalar function of two variables:  f(x, y) = c.

    Parameters
    ----------
    f : Function-like
        Must satisfy _kind() == "scalar" and dom() == 2.
    x_range, y_range : (float, float)
    n : int
        Grid resolution.
    levels : int or sequence, optional
        Forwarded to ax.contour.
    cmap : str or Colormap, optional
    clabel : bool
        If True, label the contour lines.
    """

    def __init__(self, f, x_range=(-10, 10), y_range=(-10, 10), n=100, levels=None, cmap=None, clabel=True):
        _check_function(f)
        if f._kind() != "scalar" or f.dom() != 2:
            raise ValueError("Implicit2D needs a function R^2 -> R (scalar, dom 2)")
        self.f = f
        self.x_range = x_range
        self.y_range = y_range
        self.n = n
        self.levels = levels
        self.cmap = cmap
        self.clabel = clabel
        xs = _linspace(x_range, n)
        ys = _linspace(y_range, n)
        self.X, self.Y = np.meshgrid(xs, ys)
        self.Z = _numeric(f)(self.X, self.Y)[0]

    def render(self, ax, **kw):
        cs = ax.contour(self.X, self.Y, self.Z,
                        levels=self.levels, cmap=self.cmap, **kw)
        if self.clabel:
            ax.clabel(cs, inline=True, fontsize=10)
        ax.set_xlabel(str(self.f._args[0]))
        ax.set_ylabel(str(self.f._args[1]))
        return cs