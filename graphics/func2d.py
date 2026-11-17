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
    """

    def __init__(self, f, x_range=(-10, 10), y_range=None, n=500, color=None):
        """
        - f : Function 

        - x_range : (float, float)

        - n : int - number of sample points
        """
        _check_function(f)
        if f._kind() != "scalar" or f.dom() != 1:
            raise ValueError("Curve2D needs a function R -> R (scalar, dom 1)")
        self.f = f
        """@private for pdoc"""
        self.color = color
        """@private for pdoc"""
        self.x = _linspace(x_range, n)
        """@private for pdoc"""
        self.y = _numeric(f)(self.x)[0]
        """@private for pdoc"""
        

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
    """

    def __init__(self, f, t_range=(0, 2 * np.pi), x_range=None, y_range=None, n=500, color=None):
        """
        - f : Function 

        - t_range : (float, float)

        - n : int - number of sample points
        """
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
    """

    def __init__(self, f, x_range=(-10, 10), y_range=(-10, 10), n=100, levels=None, cmap=None, clabel=True):
        """
        - f : Function

        - x_range, y_range : (float, float)

        - n : int - grid resolution

        - levels : int or sequence - optional, forwarded to ax.contour

        - cmap : str or Colormap - optional

        - clabel : bool - labels on the level lines if True (the default)
        """
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
# ==== VectorField2D ====
class VectorField2D(Renderable):
    """
    Vector field  F : R^2 -> R^2  drawn as arrows.

    Sampling locations (exactly one of the following; default = grid):
      - grid (default):  n uniformly spaced points over x_range x y_range
      - points:          an explicit list of (x, y) pairs
      - along:           a ParametricCurve2D, or a Function R -> R^2

    Each arrow is F(p) drawn at p; arrow length is F's own magnitude
    (so a field with small arrows at some points and large at others
    reads correctly). Use `scale` to shrink/enlarge all arrows uniformly.
    """

    def __init__(self, 
            f, x_range=(-2, 2), y_range=(-2, 2),
            grid_density=10, 
            points=None, 
            along=None, t_range=None, n_samples = None,
            scale=None, color=None, **style
        ):
        """
        - f : Function R^2 -> R^2

        - x_range, y_range : (float, float)  bounds of the sampling window

        - n : int or (int, int)  grid density; ignored if points/along given

        - points : list of (x, y)  sample locations; mutually exclusive with along

        - along : Function R -> R^2  sample on the curve

        - scale : float or None  forwarded to ax.quiver's scale; None = auto

        - **style : forwarded to ax.quiver (width, headwidth, pivot, ...)
        """
        _check_function(f)
        if f._kind() != "vector" or f.dom() != 2 or f.cod() != (2, 1):
            raise ValueError("VectorField2D needs a function R^2 -> R^2 (2 x 1)")

        given = [name for name, val in
                 (("points", points), ("along", along)) if val is not None]
        if len(given) > 1:
            raise ValueError(f"only one of points/along may be given; got {given}")
        if given and grid_density != 20:  # user explicitly overrode n but also gave points/along
            # not worth raising — just document that n is ignored
            pass

        self.f = f
        self.scale = scale
        self.color = color
        style.setdefault("angles", "xy")
        style.setdefault("scale_units", "xy")
        #style.setdefault("pivot", "mid")
        style.setdefault("zorder", 3)
        if scale is not None:
            style["scale"] = scale
        self.style = style

        # ---- determine sample locations ----
        if along is not None:
            if t_range is None:
                raise ValueError(
                    "along= requires t_range=(t0, t1); a curve γ: R → R^2 "
                    "does not know its own parameter interval"
                )
            if n_samples is not None:
                xs, ys = self._sample_along(along, t_range, n_samples)
            else:
                xs, ys = self._sample_along(along, t_range)
        elif points is not None:
            xs = np.asarray([p[0] for p in points], dtype=float)
            ys = np.asarray([p[1] for p in points], dtype=float)
        else:
            nx, ny = (grid_density, grid_density) if np.isscalar(grid_density) else grid_density
            xs, ys = np.meshgrid(_linspace(x_range, nx), _linspace(y_range, ny))

        # ---- evaluate F at those points ----
        u, v = _numeric(f)(xs, ys)
        self.X, self.Y = xs, ys
        self.U, self.V = u, v

    #@staticmethod
    # def _sample_along(along, t_range,n_samples=30):
    #     """Sample a curve γ: R → R^2 at n parameter values in [0, n_samples]."""
    #     _check_function(along)
    #     if along._kind() != "vector" or along.dom() != 1 or along.cod() != (2, 1):
    #         raise ValueError(
    #             "along= must be a Function R -> R^2 (a parametric curve); "
    #             "to overlay its plot, add the curve separately"
    #         )
    #     t = np.linspace(float(t_range[0]), float(t_range[1]), n_samples)
    #     x, y = _numeric(along)(t)
    #     return x, y

    @staticmethod
    def _sample_along(along, t_range, n_samples=30):
        """Sample a curve γ: R → R^2 at n_samples parameter values over t_range."""
        _check_function(along)
        if along._kind() != "vector" or along.dom() != 1 or along.cod() != (2, 1):
            raise ValueError(
                "along= must be a Function R -> R^2 (a parametric curve); "
                "to overlay its plot, add the curve separately"
            )
        t = _linspace(t_range, n_samples)          # was np.linspace(float(...), float(...), n_samples)
        x, y = _numeric(along)(t)
        return x, y

    def render(self, ax, **kw):
        merged = {**self.style, **kw}
        if self.color is not None:
            merged.setdefault("color", self.color)
        ax.quiver(self.X, self.Y, self.U, self.V, **merged)
        ax.set_xlabel(str(self.f._args[0]))
        ax.set_ylabel(str(self.f._args[1]))