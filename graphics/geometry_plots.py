"""
Plot builders for the affine-geometry classes in ``analysis.geometry``.

Each builder is a pure constructor: given a ``Vector`` or ``Point`` it returns
a plot-class instance (``Vector2D``/``Vector3D``/``Point2D``/``Point3D``),
which the caller adds to a canvas with ``canvas.add(...)``.  No canvas is
constructed here.

The four builders are registered into ``analysis.geometry``'s
``_GEOMETRY_PLOT_REGISTRY`` from ``graphics/__init__.py``; this module only
defines them.

For a ``Vector``, ``origin`` is the tail of the arrow (default: the origin $O$);
pass a ``Point`` to draw the vector *at* that point.  For a ``Point`` there is
no tail.
"""

from __future__ import annotations

import sympy as sp

from analysis.geometry import Vector, Point

# Import paths: adjust if these names are re-exported by graphics/__init__.py.
from .objects2d import Point2D, Vector2D
from .objects3d import Point3D, Vector3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_origin(origin, dom):
    """Return the tail as a tuple of ``dom`` SymPy expressions.

    ``origin is None``        -> origin O (all zeros),
    ``Vector`` / ``Point``    -> its entries (dimension must match),
    list/tuple                -> sympified entries (length must match).
    """
    if origin is None:
        return tuple(sp.Integer(0) for _ in range(dom))
    if isinstance(origin, (Vector, Point)):
        if origin.dom() != dom:
            raise ValueError(
                f"origin has dimension {origin.dom()}, expected {dom}"
            )
        return origin.entries() if isinstance(origin, Vector) else origin.coords()
    if isinstance(origin, (list, tuple)):
        if len(origin) != dom:
            raise ValueError(
                f"origin has length {len(origin)}, expected {dom}"
            )
        return tuple(sp.sympify(e) for e in origin)
    raise TypeError(
        f"cannot interpret origin {origin!r} as a vector/point/list"
    )


def _styles(style):
    """Drop ``None`` values so class defaults are preserved."""
    return {k: v for k, v in style.items() if v is not None}


# ---------------------------------------------------------------------------
# Builders: Vector
# ---------------------------------------------------------------------------

def _plot_vector_2d(obj, origin=None, **style):
    tail = _coerce_origin(origin, 2)
    tip = tuple(tail[i] + obj.entries()[i] for i in range(2))
    return Vector2D(tail, tip, **_styles(style))


def _plot_vector_3d(obj, origin=None, **style):
    tail = _coerce_origin(origin, 3)
    tip = tuple(tail[i] + obj.entries()[i] for i in range(3))
    return Vector3D(tail, tip, **_styles(style))


# ---------------------------------------------------------------------------
# Builders: Point
# ---------------------------------------------------------------------------

def _plot_point_2d(obj, **style):
    return Point2D(obj.coords(), **_styles(style))


def _plot_point_3d(obj, **style):
    return Point3D(obj.coords(), **_styles(style))