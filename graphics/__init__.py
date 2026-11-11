# graphics/__init__.py
"""Graphics layer: canvases, plot classes, geometric objects (matplotlib-based).
It is technical documentation, an
API user is not required to know any of this."""

from .canvas import Canvas2D, Canvas3D, Renderable
from .func2d import Curve2D, ParametricCurve2D, Implicit2D
from .func3d import Surface, ParametricCurve3D, ParametricSurface3D
from .objects2d import Point2D, Line2D, Vector2D, Polygon2D
from .objects3d import Point3D, Line3D, Vector3D, Plane

from analysis.function import _PLOT_REGISTRY

_PLOT_REGISTRY.update({
    # ---- phase 1 ----
    ("scalar", 1, 1): {"2d": Curve2D,           "3d": None},
    ("scalar", 2, 1): {"2d": Implicit2D,        "3d": Surface},
    ("vector", 1, 2): {"2d": ParametricCurve2D, "3d": None},
    ("vector", 1, 3): {"2d": None,              "3d": ParametricCurve3D},
    ("vector", 2, 3): {"2d": None,              "3d": ParametricSurface3D},

    # ---- phase 2 (uncomment when VectorField2D / VectorField3D exist) ----
    # ("vector", 2, 2): {"2d": VectorField2D,   "3d": None},   # incl. grad() of R^2 -> R
    # ("vector", 3, 3): {"2d": None,            "3d": VectorField3D},  # incl. grad() of R^3 -> R
})

__all__ = [
    # canvas
    "Canvas2D", "Canvas3D", "Renderable",
    # 2D function plots
    "Curve2D", "ParametricCurve2D", "Implicit2D",
    # 3D function plots
    "Surface", "ParametricCurve3D", "ParametricSurface3D",
    # 2D objects
    "Point2D", "Line2D", "Vector2D", "Polygon2D",
    # 3D objects
    "Point3D", "Line3D", "Vector3D", "Plane",
]