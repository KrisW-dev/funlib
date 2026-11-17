"""
Affine geometry and basic vector operations.

This module is deliberately *elementary*: it exists so that students can see,
concretely, the difference between

- a **vector** -- an element of a vector space $R^n$, which you may add to
  another vector and scale by a number, but which has no position, and
- a **point** -- an element of an affine space, which has a position but which
  you may *not* add to another point.

The bridge between the two is the choice of an origin ``O``::

    p.to_vector()   is the vector from O to p,
    v.to_point()    is O + v.

Both are representations of the same underlying ``n``-tuple of coordinates, but
they are *different gadgets*, and the arithmetic reflects that.  In particular::

    p + v   ->  Point      (translate a point by a vector)
    p - q   ->  Vector     (difference of two points)
    p + q   ->  TypeError  (adding points is meaningless)

SymPy-only: this module has **no plotting dependencies**.  ``Vector.plot`` and
``Point.plot`` look up a builder in ``_GEOMETRY_PLOT_REGISTRY``, which the
``graphics`` subpackage populates at import time (mirroring ``_PLOT_REGISTRY``
for ``Function``).  Import ``analysis`` and ``graphics`` to enable plotting.
"""

from __future__ import annotations

import sympy as sp
from sympy import ImmutableMatrix, Expr, latex, sympify


# ---------------------------------------------------------------------------
# Plot registry (populated by graphics at import time)
# ---------------------------------------------------------------------------

#: Maps ``(cls, dom)`` -> builder.  ``cls`` is ``Vector`` or ``Point``, ``dom``
#: is the dimension ``n``.  Builders have signature
#: ``builder(obj, origin=None, **style) -> Renderable`` for ``Vector`` and
#: ``builder(obj, **style) -> Renderable`` for ``Point``.  They construct a
#: plot-class instance (``Vector2D``/``Point3D``/...) which the caller adds to a
#: canvas with ``canvas.add(...)``.
_GEOMETRY_PLOT_REGISTRY: dict = {}


def _register_plot(cls, dom, builder):
    """Register a plotting builder for ``(cls, dom)``.  Called by graphics."""
    _GEOMETRY_PLOT_REGISTRY[(cls, dom)] = builder


def _require_plot(cls, dom):
    key = (cls, dom)
    if key not in _GEOMETRY_PLOT_REGISTRY:
        raise ImportError(
            f"No plot builder registered for {cls.__name__} in dimension {dom}. "
            f"Import `graphics` to enable plotting."
        )
    return _GEOMETRY_PLOT_REGISTRY[key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_column(entries, *, allow_scalar=False):
    """Coerce ``entries`` to an ``ImmutableMatrix`` of shape ``n x 1``.

    Accepts a flat list/tuple (canonical), an ``n x 1`` matrix, a ``Vector`` or
    ``Point`` (taking its column), or a scalar when ``allow_scalar`` (giving a
    ``1 x 1``).
    """
    if isinstance(entries, (Vector, Point)):
        return entries._m
    if isinstance(entries, ImmutableMatrix):
        if entries.cols != 1:
            raise ValueError(
                f"expected a column vector (n x 1), got shape {entries.shape}"
            )
        return entries
    if isinstance(entries, sp.MatrixBase):
        if entries.cols != 1:
            raise ValueError(
                f"expected a column vector (n x 1), got shape {entries.shape}"
            )
        return ImmutableMatrix(entries)
    if allow_scalar and not isinstance(entries, (list, tuple)):
        return ImmutableMatrix([sympify(entries)])
    if isinstance(entries, (list, tuple)):
        return ImmutableMatrix([sympify(e) for e in entries])
    raise TypeError(
        f"cannot interpret {entries!r} as a column vector; "
        f"pass a list of entries, a column matrix, or Vector/Point."
    )


# ---------------------------------------------------------------------------
# Vector
# ---------------------------------------------------------------------------

class Vector:
    """
    An element of $\\mathbb{R}^n$, stored as a SymPy $n \\times 1$ column.

    A ``Vector`` may be added to another ``Vector``, scaled by a scalar, and
    (in $n = 3$) crossed with another ``Vector``.  It has **no position**:
    to place it in space, use :meth:`plot` with an ``origin``, or
    :meth:`to_point`.

    Examples
    --------
    >>> v = Vector([3, 4])
    >>> v.norm()
    5
    >>> Vector([1, 0, 0]).cross(Vector([0, 1, 0]))
    Vector([0, 1, 0])
    """

    __slots__ = ("_m",)

    def __init__(self, entries):
        self._m = _as_column(entries)

    # -- basic accessors --------------------------------------------------

    def dom(self) -> int:
        """Dimension $n$ of the ambient space."""
        return self._m.rows

    def entries(self):
        """The entries as a tuple of SymPy expressions."""
        return tuple(self._m[i, 0] for i in range(self.dom()))

    def as_matrix(self) -> ImmutableMatrix:
        """The underlying $n \\times 1$ SymPy column matrix."""
        return self._m

    # -- arithmetic -------------------------------------------------------

    def _check_same_dom(self, other, op):
        if not isinstance(other, Vector):
            return NotImplemented
        if self.dom() != other.dom():
            raise ValueError(
                f"cannot {op} vectors of dimensions {self.dom()} and {other.dom()}"
            )
        return other

    def __add__(self, other):
        if self._check_same_dom(other, "add") is NotImplemented:
            return NotImplemented
        return Vector(self._m + other._m)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if self._check_same_dom(other, "subtract") is NotImplemented:
            return NotImplemented
        return Vector(self._m - other._m)

    def __neg__(self):
        return Vector(-self._m)

    def __mul__(self, scalar):
        if isinstance(scalar, Vector):
            raise TypeError(
                "Vector * Vector is ambiguous; use .dot(w) or .cross(w)."
            )
        if isinstance(scalar, Point):
            raise TypeError(
                "cannot multiply a vector by a point; "
                "did you mean .dot(p.to_vector())?"
            )
        return Vector(self._m * sympify(scalar))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if isinstance(scalar, (Vector, Point)):
            raise TypeError("Vector / Vector is undefined.")
        return Vector(self._m / sympify(scalar))

    # -- inner products ---------------------------------------------------

    def dot(self, other) -> Expr:
        """Euclidean scalar product $v \\cdot w$ (returns a SymPy expression)."""
        if self._check_same_dom(other, "dot") is NotImplemented:
            raise TypeError("dot product requires another Vector of the same dimension")
        return sp.expand((self._m.T * other._m)[0, 0])

    def norm(self) -> Expr:
        """Euclidean norm $\\sqrt{v \\cdot v}$ -- symbolic when the entries are."""
        return sp.sqrt(self.dot(self))

    def cross(self, other) -> "Vector":
        """Cross product, defined only for $n = 3$."""
        if self.dom() != 3:
            raise ValueError(
                f"cross product is defined only in dimension 3, not {self.dom()}"
            )
        if self._check_same_dom(other, "cross") is NotImplemented:
            raise TypeError("cross product requires another Vector of the same dimension")
        a, b = self._m, other._m
        return Vector([
            a[1, 0] * b[2, 0] - a[2, 0] * b[1, 0],
            a[2, 0] * b[0, 0] - a[0, 0] * b[2, 0],
            a[0, 0] * b[1, 0] - a[1, 0] * b[0, 0],
        ])

    # -- conversion -------------------------------------------------------

    def to_point(self) -> "Point":
        """The point $O + v$ (representative at the fixed origin $O$)."""
        return Point(self._m)

    # -- display ----------------------------------------------------------

    def __repr__(self):
        return f"Vector({list(self.entries())!r})"

    def _repr_latex_(self):
        # SymPy's latex() already emits a bmatrix for a column.
        return latex(self._m)

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self._m == other._m
        return NotImplemented

    def __hash__(self):
        return hash(("Vector", self._m))

    # -- plotting ---------------------------------------------------------

    def plot(self, origin=None, **style):
        """Return a renderable arrow for this vector.

        The tail of the arrow is ``origin`` (default: the origin $O$); pass a
        ``Point`` to draw the vector *at* that point.  The returned object is a
        plot-class instance (``Vector2D`` or ``Vector3D``); display it by adding
        it to a canvas::

            canv = Canvas2D()
            canv.add(Vector([1, 1]).plot(origin=Point([2, 2])))
            canv

        Extra keyword arguments (``color=``, ``label=``, ...) are forwarded to
        the plot class.
        """
        builder = _require_plot(Vector, self.dom())
        return builder(self, origin=origin, **style)


# ---------------------------------------------------------------------------
# Point
# ---------------------------------------------------------------------------

class Point:
    """
    An element of affine $n$-space, stored as a SymPy $n \\times 1$ column.

    A point has a *position*.  You may translate it by a vector ($p + v$),
    and take the difference of two points to get a vector ($p - q$), but
    adding two points is **not defined** -- this is the defining feature of
    affine space, not an oversight.

    Examples
    --------
    >>> p, q = Point([1, 2]), Point([4, 6])
    >>> q - p
    Vector([3, 4])
    >>> p + Vector([1, 1])
    Point([2, 3])
    """

    __slots__ = ("_m",)

    def __init__(self, coords):
        self._m = _as_column(coords)

    # -- basic accessors --------------------------------------------------

    def dom(self) -> int:
        """Dimension $n$ of the ambient space."""
        return self._m.rows

    def coords(self):
        """The coordinates as a tuple of SymPy expressions."""
        return tuple(self._m[i, 0] for i in range(self.dom()))

    # -- affine arithmetic ------------------------------------------------

    def __add__(self, other):
        if isinstance(other, Vector):
            if self.dom() != other.dom():
                raise ValueError(
                    f"cannot translate a point in dimension {self.dom()} "
                    f"by a vector in dimension {other.dom()}"
                )
            return Point(self._m + other._m)
        if isinstance(other, Point):
            raise TypeError(
                "adding two points is undefined in affine geometry; "
                "use q - p to get the vector between them."
            )
        return NotImplemented

    def __radd__(self, other):
        # v + p  ==  p + v
        if isinstance(other, Vector):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Point):
            if self.dom() != other.dom():
                raise ValueError(
                    f"cannot subtract points in dimensions "
                    f"{self.dom()} and {other.dom()}"
                )
            return Vector(self._m - other._m)
        if isinstance(other, Vector):
            if self.dom() != other.dom():
                raise ValueError(
                    f"cannot translate a point in dimension {self.dom()} "
                    f"by a vector in dimension {other.dom()}"
                )
            return Point(self._m - other._m)
        return NotImplemented

    def __mul__(self, scalar):
        raise TypeError(
            "scalar multiplication of a point is undefined in affine geometry."
        )

    __rmul__ = __mul__

    # -- conversion -------------------------------------------------------

    def to_vector(self) -> Vector:
        """The vector from the origin $O$ to this point (depends on $O$!)."""
        return Vector(self._m)

    # -- display ----------------------------------------------------------

    def __repr__(self):
        return f"Point({list(self.coords())!r})"

    def _repr_latex_(self):
        # Horizontal, parentheses:  \left(1,\,2,\,3\right)
        parts = ",\\,".join(latex(e) for e in self.coords())
        return rf"\left({parts}\right)"

    def __eq__(self, other):
        if isinstance(other, Point):
            return self._m == other._m
        return NotImplemented

    def __hash__(self):
        return hash(("Point", self._m))

    # -- plotting ---------------------------------------------------------

    def plot(self, **style):
        """Return a renderable marker for this point.

        The returned object is a plot-class instance (``Point2D`` or
        ``Point3D``); display it by adding it to a canvas::

            canv = Canvas2D()
            canv.add(Point([1, 2]).plot())
            canv

        Extra keyword arguments (``color=``, ``label=``, ...) are forwarded to
        the plot class.
        """
        builder = _require_plot(Point, self.dom())
        return builder(self, **style)