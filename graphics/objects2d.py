"""
Graphics layer: 2D geometric objects.

Point2D, Line2D, Vector2D, Circle2D, Polygon2D — plain matplotlib
artists wrapped as Renderables. They do not depend on `analysis` and
can be composed with function plots on the same Canvas2D:

    canvas = Canvas2D()
    canvas.add(Curve2D(f))
    canvas.add(Point2D(1, 1, color="red"))
    canvas.add(Vector2D((0, 0), (1, 2)))

All objects accept styling keyword arguments (color, linewidth,
linestyle, marker, ...) in their constructor and forward them to the
underlying matplotlib call in render().
"""

import numpy as np
from matplotlib.patches import Polygon #Circle

from .canvas import Renderable


# ==== Point2D ====
class Point2D(Renderable):
    """A single point (x, y) drawn as a marker."""

    def __init__(self, x, y, **style):
        self.x = x
        self.y = y
        style.setdefault("color", "black")
        style.setdefault("marker", "o")
        style.setdefault("zorder", 5)
        self.style = style

    def render(self, ax, **kw):
        merged = {**self.style, **kw}
        return ax.scatter([self.x], [self.y], **merged)


# ==== Line2D ====
class Line2D(Renderable):
    """
    Line segment through points p and q.

    If extend=True, the line is drawn across the current axes limits
    (useful for asymptotes and secants). If extend=False (default),
    only the segment between p and q is drawn.
    """

    def __init__(self, p, q, extend=False, **style):
        self.p = p
        self.q = q
        self.extend = extend
        style.setdefault("color", "black")
        style.setdefault("zorder", 4)
        self.style = style

    def render(self, ax, **kw):
        merged = {**self.style, **kw}
        if self.extend:
            x0, x1 = ax.get_xlim()
            (px, py), (qx, qy) = self.p, self.q
            dx, dy = qx - px, qy - py
            if dx == 0:
                ax.axvline(px, **merged)
                return None
            slope = dy / dx
            ys = [py + slope * (x - px) for x in (x0, x1)]
            return ax.plot([x0, x1], ys, **merged)
        return ax.plot([self.p[0], self.q[0]], [self.p[1], self.q[1]], **merged)


# ==== Vector2D ====
class Vector2D(Renderable):
    """Arrow from origin to tip (a free vector drawn at a point)."""

    def __init__(self, origin, tip, **style):
        self.origin = origin
        self.tip = tip
        style.setdefault("angles", "xy")
        style.setdefault("scale_units", "xy")
        style.setdefault("scale", 1)
        style.setdefault("zorder", 5)
        self.style = style

    def render(self, ax, **kw):
        merged = {**self.style, **kw}
        ox, oy = self.origin
        tx, ty = self.tip
        return ax.quiver(ox, oy, tx - ox, ty - oy, **merged)


# ==== Circle2D ====
# class Circle2D(Renderable):
#     """Circle of given radius centered at (cx, cy)."""

#     def __init__(self, center, radius, **style):
#         self.center = center
#         self.radius = radius
#         style.setdefault("fill", False)
#         style.setdefault("color", "black")
#         self.style = style

#     def render(self, ax, **kw):
#         merged = {**self.style, **kw}
#         cx, cy = self.center
#         circle = Circle((cx, cy), self.radius, **merged)
#         ax.add_patch(circle)
#         return circle

# ==== Polygon2D ====
class Polygon2D(Renderable):
    """Closed polygon through the given vertices (list of (x, y))."""

    def __init__(self, vertices, **style):
        self.vertices = list(vertices)
        style.setdefault("fill", False)
        style.setdefault("color", "black")
        self.style = style

    def render(self, ax, **kw):
        merged = {**self.style, **kw}
        patch = Polygon(self.vertices, closed=True, **merged)
        ax.add_patch(patch)
        return patch

