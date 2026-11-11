"""
Smoke test for analysis/ and graphics/.

Run with:  python tests/test_smoke.py

Exercises every public plot class, every geometric object, both canvases,
the registry dispatch, the display protocols, and the integration methods.
Any failure (exception or wrong numerical answer) is reported and the
script exits nonzero. No pytest dependency, no notebook needed.
"""

import sys
import numpy as np
from sympy import symbols, cos, sin, pi, Rational

from analysis import Function
import graphics
from graphics import (
    Canvas2D, Canvas3D,
    Curve2D, ParametricCurve2D, Implicit2D,
    Surface, ParametricCurve3D, ParametricSurface3D,
    Point2D, Line2D, Vector2D, Polygon2D,
    Point3D, Line3D, Vector3D, Plane,
)

x, y, z, t, u, v, r, theta, s, phi = symbols("x y z t u v r theta s phi", real=True)

failures = []

import signal

class Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise Timeout()

def check(name, fn, timeout=60):
    print(f"  ...  {name}", flush=True)
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        fn()
        print(f"  ok   {name}")
    except Timeout:
        failures.append((name, Timeout(f">{timeout}s")))
        print(f"  FAIL {name}: timeout after {timeout}s")
    except Exception as e:
        failures.append((name, e))
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# def check(name, fn):
#     try:
#         fn()
#         print(f"  ok   {name}")
#     except Exception as e:
#         failures.append((name, e))
#         print(f"  FAIL {name}: {type(e).__name__}: {e}")

def approx(a, b, tol=1e-6):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{a} != {b} (tol {tol})")

# ---- functions of each shape ----
f_scalar_1  = Function("f",    [x],    [[x**2]])                      # R -> R
f_scalar_2  = Function("g",    [x, y], [[x**2 + y**2]])               # R^2 -> R
f_curve_2   = Function("gamma",[t],    [[cos(t)], [sin(t)]])          # R -> R^2
f_curve_3   = Function("eta",  [t],    [[cos(t)], [sin(t)], [t]])     # R -> R^3
f_surf_3    = Function("sigma",[u, v], [[cos(u)*sin(v)], [sin(u)*sin(v)], [cos(v)]])  # R^2 -> R^3
f_const     = Function("h",    [u, v], [[cos(u)], [sin(u)], [0]])     # constant component

# ---- 2D plot classes ----
def t_curve2d():
    c = Canvas2D()
    c.add(f_scalar_1.plot("2d", x_range=(-2, 2), y_range=(0, 4)))
    c.close()
check("Curve2D", t_curve2d)

def t_paramcurve2d():
    c = Canvas2D(aspect="equal")
    c.add(f_curve_2.plot("2d", t_range=(0, 2*pi)))
    c.close()
check("ParametricCurve2D", t_paramcurve2d)

def t_implicit2d():
    c = Canvas2D()
    c.add(f_scalar_2.plot("2d", x_range=(-3, 3), y_range=(-3, 3), cmap="viridis"))
    c.close()
check("Implicit2D", t_implicit2d)

def t_mixed_2d():
    c = Canvas2D(aspect="equal")
    c.add(f_curve_2.plot("2d", color="C0"))
    c.add(Point2D(1, 1, color="red"))
    c.add(Vector2D((0, 0), (1, 0), color="green"))
    c.add(Line2D((0, 0), (1, 1), color="black"))
    c.close()
check("mixed 2D canvas", t_mixed_2d)

# ---- 3D plot classes ----
def t_surface():
    c = Canvas3D()
    c.add(f_scalar_2.plot("3d", x_range=(-3, 3), y_range=(-3, 3), cmap="viridis"))
    c.close()
check("Surface", t_surface)

def t_paramcurve3d():
    c = Canvas3D()
    c.add(f_curve_3.plot("3d", t_range=(0, 2*pi), color="C0"))
    c.close()
check("ParametricCurve3D", t_paramcurve3d)

def t_paramsurface3d():
    c = Canvas3D()
    c.add(f_surf_3.plot("3d", u_range=(0, 2*pi), v_range=(0, pi), cmap="inferno", alpha=0.6))
    c.close()
check("ParametricSurface3D", t_paramsurface3d)

def t_const_component():
    c = Canvas3D()
    c.add(f_const.plot("3d", u_range=(0, 2*pi), v_range=(-pi, pi), cmap="viridis"))
    c.close()
check("constant component (z=0)", t_const_component)

def t_plane():
    c = Canvas3D()
    c.add(Plane(p=(0, 0, 3), normal=(0, 0, 1), size=10, alpha=0.6))
    c.close()
check("Plane (flat, aspect)", t_plane)

def t_mixed_3d():
    c = Canvas3D()
    c.add(f_surf_3.plot("3d", u_range=(0, 2*pi), v_range=(0, pi), alpha=0.6))
    c.add(Plane(p=(0, 0, 0), normal=(0, 0, 1), size=6, alpha=0.4))
    c.add(Point3D(0, 0, 0, color="red"))
    c.add(Vector3D((0, 0, 0), (1, 1, 1), color="green"))
    c.add(Line3D((-1, 0, 0), (1, 0, 0), color="black"))
    c.close()
check("mixed 3D canvas", t_mixed_3d)

# ---- dispatch errors ----
def t_not_plottable():
    try:
        f_surf_3.plot("2d")   # R^2 -> R^3 is not 2d-plottable
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-plottable function")
check("dispatch rejects non-plottable", t_not_plottable)

def t_no_graphics():
    # if graphics were not imported, dispatch would raise "empty registry";
    # here it's imported, so just confirm plot("2d") on a scalar works
    f_scalar_2.plot("2d")
check("dispatch works after import graphics", t_no_graphics)

# ---- integration ----
def t_integrate_disk():
    Phi = Function("Phi", [r, theta], [[r*cos(theta)], [r*sin(theta)]])
    I = f_scalar_2.integrate(Phi, (0, 1), (0, 2*pi))
    approx(I, pi/2)
check("integrate: disk", t_integrate_disk)

def t_integrate_rect():
    g1 = Function("one", [x, y], [[1]])
    Phi = Function("Phi_rect", [s, t], [[s], [t]])
    I = g1.integrate(Phi, (0, 1), (0, 2))
    approx(I, 2)
check("integrate: rectangle", t_integrate_rect)

def t_integrate_ellipse():
    g1 = Function("one", [x, y], [[1]])
    Phi = Function("Phi_ell", [r, theta], [[2*r*cos(theta)], [3*r*sin(theta)]])
    I = g1.integrate(Phi, (0, 1), (0, 2*pi))
    approx(I, 6*pi)
check("integrate: ellipse (a=2,b=3)", t_integrate_ellipse)

# ---- surface area ----
def t_area_sphere():
    S = Function("S", [u, v], [[2*sin(u)*cos(v)], [2*sin(u)*sin(v)], [2*cos(u)]])
    A = S.surface_area((0, pi), (0, 2*pi))
    approx(A, 16*pi)
check("surface_area: sphere R=2", t_area_sphere)

def t_area_cylinder():
    C = Function("C", [u, v], [[2*cos(u)], [2*sin(u)], [v]])
    A = C.surface_area((0, 2*pi), (0, 3))
    approx(A, 12*pi)
check("surface_area: cylinder R=2,h=3", t_area_cylinder)

def t_area_torus():
    T = Function("T", [u, v], [[(2 + cos(v))*cos(u)], [(2 + cos(v))*sin(u)], [sin(v)]])
    A = T.surface_area((0, 2*pi), (0, 2*pi))
    approx(A, 8*pi**2)
check("surface_area: torus R=2,r=1", t_area_torus)

def t_area_numeric():
    # a=1 case: sqrt of sum of squares, no closed form; numeric path
    a = 1
    F = Function("F", [r, theta],
                 [[r*cos(theta)], [r*cos(theta)],
                  [1 + (1 - r**a)/2 + r**a*cos(theta)**2]])
    A = F.surface_area((0, 1), (0, 2*pi), numeric=True)
    if not (0 < float(A) < 100):
        raise AssertionError(f"numeric area out of range: {A}")
check("surface_area: numeric (a=1)", t_area_numeric)

def t_volume_sphere():
    B = Function("B", [r, phi, theta],
                 [[r*sin(phi)*cos(theta)], [r*sin(phi)*sin(theta)], [r*cos(phi)]])
    V = B.volume((0, 2), (0, pi), (0, 2*pi))
    approx(V, 32*pi/3)          # 4/3 * pi * 2^3
check("volume: sphere R=2", t_volume_sphere)

def t_freeze_surface():
    B = Function("B", [r, phi, theta],
                 [[r*sin(phi)*cos(theta)], [r*sin(phi)*sin(theta)], [r*cos(phi)]])
    S = B.freeze([(r, 2)])       # auto-named "B|_{r=2}"
    A = S.surface_area((0, pi), (0, 2*pi), numeric=False)
    approx(A, 16*pi)             # 4*pi*R^2 with R=2
check("freeze + surface_area: sphere R=2", t_freeze_surface)

def t_freeze_bad():
    B = Function("B", [r, phi, theta], [[r], [phi], [theta]])
    try:
        B.freeze([(x, 1)])       # x not an argument
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown frozen symbol")
check("freeze rejects unknown symbol", t_freeze_bad)

# ---- display protocols ----
def t_repr_latex():
    s = f_scalar_2._repr_latex_()
    assert isinstance(s, str) and "\\mathbb{R}" in s
check("Function._repr_latex_", t_repr_latex)

# ---- summary ----
print()
if failures:
    print(f"{len(failures)} failure(s):")
    for name, e in failures:
        print(f"  - {name}: {type(e).__name__}: {e}")
    sys.exit(1)
else:
    print("all checks passed")