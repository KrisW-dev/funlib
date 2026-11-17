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
from sympy import symbols, cos, sin, pi, Rational, sqrt

from analysis import Function, Vector, Point

import matplotlib
matplotlib.use("Agg")   # must precede any pyplot import

import graphics
from graphics import (Canvas2D, Canvas3D, 
    Curve2D, ParametricCurve2D, Implicit2D, Surface, ParametricCurve3D, ParametricSurface3D, Point2D, Line2D, Vector2D, Polygon2D,Polyline2D,
    Point3D, Line3D, Vector3D, Plane)


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

def approx(a, b, tol=1e-6):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{a} != {b} (tol {tol})")

# ---- geometry ----

assert isinstance(Vector([3, 4]).plot(), Vector2D)
assert isinstance(Vector([1, 1]).plot(origin=Point([2, 2])), Vector2D)
assert isinstance(Vector([1, 0, 0]).plot(origin=[1, 1, 1]), Vector3D)
assert isinstance(Point([1, 2]).plot(), Point2D)
assert isinstance(Point([1, 2, 3]).plot(), Point3D)

# ---- functions of each shape ----
f_scalar_1  = Function("f",    [x],    [[x**2]])                      # R -> R
f_scalar_2  = Function("g",    [x, y], [[x**2 + y**2]])               # R^2 -> R
f_curve_2   = Function("gamma",[t],    [[cos(t)], [sin(t)]])          # R -> R^2
f_curve_3   = Function("eta",  [t],    [[cos(t)], [sin(t)], [t]])     # R -> R^3
f_surf_3    = Function("sigma",[u, v], [[cos(u)*sin(v)], [sin(u)*sin(v)], [cos(v)]])  # R^2 -> R^3
f_const     = Function("h",    [u, v], [[cos(u)], [sin(u)], [0]])     # constant component

# ---- 2D plot classes ----
def t_objects_accept_sympy():
    from sympy import sqrt
    c = Canvas2D(aspect="equal")
    c.add(Point2D((sqrt(2), sqrt(2))))
    c.add(Vector2D((0, 0), (sqrt(2), 1)))
    c.add(Line2D((0, 0), (1, sqrt(3))))
    c.add(Polyline2D([(0, 0), (sqrt(2), 1), (1, sqrt(3))]))
    c.add(Polygon2D([(0, 0), (sqrt(2), 0), (0, sqrt(2))]))
    c.close()
check("geometric objects accept SymPy coordinates", t_objects_accept_sympy)

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
    c.add(Point2D((1, 1), color="red"))
    c.add(Vector2D((0, 0), (1, 0), color="green"))
    c.add(Line2D((0, 0), (1, 1), color="black"))
    c.add(Polygon2D([(0, 0), (1, 0), (1, 1)]))
    c.close()
check("mixed 2D canvas", t_mixed_2d)

def t_canvas2d_single():
    c = Canvas2D(xlim=(-1, 1), ylim=(-1, 1), title="f")
    c.add(f_scalar_2.plot("2d", x_range=(-1, 1), y_range=(-1, 1)))
    c.close()
check("Canvas2D single-panel (backward compat)", t_canvas2d_single)

def t_canvas2d_multipanel():
    c = Canvas2D(nrows=1, ncols=2, figsize=(8, 4))
    c.add(f_scalar_2.plot("2d", x_range=(-1, 1), y_range=(-1, 1)), panel=0)
    c.add(f_scalar_1.plot("2d", x_range=(-1, 1), y_range=(-1, 1)), panel=1)
    c.set_panel_title(0, "g")
    c.set_panel_title(1, "f")
    c.close()
check("Canvas2D multi-panel", t_canvas2d_multipanel)

def t_canvas2d_panel_out_of_range():
    c = Canvas2D(nrows=1, ncols=2)
    try:
        c.add(f_scalar_2.plot("2d", x_range=(-1, 1), y_range=(-1, 1)), panel=5)
    except ValueError:
        c.close()
        return
    raise AssertionError("expected ValueError for panel out of range")
check("Canvas2D panel out of range", t_canvas2d_panel_out_of_range)

# canvas idiom
canv = Canvas2D()
canv.add(Vector([3, 4]).plot(origin=Point([1, 1])))
assert canv is not None

# ---- 3D plot classes ----
def t_objects3d_accept_sympy():
    from sympy import sqrt
    c = Canvas3D()
    c.add(Point3D((sqrt(2), 0, sqrt(3))))
    c.add(Vector3D((0, 0, 0), (sqrt(2), 1, sqrt(3))))
    c.add(Line3D((0, 0, 0), (1, sqrt(2), 1)))
    c.add(Plane(p=(0, 0, sqrt(2)), normal=(0, 0, 1), size=sqrt(2)))
    c.close()
check("3D objects accept SymPy coordinates", t_objects3d_accept_sympy)

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
check("Plane (pinned limits)", t_plane)

def t_mixed_3d():
    c = Canvas3D()
    c.add(f_surf_3.plot("3d", u_range=(0, 2*pi), v_range=(0, pi), alpha=0.6))
    c.add(Plane(p=(0, 0, 0), normal=(0, 0, 1), size=6, alpha=0.4))
    c.add(Point3D((0, 0, 0), color="red"))
    c.add(Vector3D((0, 0, 0), (1, 1, 1), color="green"))
    c.add(Line3D((-1, 0, 0), (1, 0, 0), color="black"))
    c.close()
check("mixed 3D canvas", t_mixed_3d)

def t_multipanel():
    c = Canvas3D(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2),
                 nrows=1, ncols=3, axes=False)
    for i in range(3):
        c.add(Point3D((0, 0, 0), color="red"), panel=i)
        c.set_panel_title(i, f"p{i}")
    c.close()
check("Canvas3D multi-panel", t_multipanel)

def t_panel_out_of_range():
    c = Canvas3D(nrows=1, ncols=3)
    try:
        c.add(Point3D((0, 0, 0)), panel=5)
    except ValueError:
        c.close()
        return
    raise AssertionError("expected ValueError for panel out of range")
check("Canvas3D panel out of range", t_panel_out_of_range)

def t_canvas_grid_axes():
    c = Canvas3D(grid=False)
    c.add(Point3D((0, 0, 0)), panel=0)
    c.close()
    c = Canvas3D(axes=False)
    c.add(Point3D((0, 0, 0)), panel=0)
    c.close()
check("Canvas3D grid/axes flags", t_canvas_grid_axes)

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

def t_integrate_1d_poly():
    # ∫₀¹ t² dt = 1/3
    f = Function("f", [t], [t**2])
    approx(f.integrate_1d((0, 1)), Rational(1, 3))
check("integrate_1d: polynomial", t_integrate_1d_poly)

def t_integrate_1d_circle_length():
    # unit circle: |C'| = 1, length = 2π
    circle = Function("c", [t], [cos(t), sin(t)])
    speed = Function("|c'|", [t], [sqrt(sum(c.diff(t)**2 for c in circle.coord_matrix.col(0)))])
    approx(speed.integrate_1d((0, 2*pi)), 2*pi)
check("integrate_1d: unit circle length = 2π", t_integrate_1d_circle_length)

def t_integrate_1d_numeric():
    # same circle length, numeric path
    circle = Function("c", [t], [cos(t), sin(t)])
    speed = Function("|c'|", [t], [sqrt(sum(c.diff(t)**2 for c in circle.coord_matrix.col(0)))])
    approx(speed.integrate_1d((0, 2*pi), numeric=True), 2*pi)
check("integrate_1d: numeric matches exact", t_integrate_1d_numeric)

def t_integrate_1d_precondition():
    # a vector-valued function is not a scalar R -> R
    vec = Function("v", [t], [cos(t), sin(t)])
    try:
        vec.integrate_1d((0, 1))
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-scalar function")
check("integrate_1d: rejects non-scalar", t_integrate_1d_precondition)

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

# ---- line integrals of vector fields ----
def t_line_rotation_circle():
    # F = (-y, x), unit circle → 2π
    F = Function("F", [x, y], [-y, x])
    gamma = Function("gamma", [t], [cos(t), sin(t)])
    approx(F.integrate_along(gamma, (0, 2*pi)), 2*pi)
check("integrate_along: rotation, unit circle", t_line_rotation_circle)

def t_line_conservative_closed():
    # F = (x, y) = ∇(½(x²+y²)), closed curve → 0
    F = Function("F", [x, y], [x, y])
    gamma = Function("gamma", [t], [cos(t), sin(t)])
    approx(F.integrate_along(gamma, (0, 2*pi)), 0)
check("integrate_along: conservative, closed", t_line_conservative_closed)

def t_line_parabola_poly():
    # F = (-y, x), P(t) = (t²-3, 2t+1), t ∈ [-2, 3] → -175/3
    F = Function("F", [x, y], [-y, x])
    P = Function("P", [t], [t**2 - 3, 2*t + 1])
    approx(F.integrate_along(P, (-2, 3)), Rational(-175, 3))
check("integrate_along: parabola (exact, signed)", t_line_parabola_poly)

def t_line_3d_helix():
    # G = (-y, x, z), γ(t) = (cos t, sin t, t), t ∈ [0, 2π] → 2π + 2π²
    G = Function("G", [x, y, z], [-y, x, z])
    gamma = Function("gamma", [t], [cos(t), sin(t), t])
    approx(G.integrate_along(gamma, (0, 2*pi)), 2*pi + 2*pi**2)
check("integrate_along: 3D helix", t_line_3d_helix)

def t_line_orientation_reversal():
    # reversing t_range negates
    F = Function("F", [x, y], [-y, x])
    gamma = Function("gamma", [t], [cos(t), sin(t)])
    I_fwd = F.integrate_along(gamma, (0, 2*pi))
    I_rev = F.integrate_along(gamma, (2*pi, 0))
    approx(I_fwd + I_rev, 0)
check("integrate_along: orientation reversal", t_line_orientation_reversal)

def t_line_mismatched_dim():
    # F: R² → R², γ: R → R³ — must raise
    F = Function("F", [x, y], [-y, x])
    gamma = Function("gamma", [t], [cos(t), sin(t), t])
    try:
        F.integrate_along(gamma, (0, 2*pi))
    except ValueError:
        return
    raise AssertionError("expected ValueError for dimension mismatch")
check("integrate_along: dimension mismatch rejected", t_line_mismatched_dim)

def t_line_numeric_agrees():
    # numeric path must match the exact -175/3
    F = Function("F", [x, y], [-y, x])
    P = Function("P", [t], [t**2 - 3, 2*t + 1])
    approx(F.integrate_along(P, (-2, 3), numeric=True), Rational(-175, 3))
check("integrate_along: numeric matches exact", t_line_numeric_agrees)

# ---- flux integrals of vector fields ----
def t_flux_sphere_outward():
    # F = (x,y,z), unit sphere, outward → 4π (divergence theorem: 3 · 4π/3)
    F = Function("F", [x, y, z], [x, y, z])
    S = Function("S", [u, v], [sin(u)*cos(v), sin(u)*sin(v), cos(u)])
    approx(F.integrate_over(S, (0, pi), (0, 2*pi)), 4*pi)
check("integrate_over: radial field, unit sphere", t_flux_sphere_outward)

def t_flux_orientation_reversal():
    # reversing one parameter negates the flux
    F = Function("F", [x, y, z], [x, y, z])
    S = Function("S", [u, v], [sin(u)*cos(v), sin(u)*sin(v), cos(u)])
    I_fwd = F.integrate_over(S, (0, pi), (0, 2*pi))
    I_rev = F.integrate_over(S, (pi, 0), (0, 2*pi))
    approx(I_fwd + I_rev, 0)
check("integrate_over: orientation reversal", t_flux_orientation_reversal)

def t_flux_gauss_cylinder():
    # F = (x, y, 0), lateral surface of cylinder R=1, h∈[0,3] → 6π
    F = Function("F", [x, y, z], [x, y, 0])
    S = Function("S", [u, v], [cos(u), sin(u), v])
    approx(F.integrate_over(S, (0, 2*pi), (0, 3)), 6*pi)
check("integrate_over: cylinder lateral flux", t_flux_gauss_cylinder)

def t_flux_preconditions():
    F2 = Function("F2", [x, y], [-y, x])
    S = Function("S", [u, v], [sin(u)*cos(v), sin(u)*sin(v), cos(u)])
    try:
        F2.integrate_over(S, (0, pi), (0, 2*pi))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for F not R^3 -> R^3")

    F3 = Function("F3", [x, y, z], [x, y, z])
    S_bad = Function("Sb", [u, v], [u, v])
    try:
        F3.integrate_over(S_bad, (0, 1), (0, 1))
    except ValueError:
        return
    raise AssertionError("expected ValueError for S not R^2 -> R^3")
check("integrate_over: preconditions rejected", t_flux_preconditions)

def t_flux_numeric_agrees():
    F = Function("F", [x, y, z], [x, y, z])
    S = Function("S", [u, v], [sin(u)*cos(v), sin(u)*sin(v), cos(u)])
    approx(F.integrate_over(S, (0, pi), (0, 2*pi), numeric=True), 4*pi)
check("integrate_over: numeric matches exact", t_flux_numeric_agrees)\

def t_flux_across_circle():
    # F = (x/2, y/2), unit circle → π (unit disc area)
    F = Function("F", [x, y], [x/2, y/2])
    gamma = Function("gamma", [t], [cos(t), sin(t)])
    approx(F.flux_across(gamma, (0, 2*pi)), pi)
check("flux_across: radial field, unit circle", t_flux_across_circle)

def t_flux_across_orientation():
    F = Function("F", [x, y], [x/2, y/2])
    gamma = Function("gamma", [t], [cos(t), sin(t)])
    I_fwd = F.flux_across(gamma, (0, 2*pi))
    I_rev = F.flux_across(gamma, (2*pi, 0))
    approx(I_fwd + I_rev, 0)
check("flux_across: orientation reversal", t_flux_across_orientation)

def t_flux_across_preconditions():
    F2 = Function("F2", [x, y, z], [x, y, z])     # R^3 → R^3, wrong
    gamma = Function("gamma", [t], [cos(t), sin(t)])
    try:
        F2.flux_across(gamma, (0, 2*pi))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for F not R^2 -> R^2")

    F = Function("F", [x, y], [x/2, y/2])
    gamma_bad = Function("gb", [t], [cos(t), sin(t), t])
    try:
        F.flux_across(gamma_bad, (0, 2*pi))
    except ValueError:
        return
    raise AssertionError("expected ValueError for gamma not R -> R^2")
check("flux_across: preconditions rejected", t_flux_across_preconditions)

def t_flux_across_numeric():
    F = Function("F", [x, y], [x/2, y/2])
    gamma = Function("gamma", [t], [cos(t), sin(t)])
    approx(F.flux_across(gamma, (0, 2*pi), numeric=True), pi)
check("flux_across: numeric matches exact", t_flux_across_numeric)

# ---- div and curl ----
def t_div_linear():
    F = Function("F", [x, y], [x, y])
    approx(F.div().eval(0, 0), 2)
check("div: identity field, R^2", t_div_linear)

def t_div_area_field():
    F = Function("F", [x, y], [x/2, y/2])
    approx(F.div().eval(0, 0), 1)
check("div: (x/2, y/2)", t_div_area_field)

def t_div_rotation():
    F = Function("F", [x, y], [-y, x])
    approx(F.div().eval(1, 1), 0)
check("div: rotation field is divergence-free", t_div_rotation)

def t_div_3d():
    F = Function("F", [x, y, z], [x, y, z])
    approx(F.div().eval(0, 0, 0), 3)
check("div: identity field, R^3", t_div_3d)

def t_div_components():
    # F = (y, x) → ∂y/∂x + ∂x/∂y = 0; wrong pairing would give 2
    F = Function("F", [x, y], [y, x])
    approx(F.div().eval(1, 1), 0)
check("div: component/coordinate pairing", t_div_components)

def t_div_preconditions():
    F = Function("F", [t], [cos(t), sin(t)])
    try:
        F.div()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-square field")

    F = Function("F", [x, y], [x, y, x + y])
    try:
        F.div()
    except ValueError:
        return
    raise AssertionError("expected ValueError for F: R^2 -> R^3")
check("div: preconditions rejected", t_div_preconditions)

def t_curl_rotation():
    F = Function("F", [x, y], [-y, x])
    approx(F.curl().eval(0, 0), 2)
check("curl: rotation field", t_curl_rotation)

def t_curl_gradient():
    F = Function("F", [x, y], [x, y])
    approx(F.curl().eval(1, 1), 0)
check("curl: gradient field is curl-free", t_curl_gradient)

def t_curl_components():
    # F = (y², x²) → curl = 2x - 2y; sign flips at (1,0) vs (0,1)
    F = Function("F", [x, y], [y**2, x**2])
    approx(F.curl().eval(1, 0), 2)
    approx(F.curl().eval(0, 1), -2)
check("curl: component pairing and sign", t_curl_components)

def t_curl_preconditions():
    F = Function("F", [x, y, z], [x, y, z])
    try:
        F.curl()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 3D field")

    F = Function("F", [x, y], [x + y])
    try:
        F.curl()
    except ValueError:
        return
    raise AssertionError("expected ValueError for scalar function")
check("curl: preconditions rejected", t_curl_preconditions)

# ---- SymPy bounds ----

def t_vectorfield2d_along_sympy_bounds():
    # t_range with sympy.pi bound — regression test for np.linspace float coercion
    F = Function("F", [x, y], [-y, x])
    gamma = Function("gamma", [t], [cos(t), sin(t)])
    c = Canvas2D(aspect="equal")
    c.add(F.plot("2d", along=gamma, t_range=(0, 2*pi), n_samples=20))
    c.close()
check("VectorField2D along= with SymPy t_range", t_vectorfield2d_along_sympy_bounds)

# ---- display protocols ----
def t_repr_latex():
    s = f_scalar_2._repr_latex_()
    assert isinstance(s, str) and "\\mathbb{R}" in s
check("Function._repr_latex_", t_repr_latex)

# ---- canvas base ----

def t_canvas_base_shared():
    """Structural check: Canvas2D/3D share methods from a common base.

    Only runs against the package layout; skipped in the merged dist
    where the private _CanvasBase may not be exposed.
    """
    try:
        from graphics.canvas import _CanvasBase
    except (ImportError, ModuleNotFoundError):
        return   # merged dist: skip, structural guard is dev-only
    c2 = Canvas2D(); c3 = Canvas3D()
    try:
        assert isinstance(c2, _CanvasBase)
        assert isinstance(c3, _CanvasBase)
        assert Canvas2D.add is _CanvasBase.add
        assert Canvas3D.add is _CanvasBase.add
        assert Canvas2D.set_panel_title is _CanvasBase.set_panel_title
        assert Canvas3D.set_panel_title is _CanvasBase.set_panel_title
        assert Canvas2D._panel_axes is _CanvasBase._panel_axes
        assert Canvas3D._panel_axes is _CanvasBase._panel_axes
        assert not hasattr(_CanvasBase, "set_view")
    finally:
        c2.close(); c3.close()
check("Canvas2D/3D share _CanvasBase", t_canvas_base_shared)

# ---- figure leak guard

def t_canvas_close_no_leak():
    import matplotlib.pyplot as plt
    n0 = len(plt.get_fignums())
    for _ in range(3):
        c = Canvas2D()
        c.add(f_scalar_2.plot("2d", x_range=(-1, 1), y_range=(-1, 1)))
        c.close()
    n1 = len(plt.get_fignums())
    if n1 != n0:
        raise AssertionError(f"figure leak: {n0} -> {n1}")
check("Canvas2D close() releases figures", t_canvas_close_no_leak)

# ---- summary ----
print()
if failures:
    print(f"{len(failures)} failure(s):")
    for name, e in failures:
        print(f"  - {name}: {type(e).__name__}: {e}")
    sys.exit(1)
else:
    print("all checks passed")

