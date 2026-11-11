## v0.3
- Function: eval, freeze (partial evaluation), diff, grad, integrate,
  surface_area (symbolic + numeric), volume (symbolic + numeric),
  plot (2d/3d dispatch), _repr_latex_ / display.
- graphics: Canvas2D / Canvas3D, Curve2D, ParametricCurve2D, Implicit2D,
  Surface, ParametricCurve3D, ParametricSurface3D,
  Point2D, Line2D, Vector2D, Polygon2D, Point3D, Line3D, Vector3D, Plane.
- Release: build_release.py produces dist/funlib-v<version>.zip (two files).
- Smoke test: test_smoke.py, run by build_release.py on both dev tree and
  merged files.
