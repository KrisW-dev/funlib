# funlib

A small library for multivariate calculus in Jupyter notebooks, built on
SymPy and matplotlib. Two layers:

- **`analysis/`** — the abstract layer: `Function`, a sympy-backed
  representation of smooth maps `R^d → R^m`, with `eval`, `freeze`
  (partial evaluation), `diff`, `grad`, `integrate`, `surface_area`,
  `volume`, and automatic LaTeX display in Jupyter.
- **`graphics/`** — the graphics layer: `Canvas2D` / `Canvas3D`,
  plot classes for graphs, parametric curves and surfaces, implicit
  curves, and geometric objects (points, lines, vectors, polygons,
  planes). Depends on `analysis/` (never the reverse).

## Quick start

In a cell, with the two release files next to your notebook:

    from analysis import Function
    import graphics

    from sympy import symbols, sin, cos, pi
    x, y, r, theta = symbols("x y r theta", real=True)

    f = Function("f", [x, y], [[x**2 + y**2]])
    f                      # LaTeX display in Jupyter

    f.plot("2d", x_range=(-3, 3), y_range=(-3, 3))
    f.plot("3d", x_range=(-3, 3), y_range=(-3, 3))

## Development layout vs. release layout

The repo is a package tree (`analysis/`, `graphics/`) for development.
`build_release.py` merges it into two flat files, `analysis.py` and
`graphics.py`, and zips them for distribution:

    python build_release.py    # produces dist/funlib-v<version>.zip

The zip contains `analysis.py`, `graphics.py`, `README.txt` — put them
next to your notebook, then `from analysis import Function`.

## Smoke test

    python test_smoke.py

Run from the project root. `build_release.py` runs it twice: once against
the dev tree, once against the merged files in a temp directory.

## Requirements

sympy, numpy, matplotlib, scipy. Optional: ipympl (interactive 3D in
Jupyter via `%matplotlib widget`).

## Notes for students

- SymPy treats symbols as complex by default. Declare `real=True` (or
  `positive=True` for radii) so that `sqrt`, `Abs`, and `trigsimp`
  simplify as expected.
- Start Jupyter from the project root (where `analysis/` and `graphics/`
  live), then open notebooks from `tests/` or `notebooks/`. Imports work
  without `sys.path` fiddling.
