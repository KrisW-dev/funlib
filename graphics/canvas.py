"""
Graphics layer: canvases and the Renderable protocol.
...
"""

import matplotlib.pyplot as plt

# ==== Just for for export ====

def _to_float(p):
    """Coerce a point to a tuple of floats, accepting SymPy expressions."""
    return tuple(float(c) for c in p)

def _num(x):
    """Coerce a scalar to float, accepting SymPy expressions."""
    return float(x)

# ==== Renderable protocol ====
class Renderable:
    """..."""

    def render(self, ax, **kw):
        raise NotImplementedError

# ==== Shared canvas machinery ====
class _CanvasBase:
    """
    Common canvas machinery: figure + list of axes (panels),
    panel addressing (row-major) 
    
    add/show/savefig/close.

    Not part of the public API — use Canvas2D or Canvas3D.

    Subclasses must set, before returning from __init__:
        self.fig      : matplotlib Figure
        self._axes    : list of Axes (or Axes3D), row-major
        self._ncols   : int
        self.ax       : self._axes[0]   (convenience for single-panel use)
        self._children: list
    """

    def _panel_axes(self, panel):
        if not (0 <= panel < len(self._axes)):
            raise ValueError(
                f"panel {panel} out of range (0..{len(self._axes)-1})"
            )
        return self._axes[panel]

    def add(self, plot_obj, panel=0):
        """
        Add a Renderable to panel `panel` (row-major index; default 0).

        Returns the renderable, so calls can be chained or assigned.
        """
        if not hasattr(plot_obj, "render"):
            raise TypeError("expected a Renderable with .render(ax)")
        ax = self._panel_axes(panel)
        plot_obj.render(ax)
        self._children.append(plot_obj)
        return plot_obj

    def set_panel_title(self, panel, title):
        self._panel_axes(panel).set_title(title)

    def _ipython_display_(self):
        self.show()

    def show(self):
        plt.show()

    def savefig(self, path, **kw):
        self.fig.savefig(path, **kw)

    def close(self):
        plt.close(self.fig)

# ==== Canvas2D ====
class Canvas2D(_CanvasBase):
    """
    A 2D matplotlib figure with a grid of panels.

    With the defaults (nrows=ncols=1) there is a single panel and
    add(obj) draws on it, as before. With nrows=1, ncols=3, use
    add(obj, panel=0/1/2) to place each object in a different panel.

    Panel i is row i // ncols, column i % ncols (row-major).
    xlim/ylim/title apply to all panels (the shared default view).
    """

    def __init__(self, xlim=None, ylim=None, nrows=1, ncols=1,figsize=None, title=None, aspect='equal'):
        self.fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        self._axes = list(axes.flat)          # row-major
        self._ncols = ncols
        self.ax = self._axes[0]
        self._children = []

        for ax in self._axes:
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_aspect(aspect)

        if title is not None:
            self.ax.set_title(title)

# ==== Canvas3D ====
class Canvas3D(_CanvasBase):
    """
    A 3D matplotlib figure with a grid of panels.
    ...
    """

    def __init__(
            self, 
            xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
            nrows=1, ncols=1, 
            elev=30, azim=-60, roll=0,
            figsize=None, title=None, grid=True, axes=True
        ):
        self.fig = plt.figure(figsize=figsize)
        self._axes = [self.fig.add_subplot(nrows, ncols, i + 1, projection="3d")for i in range(nrows * ncols)]
        self._ncols = ncols
        self.ax = self._axes[0]
        self._children = []

        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        dz = zlim[1] - zlim[0]
        m = max(dx, dy, dz)
        box_aspect = (dx/m, dy/m, dz/m) if m > 0 else (1, 1, 1)

        for ax in self._axes:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_box_aspect(box_aspect)
            if axes:
                ax.grid(grid)
            else:
                ax.set_axis_off()

        if title is not None:
            self.ax.set_title(title)
        self.set_view(elev=elev, azim=azim, roll=roll)

    def set_view(self, elev=None, azim=None, roll=None):
        """3D only: set the camera for every panel."""
        kw = {}
        if elev is not None: kw["elev"] = elev
        if azim is not None: kw["azim"] = azim
        if roll is not None: kw["roll"] = roll
        if kw:
            for ax in self._axes:
                ax.view_init(**kw)


