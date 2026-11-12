"""
Graphics layer: canvases and the Renderable protocol. 

A plot class in this package (Curve2D, Surface, Point2D, ...) is a
Renderable: it has a .render(ax) method that draws into a matplotlib
Axes (2D) or Axes3D (3D). A Canvas is a figure + axes you add
Renderables to; it manages lifecycle (show, savefig, set_view).

No imports from `analysis` here: plot classes are duck-typed on the
Function interface (.args, .matrix, .eval, ._kind, ._shape_key) so that
this module stays free of the abstract layer and the release merge
(analysis.py, graphics.py) has no import cycle.
"""

import matplotlib.pyplot as plt

# ==== Renderable protocol ====
class Renderable:
    """
    Anything that can draw itself into a matplotlib Axes or Axes3D.

    Subclasses implement render(ax, **kw). The base class is empty;
    it exists so that Canvas.add can duck-type on .render and so that
    plot classes document the contract.
    """

    def render(self, ax, **kw):
        raise NotImplementedError # Renderable is just a stub for inheritance


# ==== Canvas2D ====
class Canvas2D:
    """
    A 2D matplotlib figure/axes to which Renderables can be added.

    Parameters
    ----------
    xlim, ylim : (float, float), optional
        Axis limits. If omitted, matplotlib auto-scales.
    figsize : (float, float), optional
    title : str, optional
    """

    def __init__(self, xlim=None, ylim=None, figsize=None, title=None,aspect='equal'):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._children = []
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        if title is not None:
            self.ax.set_title(title)
        self.ax.set_aspect(aspect)

    def add(self, plot_obj):
        """
        Add a Renderable (plot class or geometric object) to the canvas.

        Extra keyword arguments are forwarded to the renderable's
        render(ax, **kw). Returns the renderable, so calls can be
        chained or assigned.
        """
        if not hasattr(plot_obj, "render"):
            raise TypeError("expected a Renderable with .render(ax, **kw)")
        plot_obj.render(self.ax)
        self._children.append(plot_obj)
        return plot_obj

    def _ipython_display_(self):
        """
        Render the figure automatically when the canvas is the last
        expression of a Jupyter cell.
        """
        plt.show()


    def show(self):
        plt.show()

    def savefig(self, path, **kw):
        self.fig.savefig(path, **kw)

    def close(self):
        plt.close(self.fig)

class Canvas3D:
    """
    A 3D matplotlib figure with a grid of panels.

    With the defaults (nrows=ncols=1) there is a single panel and
    add(obj) draws on it, as before. With nrows=1, ncols=3, use
    add(obj, panel=0/1/2) to place each object in a different panel.

    Panel i is row i // ncols, column i % ncols (row-major).
    Each panel has its own box aspect, computed from the limits given.
    """

    def __init__(self, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),nrows=1, ncols=1, elev=30, azim=-60, roll=0,figsize=None, title=None, grid=True, axes=True):
        self.fig = plt.figure(figsize=figsize)
        self._axes = [
            self.fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            for i in range(nrows * ncols)
        ]
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

    def _panel_axes(self, panel):
        if not (0 <= panel < len(self._axes)):
            raise ValueError(f"panel {panel} out of range (0..{len(self._axes)-1})")
        return self._axes[panel]

    def add(self, plot_obj, panel=0):
        if not hasattr(plot_obj, "render"):
            raise TypeError("expected a Renderable with .render(ax)")
        ax = self._panel_axes(panel)
        plot_obj.render(ax)
        self._children.append(plot_obj)
        return plot_obj

    def set_panel_title(self, panel, title):
        self._panel_axes(panel).set_title(title)

    def set_view(self, elev=None, azim=None, roll=None):
        kw = {}
        if elev is not None: kw["elev"] = elev
        if azim is not None: kw["azim"] = azim
        if roll is not None: kw["roll"] = roll
        if kw:
            for ax in self._axes:
                ax.view_init(**kw)

    def _ipython_display_(self):
        plt.show()

    def show(self):
        plt.show()

    def savefig(self, path, **kw):
        self.fig.savefig(path, **kw)

    def close(self):
        plt.close(self.fig)