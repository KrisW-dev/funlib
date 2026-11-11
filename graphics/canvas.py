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


# ==== Canvas3D ====
class Canvas3D:
    """
    A 3D matplotlib figure/axes to which Renderables can be added.

    Parameters
    ----------
    xlim, ylim, zlim : (float, float), optional
    elev, azim, roll : float
        Initial camera angles. Use set_view() to change after creation.
    figsize : (float, float), optional
    title : str, optional
    """

    def __init__(self, xlim=None, ylim=None, zlim=None, elev=30, azim=-60, roll=0,box_aspect=None, figsize=None, title=None):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._children = []
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        if zlim is not None:
            self.ax.set_zlim(zlim)
        if title is not None:
            self.ax.set_title(title)
        # Remember whether the user pinned the box aspect.
        self._explicit_box_aspect = box_aspect is not None
        if box_aspect is not None:
            self.ax.set_box_aspect(box_aspect)
        self.set_view(elev=elev, azim=azim, roll=roll)

    def add(self, plot_obj):
        if not hasattr(plot_obj, "render"):
            raise TypeError("expected a Renderable with .render(ax, **kw)")
        plot_obj.render(self.ax)
        self._children.append(plot_obj)
        self._update_box_aspect()
        return plot_obj

    def _update_box_aspect(self):
        if self._explicit_box_aspect:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        z0, z1 = self.ax.get_zlim()
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        m = max(dx, dy, dz)
        if m > 0:
            eps = 1e-2
            self.ax.set_box_aspect((max(dx/m, eps), max(dy/m, eps), max(dz/m, eps)))

    def set_view(self, elev=None, azim=None, roll=None):
        """Update the camera angle after the canvas has been created."""
        kw = {}
        if elev is not None:
            kw["elev"] = elev
        if azim is not None:
            kw["azim"] = azim
        if roll is not None:
            kw["roll"] = roll
        if kw:
            self.ax.view_init(**kw)

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