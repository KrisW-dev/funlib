"""
Abstract layer: multivariate functions over sympy, for Jupyter notebooks (but not only)

No plotting dependencies. Importing this module does not require
matplotlib or numpy. Requires sympy. Requires IPython only for
Function.display(), which falls back to a plain-text rendering if
IPython is absent. 
"""

from sympy import Matrix, N, latex, shape, integrate as sp_integrate, expand, trigsimp, lambdify
from sympy.core.symbol import Symbol
#from IPython.display import display as _display, Latex

try:
    from IPython.display import display as _display, Latex
    _HAVE_IPYTHON = True
except ImportError:
    _HAVE_IPYTHON = False


# ==== Plot dispatch registry ====
# Populated by the graphics layer at import time. Keys are
# (_kind, dom, m); values are {"2d": cls, "3d": cls}.
# Empty until `import graphics` runs.

_PLOT_REGISTRY = {}

def dispatch(f, dim, **kwargs):
    """Instantiate the plot class registered for (f, dim)."""
    if not _PLOT_REGISTRY:
        raise ValueError(
            "plot dispatch table is empty — did you `import graphics`?"
        )
    key = f._shape_key()
    targets = _PLOT_REGISTRY.get(key)
    if not targets or targets.get(dim) is None:
        raise ValueError(f"{f._name} is not {dim}-plottable")
    return targets[dim](f, **kwargs)

# ==== Utils ====

def duplicates(l):
    seen = set()
    for x in l:
        if x in seen:
            return True
        seen.add(x)
    return False

# ==== Analysis ====

class Function:
    """
    Matrix-valued smooth functions $\\mathbb{R}^d \\rightarrow \\mathbb{R}^{m \\times n}$. 
    Vector-valued functions are identified with functions $\\mathbb{R}^d \\rightarrow \\mathbb{R}^{m \\times 1}$.
    """

    def __init__(self, name, args, coords):
        """
        Constructs an object representing a function. Takes the arguments

        - name: for the LaTeX display

        - args: the argument vector as list of symbols

        - coords: the coordinate functions as list of lists of symbolic expressions
        """
        if not isinstance(args, list):
            raise ValueError("Arguments have to be in a list, even if there is only one")
        if duplicates(args):
            raise ValueError("Duplicate argument symbols")
        argset = set(args)
        cm = Matrix(coords)
        m, n = shape(cm)
        for i in range(m):
            for j in range(n):
                if not cm[i, j].free_symbols <= argset:
                    raise ValueError("Free variables in coordinate function")
        """Instance variables:"""
        self._name = name
        self._args = args
        self._coord_matrix = cm

    @property
    def args(self):
        """The argument symbols of self."""
        return self._args

    @property
    def coord_matrix(self):
        """Matrix of self's coordinate functions"""
        return self._coord_matrix

    def dom(self):
        """The dimension of self's domain"""
        return len(self.args)

    def cod(self):
        """The shape (m, n) of self's matrix of coordinate functions."""
        return shape(self.coord_matrix)

    def _kind(self):
        """
        'scalar'  : $1 x 1$
        'vector'  : $m x 1, m > 1$  (canonical column form)
        'other'     : anything else; not diffable in our setting)
        """
        m, n = shape(self.coord_matrix)
        if n != 1:
            return "other"
        return "scalar" if m == 1 else "vector"

    def _shape_key(self):
        """Registry key: (_kind, dom, m)."""
        m, n = shape(self.coord_matrix)
        return (self._kind(), self.dom(), m)

    # ---- evaluation ----
    def eval(self, *vals, precision=None):
        """
        Precondition: the number of arguments must be the same as the dimension of self's domain.

        Returns the value of self evaluated on the arguments *vals. This value is symbolic by default and a float if 
        precision is not $\\mathtt{None}$
        """
        if len(vals) != len(self.args):
            raise ValueError("Wrong number of arguments")
        sdict = dict(zip(self.args, vals))
        m, n = shape(self.coord_matrix)
        if (m, n) == (1, 1):
            c = self.coord_matrix[0, 0].subs(sdict)
            return c if precision is None else N(c, precision)
        M = self.coord_matrix[:, :].subs(sdict)
        return M if precision is None else N(M, precision)

    def freeze(self, argvals, name=None):
        """
        Partial evaluation: fix some of self's arguments to values.

        Preconditions:

        (i) argvals is a list of pairs (symbol, value);

        (ii) every symbol in argvals is among self's args, with no
            duplicates.

        (iii) at least one argument of self is left free (freezing all
            arguments would give a point, not a function).

        Returns a new Function in the remaining arguments. If name is
        None, the name is auto-generated as "<self._name>|<sym>=<val>,...",
        which renders in LaTeX as a restriction of self.
        """
        if not argvals:
            raise ValueError("argvals must be a non-empty list of (symbol, value) pairs")
        syms = [s for (s, _) in argvals]
        if duplicates(syms):
            raise ValueError("Duplicate symbols in argvals")
        if not set(syms) <= set(self.args):
            raise ValueError("Frozen symbols must be among self's arguments")

        newargs = [x for x in self.args if x not in set(syms)]
        if not newargs:
            raise ValueError("freeze() must leave at least one argument free")

        if name is None:
            frozen = ",".join(f"{latex(s)}={latex(v)}" for (s, v) in argvals)
            name = f"{self._name}|_{{{frozen}}}"

        sdict = dict(argvals)
        M = self.coord_matrix[:, :].subs(sdict)
        return Function(name, newargs, M)

    # ---- calculus ----
    def diff(self):
        """
        Precondition: self must be a vector-valued function $f: \\mathbb{R}^d \\rightarrow \\mathbb{R}^m$ for some 
        $d,n \\in \\mathbb{R}$

        Returns self's differential (a.k.a. *total derivative*) 
        $D_f: \\mathbb{R}^d \\rightarrow \\mathbb{R}^{m \\times d}$ (notice that it is a new function).
        """
        if self._kind() == "other":
            raise ValueError("diff() is for m x 1 matrices (scalar or vector functions)")
        return Function(
            "D_" + self._name,
            self.args,
            [[c.diff(s) for s in self.args] for c in self.coord_matrix.col(0)],
        )

    def grad(self):
        """
        Precondition: self must be scalar-valued function $f: \\mathbb{R}^d \\rightarrow \\mathbb{R}$ for some 
        $d \\in \\mathbb{R}$

        Returns self's gradient $\\nabla_f: \\mathbb{R}^d \\rightarrow \\mathbb{R}^d$, i.e. the vector 
        field obtained by transposing self's differential $D_f: \\mathbb{R}^d \\rightarrow \\mathbb{R}^{1 \\times d}$ 
        (notice that it is a new function).
        """
        if self._kind() != "scalar":
            raise ValueError("grad() is only for scalar functions; use diff() otherwise")
        D = self.diff()
        return Function("\\nabla_" + self._name, self.args, D.coord_matrix.T.tolist())

    def _integrate_numeric(self, Phi, s_range, t_range):
        import numpy as np
        from scipy.integrate import dblquad
        s, t = Phi.args
        xs = Phi.coord_matrix[0, 0]
        ys = Phi.coord_matrix[1, 0]
        detJ = Phi.diff().coord_matrix.det()
        integrand_expr = self.coord_matrix[0, 0].subs({self.args[0]: xs, self.args[1]: ys}) * detJ
        f = lambdify((s, t), integrand_expr, "numpy")
        def integrand(t, s):
            return float(f(s, t))
        result, _ = dblquad(integrand,s_range[0], s_range[1],lambda _: t_range[0], lambda _: t_range[1])
        return result

    def integrate(self, Phi, s_range, t_range, numeric=False):
        """
        Preconditions: 
        
        $(i)$ self must be a scalar-valued function $f: \\mathbb{R}^2 \\rightarrow \\mathbb{R}$

        $(ii)$ the parametrisation $\\Phi: \\mathbb{R}^2 \\rightarrow \\mathbb{R}^2$ of the domain of integration
        must be *orientation-preserving*,
        i.e. $det(D_{\\Phi}) \\geq 0$ on $\\mathbf{D} = [s_0,s_1] \\times [t_0,t_1]$. Notice that
        given the precondition no absolute value is taken
        (polar, cylindrical, spherical, and all the usual textbook parametrizations satisfy this on their standard 
        domains).
        
        Returns the integral
        $$
            \\int_\\mathbf{D} \\, f \\, dx \\, dy = \\int_\\mathbf{D} f(\\Phi(s,t)) · det(D_{\\Phi}(s,t)) \\, dt \\, ds
        $$
        
        The value here is by default a SymPy expression (the exact integral, or an unevaluated
        integral if SymPy cannot do it). **Warning**: performance can be a bottleneck, even if SymPy is capable
        of providing an exact value. If `numeric = True`, the method returns a float.
        """
        # if self._kind() != "scalar" or self.dom() != 2:
        #     raise ValueError("integrate() needs a scalar function R^2 -> R")
        # if Phi.dom() != 2 or Phi.cod() != (2, 1):
        #     raise ValueError("parametrization Phi must be R^2 -> R^2 (2 x 1)")

        # s, t = Phi._args
        # xs = Phi.coord_matrix[0, 0]
        # ys = Phi.coord_matrix[1, 0]

        # detJ = Phi.diff().coord_matrix.det()

        # x_expr = self.coord_matrix[0, 0]
        # integrand = x_expr.subs({self.args[0]: xs, self.args[1]: ys}) * detJ
        # result = sp_integrate(integrand, (s, s_range[0], s_range[1]),(t, t_range[0], t_range[1]))

        # return result if precision is None else N(result, precision)

        if self._kind() != "scalar" or self.dom() != 2:
            raise ValueError("integrate() needs a scalar function R^2 -> R")
        if Phi.dom() != 2 or Phi.cod() != (2, 1):
            raise ValueError("parametrization Phi must be R^2 -> R^2 (2 x 1)")

        if numeric:
            return self._integrate_numeric(Phi, s_range, t_range)

        s, t = Phi.args
        xs = Phi.coord_matrix[0, 0]
        ys = Phi.coord_matrix[1, 0]
        detJ = Phi.diff().coord_matrix.det()
        integrand = self.coord_matrix[0, 0].subs({self.args[0]: xs, self.args[1]: ys}) * detJ
        return sp_integrate(integrand,(s, s_range[0], s_range[1]),(t, t_range[0], t_range[1]))

    def _surface_area_numeric(self, u_range, v_range, n=200):
        import numpy as np
        from scipy.integrate import dblquad

        u, v = self.args
        x_expr, y_expr, z_expr = (self.coord_matrix[i, 0] for i in range(3))
        Phi_u = [x_expr.diff(u), y_expr.diff(u), z_expr.diff(u)]
        Phi_v = [x_expr.diff(v), y_expr.diff(v), z_expr.diff(v)]

        f_u = lambdify((u, v), Phi_u, "numpy")
        f_v = lambdify((u, v), Phi_v, "numpy")

        def area_elem(v, u):          # dblquad: inner u, outer v
            pu = np.array(f_u(u, v))
            pv = np.array(f_v(u, v))
            return float(np.linalg.norm(np.cross(pu, pv)))

        result, _ = dblquad(area_elem,u_range[0], u_range[1],   # inner: u
                            lambda _: v_range[0], lambda _: v_range[1])  # outer: v
        return result

    def surface_area(self, u_range, v_range, numeric = True):
        """
        Precondition: self must be a vector-valued function $f(u,v): \\mathbb{R}^2 \\rightarrow \\mathbb{R}^3$ 
        
        Returns the surface area delimited by $f$ over some domain $\\mathbf{D} = [u_0,u_1] \\times [v_0,v_1]$ computed as
        $$
            \\int_{\\mathbf{D}} \\| \\Phi_u \\times \\Phi_v \\| \\, du \\, dv 
        $$

        where $\\Phi_u$ and $\\Phi_v$ are the partial derivatives of the coordinate functions. The value here is by default a 
        float. If `numeric=False` it is a SymPy expression
        (exact, or an unevaluated integral if
        SymPy cannot do it). **Warning**: performanace can be a 
        bottleneck, even if SymPy is capable of providing an exact value. If precision is not $\\mathtt{None$}, the method returns a float. 

        Notice that orientation is not an issue here, since a norm is positive.
        """
        if self._kind() != "vector" or self.dom() != 2 or self.cod() != (3, 1):
            raise ValueError("surface_area() needs a parametric surface R^2 -> R^3 (3 x 1)")

        if numeric:
            return self._surface_area_numeric(u_range, v_range)

        u, v = self.args
        x_expr = self.coord_matrix[0, 0]
        y_expr = self.coord_matrix[1, 0]
        z_expr = self.coord_matrix[2, 0]

        Phi_u = Matrix([x_expr.diff(u), y_expr.diff(u), z_expr.diff(u)])
        Phi_v = Matrix([x_expr.diff(v), y_expr.diff(v), z_expr.diff(v)])
        area_elem = expand(trigsimp(Phi_u.cross(Phi_v).norm()))

        return sp_integrate(area_elem, (u, u_range[0], u_range[1]),(v, v_range[0], v_range[1]))
        

    def volume(self, r_range, s_range, t_range, numeric=True):
        """
        Precondition: self must be a vector-valued function $f(r,s,t): \\mathbb{R}^3 \\rightarrow \\mathbb{R}^3$ 
        which is *orientation-preserving* over 
        $\\mathbf{D} = [r_0,r_1] \\times [s_0,s_1] \\times [t_0,t_1]$ 
        
        Returns the volume delimited by $f$ overd $\\mathbf{D}$, computed as
        $$
            \\int_\\mathbf{D} det(D_f) \\, dr \\, ds \\, dt
        $$
        The value here is by default a float. If `numeric = False` it is a
        SymPy expression (exact, or an unevaluated Integral if SymPy cannot do it). 
        **Warning**: performanace can be a 
        bottleneck, even if SymPy is capable of providing an exact value. 
        
        Notice that unlike what happens in the case of a surface area where an explicit domain parametrisation must be provided, in the case of 
        a volume the parametrisation in question is the function itself. 
        """
        if self._kind() != "vector" or self.dom() != 3 or self.cod() != (3, 1):
            raise ValueError("volume() needs a parametric solid R^3 -> R^3 (3 x 1)")

        r, s, t = self.args
        detJ = self.diff().coord_matrix.det()

        if numeric:
            return self._volume_numeric(r_range, s_range, t_range)

        integrand = expand(trigsimp(detJ))
        return sp_integrate(integrand,(r, r_range[0], r_range[1]),(s, s_range[0], s_range[1]),(t, t_range[0], t_range[1]))

    def _volume_numeric(self, r_range, s_range, t_range):
        import numpy as np
        from scipy.integrate import tplquad

        r, s, t = self.args
        detJ_expr = self.diff().coord_matrix.det()
        f_det = lambdify((r, s, t), detJ_expr, "numpy")

        def integrand(t, s, r):          # tplquad: inner r, middle s, outer t
            return float(f_det(r, s, t))

        result, _ = tplquad(integrand,r_range[0], r_range[1],lambda _: s_range[0], lambda _: s_range[1], lambda _, __: t_range[0], lambda _, __: t_range[1])
        return result

    # ==== Plotting ====

    def plot(self, dim, **kwargs):
        """dim is '2d' or '3d'. """
        return dispatch(self, dim, **kwargs)

    # ==== Display ====
    
    def _repr_latex_(self):
        """
        LaTeX representation for automatic display in Jupyter/IPython.

        Returning a string from _repr_latex_() makes IPython render it
        as math when this object is the last expression of a cell.
        The string must NOT be wrapped in $...$ — IPython adds those.
        """
        openingstr = "\\begin{array}{rccl}"
        rbb = "\\mathbb{R}"
        closingstr = "\\end{array}"

        dom = len(self.args)
        if dom > 1:
            domstr = rbb + "^{" + str(dom) + "}"
        else:
            domstr = rbb

        m, n = shape(self.coord_matrix)

        if (m, n) == (1, 1):
            codstr = rbb
        else:
            if n == 1:
                codstr = rbb + "^{" + str(m) + "}"
            else:
                codstr = rbb + "^{" + str(m) + "\\times" + str(n) + "}"

        if len(self.args) == 1:
            largs = latex(self.args[0])
        else:
            largs = latex(Matrix(self.args))

        if (m, n) == (1, 1):
            lexpr = latex(self.coord_matrix[0, 0])
        else:
            lexpr = latex(self.coord_matrix)

        return (f"{openingstr} {self._name} : &{domstr} & \\longrightarrow & {codstr} "
                f"\\\\ & {largs} & \\mapsto & {lexpr} {closingstr}")

    def display(self):
        """Explicitly renders the LaTeX form (for non-last-line use)."""
        s = self._repr_latex_()
        if _HAVE_IPYTHON:
            _display(Latex(s))
        else:
            print(s)

    def __repr__(self):
        """Plain-text fallback for print() and non-IPython use."""
        s = self._repr_latex_()
        if not _HAVE_IPYTHON:
            return s
        # Strip the array wrapper for a readable one-liner.
        return f"Function({self._name})"