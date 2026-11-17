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

def _dispatch(f, dim, **kwargs):
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

def _duplicates(l):
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
        if _duplicates(args):
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
    def name(self):
        """For LaTeX display"""
        return self._name

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

        (ii) every symbol in argvals is among self's args, with no duplicates.

        (iii) at least one argument of self is left free (freezing all arguments would give a point, not a function).

        Returns a new Function in the remaining arguments. If name is
        None, the name is auto-generated and renders in LaTeX as a restriction of self.
        """
        if not argvals:
            raise ValueError("argvals must be a non-empty list of (symbol, value) pairs")
        syms = [s for (s, _) in argvals]
        if _duplicates(syms):
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
        Differential  
        $$D_f: \\mathbb{R}^d \\rightarrow \\mathbb{R}^{m \\times d}$$ 

        Precondition: self must be a vector-valued function $f: \\mathbb{R}^d \\rightarrow \\mathbb{R}^m$ for some 
        $d,n \\in \\mathbb{R}$
        """
        if self._kind() == "other":
            raise ValueError("diff() is for m x 1 matrices (scalar or vector functions)")
        return Function("D_" + self._name, self.args, [[c.diff(s) for s in self.args] for c in self.coord_matrix.col(0)],)

    def grad(self):
        """
        Gradient 
        $$\\nabla_f: \\mathbb{R}^d \\rightarrow \\mathbb{R}^d$$

        Precondition: self must be scalar-valued function $f: \\mathbb{R}^d \\rightarrow \\mathbb{R}$ for some 
        $d \\in \\mathbb{R}$

        Notice that this is the vector field obtained by transposing the differential.
        """
        if self._kind() != "scalar":
            raise ValueError("grad() is only for scalar functions; use diff() otherwise")
        D = self.diff()
        return Function("\\nabla_" + self._name, self.args, D.coord_matrix.T.tolist())

    def _integrate_1d_numeric(self, t_range):
        """Numeric scalar integral over a parameter interval, via scipy.integrate.quad."""
        from scipy.integrate import quad
        t = self.args[0]
        f = lambdify(t, self.coord_matrix[0, 0], "numpy")
        result, _ = quad(lambda tv: float(f(tv)),float(t_range[0]), float(t_range[1]))
        return result

    def integrate_1d(self, t_range, numeric=False):
        """
        Integral of a scalar function of one variable over a parameter interval:

        $$\\int_{t_0}^{t_1} f(t) \\, dt$$

        Precondition:
        self is a scalar function $f: \\mathbb{R} \\rightarrow \\mathbb{R}$.

        Common use: arc length of a curve, as $\\int ||C'(t)|| \\, dt$ with
        $f = ||C'||$.
        """
        if self._kind() != "scalar" or self.dom() != 1:
            raise ValueError("integrate_1d needs a scalar function R -> R")
        if numeric:
            return self._integrate_1d_numeric(t_range)
        t = self.args[0]
        return sp_integrate(self.coord_matrix[0, 0],(t, t_range[0], t_range[1]))

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
        Integral
        $$
            \\int_\\mathbf{D} \\, f \\, dx \\, dy = \\int_\\mathbf{D} f(\\Phi(s,t)) · det(D_{\\Phi}(s,t)) \\, dt \\, ds
        $$

        Preconditions: 
        
        $(i)$ self must be a scalar-valued function $f: \\mathbb{R}^2 \\rightarrow \\mathbb{R}$

        $(ii)$ the parametrisation $\\Phi: \\mathbb{R}^2 \\rightarrow \\mathbb{R}^2$ of the domain of integration
        must be *orientation-preserving*,
        i.e. $det(D_{\\Phi}) \\geq 0$ on $\\mathbf{D} = [s_0,s_1] \\times [t_0,t_1]$. Notice that
        given the precondition no absolute value is taken
        (polar, cylindrical, spherical, and all the usual textbook parametrizations satisfy this on their standard 
        domains).
        
        The value here is by default a SymPy expression (the exact integral, or an unevaluated
        integral if SymPy cannot do it). **Warning**: performance can be a bottleneck, even if SymPy is capable
        of providing an exact value. If `numeric = True`, the method returns a float.
        """
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

    def _line_integrand(self, gamma):
        """
        self: vector field
        gamma: parametric curve

        Common setup for integrate_along: returns (t, integrand_expr) where
        integrand_expr is the scalar F(γ(t)) · γ'(t) in the parameter t.

        Shared by the symbolic and numeric paths so they cannot diverge.
        """
        t = gamma.args[0]
        subs = {fa: gc for fa, gc in zip(self.args, gamma.coord_matrix.col(0))}
        gamma_prime = gamma.diff().coord_matrix.col(0)
        integrand = sum(
            F_i.subs(subs) * gp_i
            for F_i, gp_i in zip(self.coord_matrix.col(0), gamma_prime)
        )
        return t, integrand

    def integrate_along(self, gamma, t_range, numeric=False):
        """
            Line integral of a vector field:  
            $$ \\int_\\gamma F dr = \\int_{t0}^{t1} F(\\gamma(t)) \\cdot \\gamma'(t) dt$$
            
    
            Preconditions:
            (i)  self  is a vector field $F: \\mathbb{R}^n \\rightarrow \\mathbb{R}^n$

            (ii) gamma is a curve $\\gamma: \\mathbb{R} \\rightarrow \\mathbb{R}^n$ matching $F$'s dimension
    
            Orientation: reversing t_range reverses the sign, as expected.
        """
        n = self.dom()
        if self._kind() != "vector" or self.cod() != (n, 1):
            raise ValueError("integrate_along needs a vector field F: R^n -> R^n")
        if gamma._kind() != "vector" or gamma.dom() != 1 or gamma.cod() != (n, 1):
            raise ValueError("gamma must be a curve R -> R^n matching F's dimension")
              
        if numeric:
            return self._integrate_along_numeric(gamma, t_range)
        t, integrand = self._line_integrand(gamma)
        return sp_integrate(expand(trigsimp(integrand)),(t, t_range[0], t_range[1]))

    def _integrate_along_numeric(self, gamma, t_range):
        import numpy as np
        from scipy.integrate import quad
        t, integrand_expr = self._line_integrand(gamma)
        f = lambdify(t, integrand_expr, "numpy")
        result, _ = quad(lambda tv: float(f(tv)),float(t_range[0]), float(t_range[1]))
        return result

    def _integrate_over_numeric(self, S, u_range, v_range):
        """Numeric flux via scipy.integrate.dblquad. Lazy imports."""
        from scipy.integrate import dblquad
        u, v, integrand_expr = self._flux_integrand(S)
        f = lambdify((u, v), integrand_expr, "numpy")

        def integrand(v_val, u_val):          # dblquad: inner u, outer v
            return float(f(u_val, v_val))

        result, _ = dblquad(
            integrand,
            float(u_range[0]), float(u_range[1]),
            lambda _: float(v_range[0]), lambda _: float(v_range[1]),
        )
        return result

    def integrate_over(self, S, u_range, v_range, numeric=False):
        """
        Flux of a vector field across a parametric surface:

        $$\\int_S F \\cdot dS = \\int_{u_0}^{u_1} \\int_{v_0}^{v_1} F(S(u,v)) \\cdot (S_u \\times S_v) \\, dv \\, du$$

        Preconditions:
        (i)  self is a vector field $F : \\mathbb{R}^3 \\to \\mathbb{R}^3$;

        (ii) S is a parametric surface $S : \\mathbb{R}^2 \\to \\mathbb{R}^3$;

        (iii) the orientation of S (given by $S_u \\times S_v$) is the one
                the user intends — reversing a parameter range reverses the sign.

        The result may be negative; that is the orientation, not an error.
        """
        if self._kind() != "vector" or self.dom() != 3 or self.cod() != (3, 1):
            raise ValueError("integrate_over needs a vector field F: R^3 -> R^3")
        if S._kind() != "vector" or S.dom() != 2 or S.cod() != (3, 1):
            raise ValueError("S must be a parametric surface R^2 -> R^3 (3 x 1)")

        if numeric:
            return self._integrate_over_numeric(S, u_range, v_range)

        u, v, integrand = self._flux_integrand(S)
        return sp_integrate(expand(trigsimp(integrand)),
                            (u, u_range[0], u_range[1]),
                            (v, v_range[0], v_range[1]))

    def _flux_integrand(self, S):
        """Returns (u, v, F(S(u,v)) · (S_u × S_v))."""
        u, v = S.args
        S_coords = S.coord_matrix.col(0)
        subs = {fa: sc for fa, sc in zip(self.args, S_coords)}
        S_u = [c.diff(u) for c in S_coords]
        S_v = [c.diff(v) for c in S_coords]
        cross = [S_u[1]*S_v[2] - S_u[2]*S_v[1],
                S_u[2]*S_v[0] - S_u[0]*S_v[2],
                S_u[0]*S_v[1] - S_u[1]*S_v[0]]
        integrand = sum(F_i.subs(subs) * c_i
                        for F_i, c_i in zip(self.coord_matrix.col(0), cross))
        return u, v, integrand

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
        The surface area delimited by $f$ over some domain $\\mathbf{D} = [u_0,u_1] \\times [v_0,v_1]$, computed as
        $$
            \\int_{\\mathbf{D}} \\| \\Phi_u \\times \\Phi_v \\| \\, du \\, dv 
        $$
        where $\\Phi_u$ and $\\Phi_v$ are the partial derivatives of the coordinate functions.

        Precondition: self must be a vector-valued function $f(u,v): \\mathbb{R}^2 \\rightarrow \\mathbb{R}^3$ 
        
        The value here is by default a 
        float. If `numeric=False` it is a SymPy expression
        (exact, or an unevaluated integral if
        SymPy cannot do it). **Warning**: performanace can be a 
        bottleneck, even if SymPy is capable of providing an exact value. If precision is not $\\mathtt{None}$, the method returns a float. 

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
        The volume delimited by $f$ overd $\\mathbf{D}$, computed as
        $$
            \\int_\\mathbf{D} det(D_f) \\, dr \\, ds \\, dt
        $$

        Precondition: self must be a vector-valued function $f(r,s,t): \\mathbb{R}^3 \\rightarrow \\mathbb{R}^3$ 
        which is *orientation-preserving* over $\\mathbf{D} = [r_0,r_1] \\times [s_0,s_1] \\times [t_0,t_1]$ 

        Notice that unlike what happens in the case of a surface area where an explicit domain parametrisation must be provided, in the case of 
        a volume the parametrisation in question is the function itself.
        
        The value here is by default a float. If `numeric = False` it is a
        SymPy expression (exact, or an unevaluated Integral if SymPy cannot do it). 
        **Warning**: performanace can be a 
        bottleneck, even if SymPy is capable of providing an exact value. 
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

    def _flux_across_integrand(self, gamma):
        """Returns (t, F_x·γ'_y − F_y·γ'_x), the integrand of ∮_C F·n ds."""
        t = gamma.args[0]
        subs = {fa: gc for fa, gc in zip(self.args, gamma.coord_matrix.col(0))}
        gx = gamma.coord_matrix[0, 0]
        gy = gamma.coord_matrix[1, 0]
        Fx = self.coord_matrix[0, 0].subs(subs)
        Fy = self.coord_matrix[1, 0].subs(subs)
        return t, Fx * gy.diff(t) - Fy * gx.diff(t)

    def flux_across(self, gamma, t_range, numeric=False):
        """
        Flux of a plane vector field across a curve (2D divergence form of Green):

        $$\\oint_C F \\cdot n \\, ds = \\int_{t_0}^{t_1} F(\\gamma(t)) \\cdot n \\, ||\\gamma'(t)|| \\, dt$$

        where $n = (\\gamma'_y, -\\gamma'_x)/||\\gamma'||$ is the outward normal for
        counterclockwise traversal. The $||\\gamma'||$ cancels, so the integrand is
        $F_x \\gamma'_y - F_y \\gamma'_x$.

        Orientation: reversing t_range reverses the sign.
        """
        if self._kind() != "vector" or self.dom() != 2 or self.cod() != (2, 1):
            raise ValueError("flux_across needs a vector field F: R^2 -> R^2")
        if gamma._kind() != "vector" or gamma.dom() != 1 or gamma.cod() != (2, 1):
            raise ValueError("gamma must be a curve R -> R^2")

        if numeric:
            return self._flux_across_numeric(gamma, t_range)

        t, integrand = self._flux_across_integrand(gamma)
        return sp_integrate(expand(trigsimp(integrand)), (t, t_range[0], t_range[1]))

    def _flux_across_numeric(self, gamma, t_range):
        from scipy.integrate import quad
        t, integrand_expr = self._flux_across_integrand(gamma)
        f = lambdify(t, integrand_expr, "numpy")
        result, _ = quad(lambda tv: float(f(tv)),
                        float(t_range[0]), float(t_range[1]))
        return result

    def div(self):
        """
        Divergence of a vector field:

        $$\\nabla \\cdot F = \\sum_{i=1}^{n} \\frac{\\partial F_i}{\\partial x_i}$$

        Precondition: self is a vector field $F: \\mathbb{R}^n \\to \\mathbb{R}^n$.

        Returns a scalar function $\\mathbb{R}^n \\to \\mathbb{R}$ (a new Function).
        """
        if self._kind() != "vector" or self.cod() != (self.dom(), 1):
            raise ValueError("div() needs a vector field F: R^n -> R^n")
        n = self.dom()
        expr = sum(self.coord_matrix[i, 0].diff(self.args[i]) for i in range(n))
        return Function("div " + self._name, self.args, [expr])

    def curl(self):
        """
        Curl of a plane vector field, in its scalar form:

        $$\\operatorname{curl} F = \\frac{\\partial F_2}{\\partial x} - \\frac{\\partial F_1}{\\partial y}$$

        Precondition: self is a vector field $F: \\mathbb{R}^2 \\to \\mathbb{R}^2$.

        Returns a scalar function $\\mathbb{R}^2 \\to \\mathbb{R}$ (a new Function).

        Note: in 2D the curl of a vector field is a scalar, not a vector — it is
        the $z$-component of the 3D curl of $(F_1, F_2, 0)$. The 3D vector-valued
        curl is a separate method (`curl3d`) to come.
        """
        if self._kind() != "vector" or self.dom() != 2 or self.cod() != (2, 1):
            raise ValueError("curl() needs a plane vector field F: R^2 -> R^2")
        P = self.coord_matrix[0, 0]
        Q = self.coord_matrix[1, 0]
        x1, x2 = self.args
        expr = Q.diff(x1) - P.diff(x2)
        return Function("curl " + self._name, self.args, [expr])

    # ==== Plotting ====

    def plot(self, plot_classification, **kwargs):
        """
        Constructs the correct `Matplotlib` plot in memory if applicable, it is the responsability of the user to place it in a Canvas object for
        plotting (cf. the `graphics` module).

        The call is highly context-sensitive, since by far not any function that can abstractly be handled by the class `Function` is eligible
        for visualisation. Notice that this is an intrinsic limitation of human spatial senses, due to the way we evolved as species. Furthermore, 
        certain eligible functions can admit different visualisations, for instance we can plot a function $\\mathbb{R}^2 \\rightarrow \\mathbb{R}$ as
        a surface or as a set of level curves.

        For these reasons, the method `plot` needs a classification provided by the argement `plot_classification`, which takes the strings '2d' or '3d'
        as values. It is important
        to keep in mind that this argument classifies the *plot* to be constructed and *not* the underlying abstract function. 
        It is in theory a decision of the user, although there is only one type of function handled by this module for which both '2d' and '3d' make sense. 
        Once this decision taken, it the responsability of the user to provide the correct range arguments. The following table make this precise

        | type of function                            | plot classification   | range argument(s)                             | dispatched to          |
        |---------------------------------------------|-----------------------| ----------------------------------------------|------------------------|
        | $\\mathbb{R} \\rightarrow \\mathbb{R}$      | `2d`                  | `x_range = (min,max)`                         | `Curve2D`              |
        | $\\mathbb{R}^2 \\rightarrow \\mathbb{R}$    | `2d`                  | `x_range = (min,max)`, `y_range = (min,max)`  | `Implicit2D`           |
        | $\\mathbb{R}^2 \\rightarrow \\mathbb{R}$    | `3d`                  | `x_range = (min,max)`, `y_range = (min,max)`  | `Surface`              |
        | $\\mathbb{R} \\rightarrow \\mathbb{R}^2$    | `2d`                  | `t_range= (min,max)`                          | `ParametricCurve2D`    |
        | $\\mathbb{R} \\rightarrow \\mathbb{R}^3$    | `3d`                  | `t_range = (min,max)`                         | `ParametricCurve3D`    |
        | $\\mathbb{R}^2 \\rightarrow \\mathbb{R}^3$  | `3d`                  | `u_range = (min,max)`, `v_range = (min,max)`  | `ParametricSurface3D`  |
        | $\\mathbb{R}^2 \\rightarrow \\mathbb{R}^2$  | `2d`                  | `x_range = (min,max)`, `x_range = (min,max)`  | `VectorField2D`        |
                                        

        The column with the dispatches is purely informative (a dispatch is part of the software architecture, not a user-level action).
        The range arguments are passed as `**kwargs` as their names and their number vary. Notice that other arguments (for instance 
        those specifying a color or a color map) can be passed as `**kwargs` as well.  
        """
        return _dispatch(self, plot_classification, **kwargs)

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
        """
        LaTeX display of a Function object in a Jupyter notebook. The method is automatically invoked if (the `Python` variable containing) the 
        object is the last line of a code cell, otherwise it needs to be invoked it explicitely. 
        """
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