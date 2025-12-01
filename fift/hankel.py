# fift/hankel.py  —  Gauss–Legendre (precomputed) + NUFHT
import numpy as np
from pathlib import Path

from .lenses import AxisymmetricLens
from .utils import CPUTracker

_HAS_CFASTFHT = False
_CFASTFHT_ERR = None
_HAS_JULIA_FHT = False
_JULIA_FHT_ERR = None
j_fht_batch = None
jl = None

try:
    from .fast_hankel_c import fht_batch as c_fht_batch  # type: ignore

    _HAS_CFASTFHT = True
except Exception as exc:  # pragma: no cover - backend optional
    _CFASTFHT_ERR = exc

try:
    from juliacall import Main as jl  # type: ignore

    jl.seval("""
        using FastHankelTransform
        Base.eval(FastHankelTransform, :(using ForwardDiff))
        import FastHankelTransform: nufht

        function fht_batch(nu, rs, ys, c_re_batch, c_im_batch; tol=1e-12)
            n_w = size(c_re_batch, 1)
            n_y = length(ys)
            g_re_all = Matrix{Float64}(undef, n_w, n_y)
            g_im_all = Matrix{Float64}(undef, n_w, n_y)
            for i in 1:n_w
                cre = @view c_re_batch[i, :]
                cim = @view c_im_batch[i, :]
                gre = nufht(nu, rs, cre, ys; tol=tol)
                gim = nufht(nu, rs, cim, ys; tol=tol)
                @inbounds g_re_all[i, :] .= gre
                @inbounds g_im_all[i, :] .= gim
            end
            return g_re_all, g_im_all
        end
    """)

    j_fht_batch = jl.fht_batch
    _HAS_JULIA_FHT = True
except Exception as exc:  # pragma: no cover - backend optional
    _JULIA_FHT_ERR = exc
    jl = None


def _backend_available() -> bool:
    return _HAS_CFASTFHT or _HAS_JULIA_FHT


def _backend_error() -> str:
    parts = []
    if _CFASTFHT_ERR is not None:
        parts.append(f"C backend: {_CFASTFHT_ERR!r}")
    if _JULIA_FHT_ERR is not None:
        parts.append(f"Julia backend: {_JULIA_FHT_ERR!r}")
    if not parts:
        return "No Hankel backend available"
    return "; ".join(parts)


def _run_nufht(nu, rs, ys, coeff_re, coeff_im, tol, jr=None, jy=None):
    coeff_re = np.ascontiguousarray(coeff_re, dtype=np.float64)
    coeff_im = np.ascontiguousarray(coeff_im, dtype=np.float64)
    if _HAS_CFASTFHT:
        rs_arr = np.ascontiguousarray(rs, dtype=np.float64)
        ys_arr = np.ascontiguousarray(ys, dtype=np.float64)
        return c_fht_batch(nu, rs_arr, ys_arr, coeff_re, coeff_im, tol=tol)
    if _HAS_JULIA_FHT and j_fht_batch is not None and jl is not None:
        jr_use = jr if jr is not None else jl.Array(np.ascontiguousarray(rs, dtype=float))
        jy_use = jy if jy is not None else jl.Array(np.ascontiguousarray(ys, dtype=float))
        j_c_re = jl.Array(coeff_re)
        j_c_im = jl.Array(coeff_im)
        g_re_all, g_im_all = j_fht_batch(nu, jr_use, jy_use, j_c_re, j_c_im, tol=tol)
        return np.array(g_re_all), np.array(g_im_all)
    raise ImportError(_backend_error())


# ----------------------------------------------------------------------
# Fresnel integral with *precomputed* 1-D Gauss–Legendre nodes
# ----------------------------------------------------------------------
class FresnelHankelAxisymmetric:
    """
    Fresnel integral for axisymmetric lenses using **precomputed GL nodes**
    and **FastHankelTransform NUFHT**.

    Precomputed files must exist:

        {GL_DIR}/gl2d_n{n_gl}_U{Umax}.x.npy
        {GL_DIR}/gl2d_n{n_gl}_U{Umax}.w.npy

    We discard x<0 (symmetry) and use:
        rs = x[x>0]
        u_weights = rs * w[x>0]
    """

    def __init__(self,
                 lens: AxisymmetricLens,
                 n_gl: int,
                 Umax: float,
                 gl_dir: str,
                 tol: float = 1e-12):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError("FresnelHankelAxisymmetric requires AxisymmetricLens")

        if not _backend_available():
            raise ImportError(_backend_error())

        self.lens = lens
        self.n_gl = int(n_gl)
        self.Umax = float(Umax)
        self.tol = float(tol)

        # Load precomputed GL nodes
        gl_dir = Path(gl_dir)
        x_path = gl_dir / f"gl2d_n{n_gl}_U{int(Umax)}.x.npy"
        w_path = gl_dir / f"gl2d_n{n_gl}_U{int(Umax)}.w.npy"

        if not x_path.exists() or not w_path.exists():
            raise FileNotFoundError(f"Missing GL files:\n  {x_path}\n  {w_path}")

        x = np.load(x_path)
        w = np.load(w_path)

        # Keep only positive u (symmetry of Gauss–Legendre)
        mask = x > 0
        rs = x[mask]          # u_k
        du = w[mask]          # Δu_k
        u_weights = rs * du   # u_k Δu_k for ∫ u f(u) du

        self._rs = rs.astype(float)
        self._u_weights = u_weights.astype(float)

        self._jr = jl.Array(self._rs) if _HAS_JULIA_FHT and jl is not None else None

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def __call__(self, w_vec, y_vec):

        w_vec = np.asarray(w_vec, dtype=float).ravel()
        y_vec = np.asarray(y_vec, dtype=float).ravel()
        if np.any(w_vec == 0):
            raise ValueError("All w must be nonzero.")

        rs = self._rs
        u_weights = self._u_weights

        nu = 0
        tol = self.tol

        n_w = len(w_vec)
        n_y = len(y_vec)

        F = np.empty((n_w, n_y), dtype=np.complex128)
        quad_phase = 0.5 * y_vec**2

        with CPUTracker() as tracker:

            # ---------- Build c_re and c_im batches ----------
            c_re = np.empty((n_w, rs.size), float)
            c_im = np.empty((n_w, rs.size), float)

            for i, w in enumerate(w_vec):
                phase = (rs**2)/(2*w) - w*self.lens.psi_r(rs/w)
                fw = np.exp(1j*phase)
                ck = u_weights * fw
                c_re[i,:] = ck.real
                c_im[i,:] = ck.imag

            jy = jl.Array(y_vec) if _HAS_JULIA_FHT and jl is not None else None

            # ---------- Run batched NUFHT ----------
            g_re, g_im = _run_nufht(
                nu,
                rs,
                y_vec,
                c_re,
                c_im,
                tol,
                jr=self._jr if jy is not None else None,
                jy=jy,
            )

            # ---------- Assemble full Fresnel integral ----------
            for i, w in enumerate(w_vec):
                g = g_re[i] + 1j*g_im[i]
                F[i,:] = np.exp(1j*w*quad_phase) * (g/(1j*w))

        print(tracker.report("[FresnelHankelAxisymmetric GL_precomputed + NUFHT]"))
        return F

    
    
class FresnelHankelAxisymmetricTrapezoidal:
    r"""
    Fresnel integral for axisymmetric lenses using a fast Hankel transform.

        F(w, y) = e^{i w y^2 / 2} / (i w) ∫_0^∞ u du
                  exp{i w [ u^2 / (2 w^2) - ψ(u / w) ]} J_0(u y),

    We discretize the radial u-integral on [0, Umax] and evaluate the Bessel sum 
    with FastHankelTransform.jl's `nufht` (nonuniform fast Hankel transform). 

        ∫_0^Umax u f_w(u) J_0(u y) du ≈ ∑_k c_k(w) J_0(y r_k),

    where r_k are radial nodes, and

        c_k(w) = u_k Δu_k * exp{i w [ u_k^2 / (2 w^2) - ψ(u_k / w) ]}.

    nufht(ν, r_k, c_k, y_j; tol) then returns the vector of Hankel sums
    g_j ≈ ∑_k c_k J_ν(y_j r_k).  We have to perform a zeroth-order FHT (ν=0).
    """

    def __init__(self, lens: AxisymmetricLens,
                 n_r: int = 1024,
                 Umax: float = 50.0,
                 tol: float = 1e-12):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError(
                "FresnelHankelAxisymmetric requires an AxisymmetricLens instance."
            )

        if not _backend_available():
            raise ImportError(_backend_error())

        self.lens = lens
        self.n_r = int(n_r)
        self.Umax = float(Umax)
        self.tol = float(tol)

        # Build radial grid r_k and weights w_k for ∫_0^{Umax} u f(u) du.
        self._rs, self._u_weights = self._build_radial_grid()

        self._jr = jl.Array(self._rs) if _HAS_JULIA_FHT and jl is not None else None

    # ------------------------------------------------------------------
    # Radial grid and quadrature weights
    # ------------------------------------------------------------------
    def _build_radial_grid(self):
        """
        Simple uniform radial grid on (0, Umax] with trapezoidal weights.

        We avoid r=0 to keep things well-behaved numerically; the missing
        interval [0, r_min] is negligible for sufficiently large n_r.
        """
        n = self.n_r
        Umax = self.Umax

        # n+1 points from 0 to Umax, then drop the first (0).
        rs_full = np.linspace(0.0, Umax, n + 1, dtype=float)
        rs = rs_full[1:]               # shape (n,)
        dr = rs_full[1] - rs_full[0]

        # Trapezoidal weights for ∫_0^{Umax} … du.
        w = np.ones_like(rs) * dr
        w[0] *= 0.5
        w[-1] *= 0.5

        # For ∫ u f(u) du, combine with the extra factor u:
        u_weights = rs * w            # u_k Δu_k

        return rs, u_weights

    def __call__(self, w_vec, y_vec):
        """
        Evaluate F(w, y) for arrays of frequencies w and radii y.
        """
        w_vec = np.asarray(w_vec, dtype=float).ravel()
        y_vec = np.asarray(y_vec, dtype=float).ravel()

        if np.any(w_vec == 0.0):
            raise ValueError("All w must be nonzero.")

        rs = self._rs                # u_k
        u_weights = self._u_weights  # u_k Δu_k

        nu = 0                       # J_0 Hankel transform
        tol = self.tol

        n_w = len(w_vec)
        n_y = len(y_vec)

        F = np.empty((n_w, n_y), dtype=np.complex128)
        quad_phase = 0.5 * (y_vec ** 2)   # y^2 / 2

        with CPUTracker() as tracker:
            # ---------------- Precompute coefficients in Python ----------------
            all_c_re = []
            all_c_im = []
            for w in w_vec:
                # phase(u) = u^2/(2w) - w ψ(u/w)
                phase = (rs * rs) / (2.0 * w) - w * self.lens.psi_r(rs / w)
                f_w = np.exp(1j * phase)

                # coefficients c_k(w) = u_k Δu_k f_w(u_k)
                c_k = u_weights * f_w

                all_c_re.append(c_k.real.astype(float))
                all_c_im.append(c_k.imag.astype(float))

            # Stack to (n_w, n_r) arrays
            c_re_batch = np.stack(all_c_re, axis=0)
            c_im_batch = np.stack(all_c_im, axis=0)

            jy = jl.Array(y_vec) if _HAS_JULIA_FHT and jl is not None else None

            # ---------------- Batched Hankel transform ----------------
            g_re, g_im = _run_nufht(
                nu,
                rs,
                y_vec,
                c_re_batch,
                c_im_batch,
                tol,
                jr=self._jr if jy is not None else None,
                jy=jy,
            )

            # ---------------- Apply quadratic phase + 1/(i w) factor ----------------
            for i, w in enumerate(w_vec):
                g = g_re[i, :] + 1j * g_im[i, :]
                F[i, :] = np.exp(1j * w * quad_phase) * g / (1j * w)

        print(tracker.report("[FresnelHankelAxisymmetric]"))
        return F
