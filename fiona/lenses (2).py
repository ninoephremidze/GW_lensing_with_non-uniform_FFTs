####################################################################
# fiona/lenses.py
####################################################################

import numpy as np
from abc import ABC, abstractmethod

try:
    import numexpr as ne
    _HAS_NUMEXPR = True
except Exception:
    _HAS_NUMEXPR = False

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)  # we want float64 for lensing
    _HAS_JAX = True
except Exception:
    _HAS_JAX = False

# ----------------------------------------------------------------------
# Base classes
# ----------------------------------------------------------------------

class Lens(ABC):
    """Abstract base for a projected lensing potential ψ(x1,x2)."""

    @abstractmethod
    def psi_xy(self, x1, x2):
        """Return ψ(x) on R^2 (broadcast over arrays)."""
        ...


class AxisymmetricLens(Lens):
    """Axisymmetric lenses: ψ(x) = ψ(r), with r = sqrt(x1^2 + x2^2)."""

    @abstractmethod
    def psi_r(self, r):
        ...

    def psi_xy(self, x1, x2):
        """
        Compute ψ(x1,x2) by first forming r = sqrt(x1^2 + x2^2)
        and then calling psi_r(r). Uses numexpr if available.
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        if _HAS_NUMEXPR:
            x1b, x2b = np.broadcast_arrays(x1, x2)
            r = ne.evaluate("sqrt(x1b*x1b + x2b*x2b)")
        else:
            r = np.hypot(x1, x2)

        return self.psi_r(r)


# ----------------------------------------------------------------------
# Simple axisymmetric lenses
# ----------------------------------------------------------------------

class SIS(AxisymmetricLens):
    r"""Singular Isothermal Sphere: ψ(r) = ψ0 * r."""
    def __init__(self, psi0=1.0):
        self.psi0 = float(psi0)

    def psi_r(self, r):
        r = np.asarray(r, dtype=float)
        psi0 = self.psi0
        if _HAS_NUMEXPR:
            return ne.evaluate("psi0 * r")
        else:
            return psi0 * r


class PointLens(AxisymmetricLens):
    r"""Point lens with Plummer softening: ψ(r) = 0.5 ψ0 log(r^2 + x_c^2)."""
    def __init__(self, psi0=1.0, xc=0.0):
        self.psi0 = float(psi0)
        self.xc   = float(xc)

    def psi_r(self, r):
        r = np.asarray(r, dtype=float)
        psi0 = self.psi0
        xc2  = self.xc * self.xc
        if _HAS_NUMEXPR:
            return ne.evaluate("0.5 * psi0 * log(r*r + xc2)")
        else:
            return 0.5 * psi0 * np.log(r*r + xc2)


# ----------------------------------------------------------------------
# NFW (axisymmetric) + off-center NFW
# ----------------------------------------------------------------------

class NFW(AxisymmetricLens):
    r"""
    Axisymmetric Navarro–Frenk–White (NFW) lens.

    Follows the analytic form used in GLoW (up to small-u series approximations):

        u = r / x_s

        F(u) = 1/sqrt(u^2 - 1) * arctan(sqrt(u^2 - 1))          (u > 1)
             = 1/sqrt(1 - u^2) * atanh(sqrt(1 - u^2))          (u < 1)
             = 1                                              (u = 1)

        ψ(r) = 0.5 * ψ0 * [ log^2(u/2) + (u^2 - 1) * F(u)^2 ].
    """
    def __init__(self, psi0=1.0, xs=0.1):
        self.psi0 = float(psi0)
        self.xs   = float(xs)

    @staticmethod
    def _F_nfw(u):
        u = np.asarray(u, dtype=float)
        out = np.empty_like(u)

        gt1 = u > 1.0
        lt1 = u < 1.0
        eq1 = ~(gt1 | lt1)

        if np.any(gt1):
            ug = u[gt1]
            s = np.sqrt(ug*ug - 1.0)
            out[gt1] = np.arctan(s) / s

        if np.any(lt1):
            ul = u[lt1]
            s = np.sqrt(1.0 - ul*ul)
            out[lt1] = np.arctanh(s) / s

        if np.any(eq1):
            out[eq1] = 1.0

        return out

    def psi_r(self, r):
        r = np.asarray(r, dtype=float)
        psi0 = self.psi0
        xs   = self.xs

        u = r / xs
        F = self._F_nfw(u)

        if _HAS_NUMEXPR:
            # combine algebra with numexpr
            return ne.evaluate(
                "0.5 * psi0 * (log(u/2.0)**2 + (u*u - 1.0) * F*F)"
            )
        else:
            log_term = np.log(u / 2.0)
            return 0.5 * psi0 * (log_term*log_term + (u*u - 1.0)*F*F)


class OffcenterNFW(Lens):
    r"""
    Off-center NFW lens:

        ψ(x1, x2) = ψ_NFW( sqrt((x1 - xc1)^2 + (x2 - xc2)^2) ).

    Parameters match GLoW's Psi_offcenterNFW (psi0, xs, xc1, xc2).
    """
    def __init__(self, psi0=1.0, xs=0.1, xc1=0.0, xc2=0.0):
        self.psi0 = float(psi0)
        self.xs   = float(xs)
        self.xc1  = float(xc1)
        self.xc2  = float(xc2)
        self._nfw = NFW(psi0=self.psi0, xs=self.xs)

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        dx1 = x1 - self.xc1
        dx2 = x2 - self.xc2

        if _HAS_NUMEXPR:
            r = ne.evaluate("sqrt(dx1*dx1 + dx2*dx2)")
        else:
            r = np.hypot(dx1, dx2)

        return self._nfw.psi_r(r)


# ----------------------------------------------------------------------
# Non-axisymmetric single-lens models
# ----------------------------------------------------------------------

class SISPlusExternalShear(Lens):
    r"""
    SIS + external shear:

        ψ(x1, x2) = ψ0 * sqrt(x1^2 + x2^2)
                    + 0.5 * γ1 * (x1^2 - x2^2)
                    + γ2 * x1 * x2

    γ1, γ2 are the usual Cartesian shear components.  When (γ1, γ2) = (0, 0)
    this reduces to a pure SIS.
    """
    def __init__(self, psi0=1.0, gamma1=0.1, gamma2=0.0):
        self.psi0   = float(psi0)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        psi0 = self.psi0
        g1   = self.gamma1
        g2   = self.gamma2

        if _HAS_NUMEXPR:
            return ne.evaluate(
                "psi0 * sqrt(x1*x1 + x2*x2)"
                " + 0.5 * g1 * (x1*x1 - x2*x2)"
                " + g2 * x1 * x2"
            )
        else:
            r = np.hypot(x1, x2)
            return psi0 * r + 0.5 * g1 * (x1*x1 - x2*x2) + g2 * x1 * x2


class PIED(Lens):
    r"""
    Elliptical pseudo-isothermal potential (toy model):

        ψ(x1, x2) = ψ0 * sqrt(r_c^2 + x1^2 + x2^2 / q^2)

    where q is the axis ratio (q < 1: flattened along x2; q > 1: flattened along x1)
    and r_c is a core radius to keep the center finite.
    """
    def __init__(self, psi0=1.0, q=0.7, r_core=0.05):
        if q <= 0:
            raise ValueError("Axis ratio q must be > 0.")
        self.psi0   = float(psi0)
        self.q      = float(q)
        self.r_core = float(r_core)

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        psi0 = self.psi0
        q    = self.q
        rc2  = self.r_core * self.r_core

        if _HAS_NUMEXPR:
            return ne.evaluate("psi0 * sqrt(rc2 + x1*x1 + (x2*x2) / (q*q))")
        else:
            return psi0 * np.sqrt(rc2 + x1*x1 + (x2*x2) / (q*q))


class EllipticalSIS(Lens):
    r"""
    Elliptical SIS (pseudo-elliptical potential):

        Define rotated coordinates (x1', x2') around center (xc1, xc2):

            dx1 = x1 - xc1
            dx2 = x2 - xc2
            x1' =  cosα * dx1 + sinα * dx2
            x2' = -sinα * dx1 + cosα * dx2

        Elliptical radius:
            R = sqrt( x1'^2 + (x2'/q)^2 )

        Potential:
            ψ(x1, x2) = ψ0 * R

        For q = 1, this reduces to a circular SIS with ψ = ψ0 * r.
    """
    def __init__(self, psi0=1.0, q=0.7, alpha=0.0, xc1=0.0, xc2=0.0):
        if q <= 0:
            raise ValueError("Axis ratio q must be > 0.")
        self.psi0  = float(psi0)
        self.q     = float(q)
        self.alpha = float(alpha)
        self.xc1   = float(xc1)
        self.xc2   = float(xc2)

        self._cos = np.cos(self.alpha)
        self._sin = np.sin(self.alpha)

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        dx1 = x1 - self.xc1
        dx2 = x2 - self.xc2

        c = self._cos
        s = self._sin
        q = self.q
        psi0 = self.psi0

        if _HAS_NUMEXPR:
            # rotated coords
            x1p = ne.evaluate("dx1*c + dx2*s")
            x2p = ne.evaluate("-dx1*s + dx2*c")
            R   = ne.evaluate("sqrt(x1p*x1p + (x2p/q)*(x2p/q))")
            return ne.evaluate("psi0 * R")
        else:
            x1p = c*dx1 + s*dx2
            x2p = -s*dx1 + c*dx2
            R = np.sqrt(x1p*x1p + (x2p/q)**2)
            return psi0 * R


# ----------------------------------------------------------------------
# Vectorized clumpy lens: host NFW + many elliptical SIS subhalos
# ----------------------------------------------------------------------

class ClumpySIELens(Lens):
    r"""
    Vectorized "clumpy" lens consisting of:

      - One axisymmetric NFW host centered at the origin.
      - K elliptical SIS (eSIS) subhalos with parameters
        psi0_k, q_k, alpha_k, centers (xc1_k, xc2_k).

    All components are evaluated in a single broadcasted pass
    (no Python loop over subhalos), which is critical for speed
    when used inside FIONA's 2D quadrature.

    Parameters
    ----------
    psi0_host : float
        Normalization of the host NFW potential.
    xs_host : float
        Scale radius of the host NFW.
    psi0_sub : array-like, shape (K,)
        Normalizations of the SIS subhalo potentials.
    q_sub : array-like, shape (K,)
        Axis ratios of the subhalos.
    alpha_sub : array-like, shape (K,)
        Position angles (radians) of the subhalos, measured from x1.
    xc1_sub, xc2_sub : array-like, shape (K,)
        Centers of the subhalos.
    """
    def __init__(
        self,
        psi0_host,
        xs_host,
        psi0_sub,
        q_sub,
        alpha_sub,
        xc1_sub,
        xc2_sub,
    ):
        self.host = NFW(psi0=psi0_host, xs=xs_host)

        psi0_sub  = np.asarray(psi0_sub,  dtype=float)
        q_sub     = np.asarray(q_sub,     dtype=float)
        alpha_sub = np.asarray(alpha_sub, dtype=float)
        xc1_sub   = np.asarray(xc1_sub,   dtype=float)
        xc2_sub   = np.asarray(xc2_sub,   dtype=float)

        if not (psi0_sub.shape == q_sub.shape == alpha_sub.shape == xc1_sub.shape == xc2_sub.shape):
            raise ValueError("All subhalo parameter arrays must have the same shape.")

        if np.any(q_sub <= 0):
            raise ValueError("All subhalo axis ratios q must be > 0.")

        self.psi0_sub  = psi0_sub          # (K,)
        self.q_sub     = q_sub
        self.alpha_sub = alpha_sub
        self.xc1_sub   = xc1_sub
        self.xc2_sub   = xc2_sub

        self.cos_alpha = np.cos(alpha_sub) # (K,)
        self.sin_alpha = np.sin(alpha_sub)

    def psi_xy(self, x1, x2):
        """
        Evaluate ψ_total(x1,x2) = ψ_host_NFW + Σ_k ψ_eSIS_k in a single vectorized pass.
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        # Host NFW (axisymmetric)
        if _HAS_NUMEXPR:
            x1b, x2b = np.broadcast_arrays(x1, x2)
            r_host = ne.evaluate("sqrt(x1b*x1b + x2b*x2b)")
        else:
            r_host = np.hypot(x1, x2)
        psi_host = self.host.psi_r(r_host)  # same shape as x1/x2

        # Subhalos: broadcast over last axis (K)
        dx1 = x1[..., None] - self.xc1_sub   # (..., K)
        dx2 = x2[..., None] - self.xc2_sub   # (..., K)

        c = self.cos_alpha   # (K,)
        s = self.sin_alpha
        q = self.q_sub
        psi0 = self.psi0_sub

        if _HAS_NUMEXPR:
            # rotated coordinates (x1', x2') for all subhalos simultaneously
            x1p = ne.evaluate("dx1*c + dx2*s")
            x2p = ne.evaluate("-dx1*s + dx2*c")
            R   = ne.evaluate("sqrt(x1p*x1p + (x2p/q)*(x2p/q))")
            psi_sub = ne.evaluate("psi0 * R")  # (..., K)
        else:
            x1p = dx1*c + dx2*s
            x2p = -dx1*s + dx2*c
            R   = np.sqrt(x1p*x1p + (x2p/q)**2)
            psi_sub = psi0 * R

        psi_sub_total = np.sum(psi_sub, axis=-1)  # sum over subhalos
        return psi_host + psi_sub_total

    
class ClumpyNFWLens(Lens):
    """
    Host NFW at the origin + many off-center NFW subhalos, evaluated
    in a single vectorized pass.

    host:  psi0_host, xs_host
    subs:  psi0_sub[k], xs_sub[k], centers (xc1_sub[k], xc2_sub[k]), k=0..K-1
    """

    def __init__(self, psi0_host, xs_host, psi0_sub, xs_sub, xc1_sub, xc2_sub):
        # Use the NFW class defined above as the host
        self.host = NFW(psi0=psi0_host, xs=xs_host)

        psi0_sub = np.asarray(psi0_sub, dtype=float)
        xs_sub   = np.asarray(xs_sub,   dtype=float)
        xc1_sub  = np.asarray(xc1_sub,  dtype=float)
        xc2_sub  = np.asarray(xc2_sub,  dtype=float)

        if not (psi0_sub.shape == xs_sub.shape == xc1_sub.shape == xc2_sub.shape):
            raise ValueError("Subhalo parameter arrays must all have the same shape.")

        self.psi0_sub = psi0_sub
        self.xs_sub   = xs_sub
        self.xc1_sub  = xc1_sub
        self.xc2_sub  = xc2_sub

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        # Host NFW (axisymmetric, centered at origin)
        r_host = np.hypot(x1, x2)
        psi_host = self.host.psi_r(r_host)

        # Subhalos: broadcast over last axis (K)
        dx1 = x1[..., None] - self.xc1_sub   # (..., K)
        dx2 = x2[..., None] - self.xc2_sub   # (..., K)
        r   = np.hypot(dx1, dx2)             # (..., K)

        # NFW radial coordinate for each subhalo
        u = r / self.xs_sub                  # (..., K)

        # Use NFW._F_nfw on flattened u, then reshape back
        u_flat = u.ravel()
        F_flat = NFW._F_nfw(u_flat)
        F = F_flat.reshape(u.shape)          # (..., K)

        # NFW potential: ψ = 0.5 * ψ0 * [log^2(u/2) + (u^2 - 1) * F^2]
        log_term = np.log(u / 2.0)
        psi_sub = 0.5 * self.psi0_sub * (log_term*log_term + (u*u - 1.0)*F*F)  # (..., K)

        psi_sub_total = np.sum(psi_sub, axis=-1)  # sum over subhalos

        return psi_host + psi_sub_total

if _HAS_JAX:
    def _jax_F_nfw(u):
        """
        JAX version of NFW auxiliary function F(u) for the lens potential.
        Works on jnp arrays, fully vectorized.
        """
        u = jnp.asarray(u, dtype=jnp.float64)
        gt1 = u > 1.0
        lt1 = u < 1.0
        eq1 = ~(gt1 | lt1)

        def branch_gt1(u):
            s = jnp.sqrt(u*u - 1.0)
            return jnp.arctan(s) / s

        def branch_lt1(u):
            s = jnp.sqrt(1.0 - u*u)
            return jnp.arctanh(s) / s

        F_gt = branch_gt1(u)
        F_lt = branch_lt1(u)

        # piecewise assemble
        out = jnp.where(gt1, F_gt, jnp.where(lt1, F_lt, 1.0))
        return out

    def _jax_nfw_potential(r, psi0, xs):
        """
        JAX implementation of your NFW lens potential:

          u = r / x_s
          ψ(r) = 0.5 * ψ0 * [log^2(u/2) + (u^2 - 1) * F(u)^2]
        """
        r = jnp.asarray(r, dtype=jnp.float64)
        psi0 = jnp.asarray(psi0, dtype=jnp.float64)
        xs   = jnp.asarray(xs,   dtype=jnp.float64)

        u = r / xs
        F = _jax_F_nfw(u)
        log_term = jnp.log(u / 2.0)
        return 0.5 * psi0 * (log_term*log_term + (u*u - 1.0)*F*F)

    
class JAXClumpyNFWLens(Lens):
    """
    Host NFW at the origin + many off-center NFW subhalos,
    evaluated by a single JAX-jitted kernel.

    host:  psi0_host, xs_host
    subs:  psi0_sub[k], xs_sub[k], centers (xc1_sub[k], xc2_sub[k]), k=0..K-1

    If JAX is not available, this class will raise ImportError on init.
    """

    def __init__(self, psi0_host, xs_host, psi0_sub, xs_sub, xc1_sub, xc2_sub):
        if not _HAS_JAX:
            raise ImportError("JAX is not available; cannot use JAXClumpyNFWLens.")

        # store host (just scalars; we use JAX formula directly)
        self.psi0_host = float(psi0_host)
        self.xs_host   = float(xs_host)

        psi0_sub = np.asarray(psi0_sub, dtype=float)
        xs_sub   = np.asarray(xs_sub,   dtype=float)
        xc1_sub  = np.asarray(xc1_sub,  dtype=float)
        xc2_sub  = np.asarray(xc2_sub,  dtype=float)

        if not (psi0_sub.shape == xs_sub.shape == xc1_sub.shape == xc2_sub.shape):
            raise ValueError("Subhalo parameter arrays must all have the same shape.")

        self.psi0_sub = psi0_sub
        self.xs_sub   = xs_sub
        self.xc1_sub  = xc1_sub
        self.xc2_sub  = xc2_sub

        # JAX copies of parameters (constant for the life of the lens)
        self._psi0_host_j = jnp.asarray(self.psi0_host, dtype=jnp.float64)
        self._xs_host_j   = jnp.asarray(self.xs_host,   dtype=jnp.float64)
        self._psi0_sub_j  = jnp.asarray(self.psi0_sub, dtype=jnp.float64)
        self._xs_sub_j    = jnp.asarray(self.xs_sub,   dtype=jnp.float64)
        self._xc1_sub_j   = jnp.asarray(self.xc1_sub,  dtype=jnp.float64)
        self._xc2_sub_j   = jnp.asarray(self.xc2_sub,  dtype=jnp.float64)

        # JIT-compile the core kernel (x1,x2) -> psi
        self._psi_kernel = jax.jit(self._psi_kernel_fn)

    def _psi_kernel_fn(self, x1, x2):
        """
        Core JAX kernel: x1,x2 are jnp arrays (shape (M,N)), returns ψ(x1,x2).
        """

        # Host NFW at origin
        r_host = jnp.sqrt(x1*x1 + x2*x2)
        psi_host = _jax_nfw_potential(r_host, self._psi0_host_j, self._xs_host_j)

        # Subhalos: broadcast over last axis (K)
        dx1 = x1[..., None] - self._xc1_sub_j  # (..., K)
        dx2 = x2[..., None] - self._xc2_sub_j  # (..., K)
        r   = jnp.sqrt(dx1*dx1 + dx2*dx2)      # (..., K)

        psi_sub = _jax_nfw_potential(r, self._psi0_sub_j, self._xs_sub_j)  # (..., K)
        psi_sub_total = jnp.sum(psi_sub, axis=-1)  # (...,)

        return psi_host + psi_sub_total

    def psi_xy(self, x1, x2):
        """
        Public API: accept NumPy arrays, return NumPy arrays.

        FIONA will call this once with large (M,N) arrays; JAX will
        compile on the first call and then reuse the compiled kernel.
        """
        if not _HAS_JAX:
            raise ImportError("JAX is not available; cannot use JAXClumpyNFWLens.")

        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        x1_j = jnp.asarray(x1, dtype=jnp.float64)
        x2_j = jnp.asarray(x2, dtype=jnp.float64)

        psi_j = self._psi_kernel(x1_j, x2_j)   # JAX DeviceArray
        psi_np = np.asarray(psi_j)             # back to NumPy

        return psi_np