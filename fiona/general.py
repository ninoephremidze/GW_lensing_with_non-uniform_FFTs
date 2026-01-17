####################################################################
# fiona/general.py
####################################################################

### This code computes integral in the u-domain.
### In the u-domain formulation, all frequency dependence is moved into the integrand coefficients, 
### while the oscillatory kernel is frequency-independent. This allows reuse of a single NUFFT geometry 
### and efficient batched evaluation over many frequencies, but requires recomputing the lens potential
### at scaled arguments ψ(u/w) for each frequency. In contrast, the x-domain formulation evaluates the
### lens potential ψ(x) once on a fixed spatial quadrature grid and is therefore advantageous when ψ is
### expensive to compute, but the NUFFT targets scale with frequency, so a separate NUFFT must be executed
### for each frequency, making it less efficient when many frequencies are needed.

import os
import numpy as np
import numexpr as ne

from .utils import (
    gauss_legendre_2d,
    gauss_legendre_polar_2d,
    gauss_legendre_polar_uniform_theta_2d,  # keep utils name; see polar branch
    CPUTracker,
)

try:
    from finufft import nufft2d3
    _FINUFFT = True
except Exception:
    _FINUFFT = False

_TWO_PI = 2.0 * np.pi


def _integrand_coeffs(u1, u2, w, lens, W):
    """
    I(y) = ∫ d^2u e^{-i u·y} exp{i [u^2/(2w) - w ψ(u/w)]}.
    """
    phase = (u1 * u1 + u2 * u2) / (2.0 * w) - w * lens.psi_xy(u1 / w, u2 / w)
    return np.exp(1j * phase) * W


class FresnelNUFFT3:
    """
    Fresnel integral evaluator using FINUFFT (type-3 2D NUFFT) on a quadrature grid.

    Key ideas
    ---------
    - Per-frequency mode: build quadrature grid and call NUFFT for each w independently.
    - Batched mode: share one quadrature + NUFFT setup across a *group* of frequencies.
      Optionally chunk frequencies into groups, using either log or linear binning in |w|.

    Parameters (public API)
    -----------------------
    lens : object
        Must implement lens.psi_xy(u1, u2) with numpy broadcasting.

    gl_nodes_per_dim : int
        Gauss-Legendre nodes per dimension (Cartesian). Total nodes ~ gl_nodes_per_dim^2.

    max_physical_radius : float
        Physical radius R (or cap on R if auto_R_from_gl_nodes=True).

    nufft_tol : float
        FINUFFT accuracy tolerance (passed as eps=...).

    batch_frequencies : bool
        If True, evaluate frequencies in batches (shared quadrature + NUFFT per batch).

    chunk_frequencies : bool
        If True (and batch_frequencies=True), split w into multiple batches by binning.

    frequency_binning : {"log", "linear"}
        Controls how frequencies are binned when chunk_frequencies=True:
          - "log": bins in log10(|w|)
          - "linear": bins in |w|

    frequency_bin_width : float or None
        Bin width for chunking:
          - if frequency_binning="log": width in decades of log10(|w|)
          - if frequency_binning="linear": width in units of |w|
        If None, chunking is disabled (single batch).

    auto_R_from_gl_nodes : bool
        If True (Cartesian + batched), choose R per batch using:
            R_adapt = sqrt(gl_nodes_per_dim / (2 * w_use))
            R = min(max_physical_radius, R_adapt)
        where w_use = max(|w|) in the batch.
        If False, use R = max_physical_radius.

    use_tail_correction : bool
        If True, use the "(exp(-iwψ) - 1)" formulation and add +1 afterward.

    coordinate_system : {"cartesian", "polar"}
        Quadrature coordinate system. Note: chunking/auto_R currently only for cartesian.

    polar_radial_nodes, polar_angular_nodes : int
        Required if coordinate_system="polar".

    uniform_angular_sampling : bool
        If True, use gauss_legendre_polar_uniform_theta_2d; else gauss_legendre_polar_2d.

    numexpr_nthreads : int or None
        Thread count for numexpr only (not FINUFFT). Will be capped by NUMEXPR_MAX_THREADS and cores.

    verbose : bool
        Print configuration and chunk diagnostics.
    """

    def __init__(
        self,
        lens,
        gl_nodes_per_dim=128,
        max_physical_radius=12.0,
        nufft_tol=1e-12,
        batch_frequencies=True,
        chunk_frequencies=True,
        frequency_binning="log",     # "log" or "linear"
        frequency_bin_width=0.5,     # decades if log, |w| units if linear
        auto_R_from_gl_nodes=True,
        use_tail_correction=True,
        coordinate_system="cartesian",  # "cartesian" or "polar"
        polar_radial_nodes=None,
        polar_angular_nodes=None,
        uniform_angular_sampling=True,
        numexpr_nthreads=None,
        verbose=True,
    ):
        if not _FINUFFT:
            raise ImportError(
                "finufft is required for FresnelNUFFT3; install finufft or use FresnelDirect3."
            )

        self.lens = lens

        # Core numeric controls
        self.gl_nodes_per_dim = int(gl_nodes_per_dim)
        self.max_physical_radius = float(max_physical_radius)
        self.nufft_tol = float(nufft_tol)

        # Execution strategy
        self.batch_frequencies = bool(batch_frequencies)
        self.chunk_frequencies = bool(chunk_frequencies)

        if frequency_binning not in ("log", "linear"):
            raise ValueError("frequency_binning must be 'log' or 'linear'.")
        self.frequency_binning = frequency_binning

        self.frequency_bin_width = None if frequency_bin_width is None else float(frequency_bin_width)

        self.auto_R_from_gl_nodes = bool(auto_R_from_gl_nodes)

        # Model/derivation options
        self.use_tail_correction = bool(use_tail_correction)

        # Coordinates
        if coordinate_system not in ("cartesian", "polar"):
            raise ValueError("coordinate_system must be 'cartesian' or 'polar'.")
        self.coordinate_system = coordinate_system
        self.uniform_angular_sampling = bool(uniform_angular_sampling)

        if self.coordinate_system == "polar":
            if polar_radial_nodes is None or polar_angular_nodes is None:
                raise ValueError(
                    "When coordinate_system='polar', both polar_radial_nodes and "
                    "polar_angular_nodes must be specified."
                )
            self.polar_radial_nodes = int(polar_radial_nodes)
            self.polar_angular_nodes = int(polar_angular_nodes)
        else:
            # cartesian
            if self.gl_nodes_per_dim < 2:
                raise ValueError("gl_nodes_per_dim must be >= 2.")

        # Verbosity / reporting
        self.verbose = bool(verbose)

        # NumExpr threading (only affects numexpr evaluations, not FINUFFT)
        self._numexpr_nthreads = None
        if numexpr_nthreads is not None:
            requested = int(numexpr_nthreads)

            max_env = os.environ.get("NUMEXPR_MAX_THREADS")
            if max_env is not None:
                try:
                    max_env_int = int(max_env)
                except ValueError:
                    max_env_int = requested
            else:
                max_env_int = requested

            max_cores = ne.detect_number_of_cores()
            effective = min(requested, max_env_int, max_cores)
            effective = max(effective, 1)

            ne.set_num_threads(effective)
            self._numexpr_nthreads = effective

            if self.verbose:
                print(
                    f"[numexpr] using {effective} threads "
                    f"(requested={requested}, MAX={max_env_int}, cores={max_cores})"
                )
        else:
            self._numexpr_nthreads = ne.get_num_threads()
            if self.verbose:
                print(f"[numexpr] using default thread count: {self._numexpr_nthreads}")

        # Current limitation guard (matches your existing behavior)
        if self.coordinate_system == "polar" and (self.auto_R_from_gl_nodes or self.chunk_frequencies):
            raise NotImplementedError(
                "chunk_frequencies/auto_R_from_gl_nodes are currently implemented only for "
                "coordinate_system='cartesian' (Cartesian Gauss-Legendre in u1,u2)."
            )

    def __call__(self, w, y1, y2, verbose=None):
        """
        Evaluate F(w, y) for frequencies w and target coordinates (y1, y2).

        Parameters
        ----------
        w : array_like
            Frequencies (nonzero). Can be scalar or vector.
        y1, y2 : array_like
            Target coordinates; must have the same shape.
        verbose : bool or None
            If None, uses self.verbose.
        """
        if verbose is None:
            verbose = self.verbose

        w_vec = np.asarray(w, dtype=float).ravel()
        if np.any(w_vec == 0.0):
            raise ValueError("All w must be nonzero.")

        y1 = np.asarray(y1, dtype=float)
        y2 = np.asarray(y2, dtype=float)
        if y1.shape != y2.shape:
            raise ValueError("y1 and y2 must have the same shape.")

        # Precompute target quadratic phase once (used in both modes)
        quad_phase = (y1**2 + y2**2) / 2.0

        # =========================
        # Batched (shared-geometry) path
        # =========================
        if self.batch_frequencies:
            # ---- build groups (chunking independent of auto_R) ----
            if (
                self.chunk_frequencies
                and (self.frequency_bin_width is not None)
                and (len(w_vec) > 1)
            ):
                absw = np.abs(w_vec)

                if self.frequency_bin_width <= 0.0:
                    groups = [np.arange(len(w_vec))]
                else:
                    if self.frequency_binning == "log":
                        # bin in log10(|w|)
                        logw = np.log10(absw)
                        vmin = float(logw.min())
                        vmax = float(logw.max())
                        span = vmax - vmin

                        if span <= 0.0:
                            groups = [np.arange(len(w_vec))]
                        else:
                            n_bins = int(np.ceil(span / self.frequency_bin_width))
                            n_bins = max(1, n_bins)
                            edges = vmin + self.frequency_bin_width * np.arange(n_bins + 1, dtype=float)
                            edges[-1] = vmax + 1e-12  # include right edge
                            bin_idx = np.digitize(logw, edges) - 1
                            groups = [np.where(bin_idx == k)[0] for k in range(n_bins)]
                            groups = [g for g in groups if g.size > 0]
                    else:
                        # bin in linear |w|
                        vmin = float(absw.min())
                        vmax = float(absw.max())
                        span = vmax - vmin

                        if span <= 0.0:
                            groups = [np.arange(len(w_vec))]
                        else:
                            n_bins = int(np.ceil(span / self.frequency_bin_width))
                            n_bins = max(1, n_bins)
                            edges = vmin + self.frequency_bin_width * np.arange(n_bins + 1, dtype=float)
                            edges[-1] = vmax + 1e-12  # include right edge
                            bin_idx = np.digitize(absw, edges) - 1
                            groups = [np.where(bin_idx == k)[0] for k in range(n_bins)]
                            groups = [g for g in groups if g.size > 0]
            else:
                groups = [np.arange(len(w_vec))]

            F_out = np.empty((len(w_vec),) + y1.shape, dtype=np.complex128)

            for chunk_idx, g in enumerate(groups):
                w_sub = w_vec[g]
                w_use = float(np.max(np.abs(w_sub)))  # representative for sizing

                # ---- choose R and Umax for this chunk ----
                if self.auto_R_from_gl_nodes:
                    # Keep gl_nodes_per_dim fixed and choose R from the chunk's max |w|
                    R_adapt = np.sqrt(self.gl_nodes_per_dim / (2.0 * w_use))
                    R = min(self.max_physical_radius, float(R_adapt))
                else:
                    # Fixed physical window radius
                    R_adapt = None
                    R = self.max_physical_radius

                Umax = w_use * R
                h = np.pi / Umax

                # ---- reporting ----
                if verbose:
                    w_abs = np.abs(w_sub)
                    w_min = float(w_abs.min())
                    w_max = float(w_abs.max())

                    if self.chunk_frequencies and (self.frequency_bin_width is not None):
                        if self.frequency_binning == "log":
                            bin_msg = f" | binning=log10(|w|), Δ={self.frequency_bin_width:g} decades"
                        else:
                            bin_msg = f" | binning=linear(|w|), Δ={self.frequency_bin_width:g}"
                    else:
                        bin_msg = ""

                    if self.auto_R_from_gl_nodes:
                        capped = (R < float(R_adapt) - 1e-15)
                        cap_msg = " (capped by max_physical_radius)" if capped else ""
                        msg = (
                            f"[chunk {chunk_idx+1}/{len(groups)}] "
                            f"count={len(w_sub)} | |w| in [{w_min:.6g}, {w_max:.6g}] | "
                            f"N={self.gl_nodes_per_dim} fixed -> "
                            f"R={R:.6g}{cap_msg}, Umax={Umax:.6g}, h={h:.6g}"
                        )
                    else:
                        msg = (
                            f"[chunk {chunk_idx+1}/{len(groups)}] "
                            f"count={len(w_sub)} | |w| in [{w_min:.6g}, {w_max:.6g}] | "
                            f"R={R:.6g} fixed, N={self.gl_nodes_per_dim} (per dim), "
                            f"Umax={Umax:.6g}, h={h:.6g}"
                        )
                    print(msg + bin_msg)

                # ---- 1) Quadrature nodes/weights in u-space (u1,u2 in [-Umax, Umax]) ----
                u1, u2, W = gauss_legendre_2d(self.gl_nodes_per_dim, Umax)

                # ---- 2) NUFFT scalings (invariant product: (h*u) * (y/h) = u*y) ----
                xj = h * u1
                yj = h * u2
                sk = y1 / h
                tk = y2 / h

                # ---- 3) Build coefficients cj for all w in this chunk ----
                w2d = w_sub[:, None]    # (M, 1)
                u1_2d = u1[None, :]     # (1, N)
                u2_2d = u2[None, :]     # (1, N)
                W_2d = W[None, :]       # (1, N)

                # u/w for lens evaluation
                u1_over_w = ne.evaluate("u1_2d / w2d")
                u2_over_w = ne.evaluate("u2_2d / w2d")
                psi = self.lens.psi_xy(u1_over_w, u2_over_w)  # (M, N)

                base_quad = (u1 * u1 + u2 * u2) / 2.0     # (N,)
                base_quad_2d = base_quad[None, :]         # (1, N)

                if self.use_tail_correction:
                    # cj = exp(i * (u^2/(2w))) * (exp(-i*w*psi) - 1) * W
                    phase_quad = ne.evaluate("base_quad_2d / w2d")
                    cosq = ne.evaluate("cos(phase_quad)")
                    sinq = ne.evaluate("sin(phase_quad)")
                    exp_quad = cosq + 1j * sinq

                    phase_lens = ne.evaluate("-w2d * psi")
                    cosl = ne.evaluate("cos(phase_lens)")
                    sinl = ne.evaluate("sin(phase_lens)")
                    exp_lens = cosl + 1j * sinl

                    cj = ne.evaluate("W_2d") * exp_quad * (exp_lens - 1.0)
                else:
                    # cj = exp(i * (u^2/(2w) - w*psi)) * W
                    phase = ne.evaluate("base_quad_2d / w2d - w2d * psi")
                    cosp = ne.evaluate("cos(phase)")
                    sinp = ne.evaluate("sin(phase)")
                    cj = (cosp + 1j * sinp) * W_2d

                # ---- 4) NUFFT ----
                I = nufft2d3(xj, yj, cj, sk, tk, isign=-1, eps=self.nufft_tol)

                # ---- 5) Fresnel prefactor (+ optional "+1" tail correction) ----
                for i, w_i in enumerate(w_sub):
                    val = np.exp(1j * w_i * quad_phase) * I[i] / (1j * w_i * _TWO_PI)
                    if self.use_tail_correction:
                        val = 1.0 + val
                    F_out[g[i]] = val

            return F_out

        # =========================
        # Per-frequency path (independent geometry per w)
        # =========================
        F_list = []
        for w_i in w_vec:
            Umax = abs(w_i) * self.max_physical_radius

            if self.coordinate_system == "cartesian":
                u1, u2, W = gauss_legendre_2d(self.gl_nodes_per_dim, Umax)
            else:
                # coordinate_system == "polar"
                if self.uniform_angular_sampling:
                    r, theta, W = gauss_legendre_polar_uniform_theta_2d(
                        self.polar_radial_nodes,
                        self.polar_angular_nodes,
                        Umax,
                    )
                else:
                    r, theta, W = gauss_legendre_polar_2d(
                        self.polar_radial_nodes,
                        self.polar_angular_nodes,
                        Umax,
                    )
                u1 = r * np.cos(theta)
                u2 = r * np.sin(theta)

            h = np.pi / Umax
            xj = h * u1
            yj = h * u2
            sk = y1 / h
            tk = y2 / h

            phase = (u1 * u1 + u2 * u2) / (2.0 * w_i) - w_i * self.lens.psi_xy(u1 / w_i, u2 / w_i)
            cj = np.exp(1j * phase) * W

            I = nufft2d3(xj, yj, cj, sk, tk, isign=-1, eps=self.nufft_tol)
            Fw = np.exp(1j * w_i * quad_phase) * I / (1j * w_i * _TWO_PI)
            if self.use_tail_correction:
                Fw = 1.0 + Fw

            F_list.append(Fw)

        return np.stack(F_list, axis=0)
