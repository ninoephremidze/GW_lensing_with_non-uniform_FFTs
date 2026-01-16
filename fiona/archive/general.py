####################################################################
# fiona/general.py
####################################################################

import os
import time
import numpy as np
import numexpr as ne

from .utils import (
    gauss_legendre_2d,
    gauss_legendre_polar_2d,
    gauss_legendre_polar_uniform_theta_2d,
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
    def __init__(
        self,
        lens,
        n_gl=128,
        Umax=12.0,
        eps=1e-12,
        shared_Umax=True,
        numexpr_threads=None,
        track_perf=True,
        polar_coord=False,
        n_r=None,
        n_theta=None,
        uniform_theta=True,
        analytic_tail=True
    ):

        if not _FINUFFT:
            raise ImportError(
                "finufft is required for FresnelNUFFT3Vec; "
                "install finufft or use FresnelDirect3."
            )

        self.lens = lens
        self.Umax = float(Umax)
        self.eps = float(eps)
        self.shared_Umax = bool(shared_Umax)
        self.polar_coord = bool(polar_coord)
        self.uniform_theta = bool(uniform_theta)
        self.analytic_tail = bool(analytic_tail)

        if self.polar_coord:
            if n_r is None or n_theta is None:
                raise ValueError(
                    "When polar_coord=True, both n_r and n_theta must be specified."
                )
            self.n_r = int(n_r)
            self.n_theta = int(n_theta)
            
        else:
            self.n_gl = int(n_gl)


        self._numexpr_threads = None
        if numexpr_threads is not None:
            requested = int(numexpr_threads)
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
            if effective < 1:
                effective = 1

            ne.set_num_threads(effective)
            self._numexpr_threads = effective
            if track_perf:
                print(f"[numexpr] using {effective} threads "
                      f"(requested={requested}, MAX={max_env_int}, cores={max_cores})")
        else:
            self._numexpr_threads = ne.get_num_threads()
            if track_perf:
                print(f"[numexpr] using default thread count: {self._numexpr_threads}")

    def __call__(self, w_vec, y1_targets, y2_targets, track_perf=True):

        w_vec = np.asarray(w_vec, dtype=float).ravel()
        if np.any(w_vec == 0):
            raise ValueError("All w must be nonzero.")

        y1_targets = np.asarray(y1_targets, dtype=float)
        y2_targets = np.asarray(y2_targets, dtype=float)
        if y1_targets.shape != y2_targets.shape:
            raise ValueError("y1_targets and y2_targets must have the same shape.")

        if self.shared_Umax:
            
            t0 = time.perf_counter()

            Umax = self.Umax

            if not self.polar_coord:
                u1, u2, W = gauss_legendre_2d(self.n_gl, Umax)
            else:
                if self.uniform_theta:
                    r, theta, W = gauss_legendre_polar_uniform_theta_2d(
                        self.n_r, self.n_theta, Umax)
                else:
                    r, theta, W = gauss_legendre_polar_2d(
                        self.n_r, self.n_theta, Umax)

                u1 = r * np.cos(theta)
                u2 = r * np.sin(theta)

            t1 = time.perf_counter()

            with CPUTracker() as _c_total:
                t2_0 = time.perf_counter()

                # 2a) simple scalings independent of w
                t2a_0 = time.perf_counter()
                h = np.pi / Umax
                xj = h * u1
                yj = h * u2
                sk = y1_targets / h
                tk = y2_targets / h
                t2a_1 = time.perf_counter()

                # 2b) coefficient computation over all w
                t2b_0 = time.perf_counter()

                # shapes:     u1, u2, W: (N,)
                #             w_vec:     (M,)
                # we build (M, N) arrays for u1_over_w, u2_over_w, psi, phase, cj

                M = len(w_vec)
                N = u1.size

                w2d = w_vec[:, None]          # (M, 1)
                u1_2d = u1[None, :]           # (1, N)
                u2_2d = u2[None, :]           # (1, N)
                W_2d = W[None, :]             # (1, N)

                # --- scale u/w for all w (using numexpr) ---
                scale_uw_0 = time.perf_counter()
                # u1_over_w = u1_2d / w2d
                # u2_over_w = u2_2d / w2d
                u1_over_w = ne.evaluate("u1_2d / w2d")      # (M, N)
                u2_over_w = ne.evaluate("u2_2d / w2d")      # (M, N)
                scale_uw_1 = time.perf_counter()

                # --- psi_xy for all w ---
                psi_0 = time.perf_counter()
                psi = self.lens.psi_xy(u1_over_w, u2_over_w)   # (M, N)
                psi_1 = time.perf_counter()

                # --- phase calculation for all w (using numexpr) ---
                phase_0 = time.perf_counter()
                base_quad = (u1 * u1 + u2 * u2) / 2.0          # (N,)
                base_quad_2d = base_quad[None, :]              # (1, N)
                phase = ne.evaluate("base_quad_2d / w2d - w2d * psi")  # (M, N)
                phase_1 = time.perf_counter()

                # --- exponential for all w (numexpr via cos/sin) ---
                exp_0 = time.perf_counter()
                # exp(i * phase) = cos(phase) + i sin(phase)
                cos_phase = ne.evaluate("cos(phase)")          # (M, N)
                sin_phase = ne.evaluate("sin(phase)")          # (M, N)
                re = ne.evaluate("cos_phase * W_2d")           # (M, N)
                im = ne.evaluate("sin_phase * W_2d")           # (M, N)
                cj = re + 1j * im                              # (M, N) complex128
                exp_1 = time.perf_counter()

                t2b_1 = time.perf_counter()
                t2_1 = time.perf_counter()

            # fine-grained timings
            scale_uw_time = scale_uw_1 - scale_uw_0
            psi_time = psi_1 - psi_0
            phase_time = phase_1 - phase_0
            exp_time = exp_1 - exp_0
            loop_wall = t2b_1 - t2b_0
            sub_total = scale_uw_time + psi_time + phase_time + exp_time
            unaccounted = loop_wall - sub_total

            t2 = time.perf_counter()

            # ------------------------------------------------------------------
            # 3) NUFFT
            # ------------------------------------------------------------------
            with CPUTracker() as _n:
                I = nufft2d3(xj, yj, cj, sk, tk, isign=-1, eps=self.eps)
            if track_perf:
                print(_n.report("[nufft]"))
            t3 = time.perf_counter()

            # ------------------------------------------------------------------
            # 4) quad phase + allocate output
            # ------------------------------------------------------------------
            quad_phase = (y1_targets ** 2 + y2_targets ** 2) / 2.0
            F = np.empty_like(I, dtype=np.complex128)
            t4 = time.perf_counter()

            # ------------------------------------------------------------------
            # 5) final per-frequency multiplication
            # ------------------------------------------------------------------
            for i, w in enumerate(w_vec):
                F[i] = np.exp(1j * w * quad_phase) * I[i] / (1j * w * _TWO_PI)
            t5 = time.perf_counter()

            def pct(part, whole):
                return (part / whole * 100.0) if whole > 0 else 0.0

            step2_total = (t2_1 - t2_0)
            total_time = (t5 - t0)

            
            if track_perf:
                
                print()
                print("────────────────────────────────────────────────────────────")
                print(" Step 2: Coefficient Computation (Vectorized + NumExpr)")
                print("────────────────────────────────────────────────────────────")
                print(_c_total.report("  CPU usage summary"))
                print()

                print("  Breakdown (percent of Step 2):")
                print(f"    2a. Pre-scaling (host)          : "
                      f"{pct(t2a_1 - t2a_0, step2_total):6.2f}%  ({t2a_1 - t2a_0:10.6f} s)")

                print()
                print(f"    2b. Coefficient build (total)   : "
                      f"{pct(loop_wall, step2_total):6.2f}%  ({loop_wall:10.6f} s)")

                print(f"        ├─ scale u/w (all w)        : "
                      f"{pct(scale_uw_time, step2_total):6.2f}%  ({scale_uw_time:10.6f} s)")
                print(f"        ├─ lens potential ψ(x)      : "
                      f"{pct(psi_time, step2_total):6.2f}%  ({psi_time:10.6f} s)")
                print(f"        ├─ phase calculation        : "
                      f"{pct(phase_time, step2_total):6.2f}%  ({phase_time:10.6f} s)")
                print(f"        ├─ exp(i·phase)             : "
                      f"{pct(exp_time, step2_total):6.2f}%  ({exp_time:10.6f} s)")
                print(f"        └─ unaccounted              : "
                      f"{pct(unaccounted, step2_total):6.2f}%  ({unaccounted:10.6f} s)")

                print()
                print(f"  Step 2 total                      : "
                      f"{pct(step2_total, step2_total):6.2f}%  ({step2_total:10.6f} s)")
                print("────────────────────────────────────────────────────────────")
                print()

                # -------------------------------------------------------------
                # Overall run summary
                # -------------------------------------------------------------
                print("Overall Timing Summary (percent of TOTAL)")
                print("────────────────────────────────────────────────────────────")

                print(f"  1. Quadrature setup               : "
                      f"{pct(t1 - t0, total_time):6.2f}%  ({t1 - t0:10.6f} s)")

                print(f"  2. Coefficients (total)           : "
                      f"{pct(t2 - t1, total_time):6.2f}%  ({t2 - t1:10.6f} s)")

                print(f"  3. NUFFT                          : "
                      f"{pct(t3 - t2, total_time):6.2f}%  ({t3 - t2:10.6f} s)")

                print(f"  4. quad_phase + alloc             : "
                      f"{pct(t4 - t3, total_time):6.2f}%  ({t4 - t3:10.6f} s)")

                print(f"  5. final per-w loop               : "
                      f"{pct(t5 - t4, total_time):6.2f}%  ({t5 - t4:10.6f} s)")

                print("────────────────────────────────────────────────────────────")
                print(f"  TOTAL                             : "
                      f"{pct(total_time, total_time):6.2f}%  ({total_time:10.6f} s)")
                print("────────────────────────────────────────────────────────────")
                print()
                
            if self.analytic_tail == True:

                # ==============================================================
                # Analytic free-Fresnel tail via subtraction
                #   F_tail = 1 - F_free_core
                # ==============================================================

                # Allocate free-core array
                F_free_core = np.empty_like(F)

                # ---- build free (psi = 0) coefficients ----
                # phase_free = u^2 / (2w)

                phase_free = ne.evaluate("base_quad_2d / w2d") # (M, N)

                cos_phase_free = ne.evaluate("cos(phase_free)")
                sin_phase_free = ne.evaluate("sin(phase_free)")

                # cj_free = exp(i * phase_free) * W
                re_free = ne.evaluate("cos_phase_free * W_2d")
                im_free = ne.evaluate("sin_phase_free * W_2d")
                cj_free = re_free + 1j * im_free               # (M, N)

                # ---- NUFFT for free core ----
                with CPUTracker() as _n_free:
                    I_free = nufft2d3(
                        xj, yj, cj_free, sk, tk,
                        isign=-1, eps=self.eps
                    )

                # ---- apply Fresnel prefactor ----
                for i, w in enumerate(w_vec):
                    F_free_core[i] = (
                        np.exp(1j * w * quad_phase)
                        * I_free[i]
                        / (1j * w * _TWO_PI)
                    )

                # ---- analytic tail ----
                F_tail = 1.0 - F_free_core

                # ---- full Fresnel integral ----
                F = F + F_tail

            return F

        else:
            # ------------------------------------------------------------------
            # Per-w quadrature (no multiprocessing)
            # ------------------------------------------------------------------
            F_list = []
            for w in w_vec:
                Umax = abs(w) * self.Umax

            if not self.polar_coord:
                u1, u2, W = gauss_legendre_2d(self.n_gl, Umax)
            else:
                if self.uniform_theta:
                    r, theta, W = gauss_legendre_polar_uniform_theta_2d(
                        self.n_r, self.n_theta, Umax
                    )
                else:
                    r, theta, W = gauss_legendre_polar_2d(
                        self.n_r, self.n_theta, Umax
                    )

                u1 = r * np.cos(theta)
                u2 = r * np.sin(theta)

                h = np.pi / Umax
                xj = h * u1
                yj = h * u2
                sk = y1_targets / h
                tk = y2_targets / h

                phase = (u1 * u1 + u2 * u2) / (2.0 * w) - w * self.lens.psi_xy(
                    u1 / w, u2 / w
                )
                cj = np.exp(1j * phase) * W

                I = nufft2d3(xj, yj, cj, sk, tk, isign=-1, eps=self.eps)
                Fw = (
                    np.exp(1j * w * (y1_targets ** 2 + y2_targets ** 2) / 2.0)
                    * I
                    / (1j * w * _TWO_PI)
                )
                F_list.append(Fw)

            return np.stack(F_list, axis=0)
