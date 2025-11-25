import numpy as np
import time
from .utils import gauss_legendre_2d, CPUTracker

try:
    from finufft import nufft2d3
    _FINUFFT = True
except Exception:
    _FINUFFT = False

_TWO_PI = 2.0*np.pi

import psutil
import threading
import time
import os, math
import multiprocessing as mp
from threadpoolctl import threadpool_limits

_G_u1 = _G_u2 = _G_W = _G_lens = None

def _init_worker(u1, u2, W, lens):
    
    # globals let us avoid re-sending big arrays each task when using 'fork'
    global _G_u1, _G_u2, _G_W, _G_lens
    _G_u1, _G_u2, _G_W, _G_lens = u1, u2, W, lens
    # ensure math inside worker uses 1 thread (MKL/OpenBLAS/NumPy ufuncs)
    threadpool_limits(limits=1)

def _compute_cj_for_w(w):
    
    u1, u2, W, lens = _G_u1, _G_u2, _G_W, _G_lens
    phase = (u1*u1 + u2*u2)/(2.0*w) - w*lens.psi_xy(u1/w, u2/w)
    return np.exp(1j*phase) * W

def _integrand_coeffs(u1, u2, w, lens, W):
    """
    I(y) = ∫ d^2u e^{-i u·y} exp{i [u^2/(2w) - w ψ(u/w)]}.
    """
    phase = (u1*u1 + u2*u2) / (2.0*w) - w * lens.psi_xy(u1/w, u2/w)
    return np.exp(1j * phase) * W

class FresnelNUFFT3Vec:
    r"""
    Batched version of FresnelNUFFT3 for multiple frequencies w.

    If shared_Umax=True (default), uses a single quadrature grid with
    Umax = max(|w|)*Xmax. If shared_Umax=False, evaluates each w with its
    own quadrature (slower, more accurate when low w require large Xmax).
    """
    def __init__(self, lens, n_gl=128, Umax=12.0, eps=1e-12, shared_Umax=True):
        
        if not _FINUFFT:
            raise ImportError("finufft is required for FresnelNUFFT3Vec; install finufft or use FresnelDirect3.")
        
        self.lens = lens
        self.n_gl = int(n_gl)
        self.Umax = float(Umax)
        self.eps  = float(eps)
        self.shared_Umax = bool(shared_Umax)

    def __call__(self, w_vec, y1_targets, y2_targets):
        
        w_vec = np.asarray(w_vec, dtype=float).ravel()
        if np.any(w_vec == 0):
            raise ValueError("All w must be nonzero.")
        y1_targets = np.asarray(y1_targets, dtype=float)
        y2_targets = np.asarray(y2_targets, dtype=float)
        if y1_targets.shape != y2_targets.shape:
            raise ValueError("y1_targets and y2_targets must have the same shape.")

        if self.shared_Umax:
            t0 = time.perf_counter()

            # 1) quadrature setup
            Umax = self.Umax
            u1, u2, W = gauss_legendre_2d(self.n_gl, Umax)
            t1 = time.perf_counter()

            #2) scales/phases/coeffs
            
            with CPUTracker() as _c:
                
                h  = np.pi / Umax
                xj = h * u1
                yj = h * u2
                sk = y1_targets / h
                tk = y2_targets / h

                ctx = mp.get_context("fork")
                n_procs = min(len(w_vec), 112)
                chunk = max(1, len(w_vec) // (n_procs * 4))

                with ctx.Pool(processes=n_procs, initializer=_init_worker,
                              initargs=(u1, u2, W, self.lens)) as pool:
                    cj_iter = pool.imap(_compute_cj_for_w, map(float, w_vec), chunksize=chunk)
                    cj_list = list(cj_iter)

                cj = np.stack(cj_list, axis=0)
            
            print(_c.report("[coeffs]"))
                
            t2 = time.perf_counter()

            # 3) NUFFT
            with CPUTracker() as _n:
                I = nufft2d3(xj, yj, cj, sk, tk, isign=-1, eps=self.eps)
            print(_n.report("[nufft]"))
            t3 = time.perf_counter()

            # 4) quad phase + allocate output
            quad_phase = (y1_targets**2 + y2_targets**2)/2.0
            F = np.empty_like(I, dtype=np.complex128)
            t4 = time.perf_counter()

            # 5) final per-frequency multiplication
            for i, w in enumerate(w_vec):
                F[i] = np.exp(1j*w*quad_phase) * I[i] / (1j*w*_TWO_PI)
            t5 = time.perf_counter()

            # Prints
            print(f"[timing] 1. quadrature setup:        {t1 - t0:.6f} s")
            print(f"[timing] 2. scales/phases/coeffs:    {t2 - t1:.6f} s")
            print(f"[timing] 3. NUFFT:                   {t3 - t2:.6f} s")
            print(f"[timing] 4. quad_phase+alloc:        {t4 - t3:.6f} s")
            print(f"[timing] 5. final loop:              {t5 - t4:.6f} s")
            print(f"[timing] total:                      {t5 - t0:.6f} s")

            return F

        else:
            # Per-w quadrature
            F_list = []
            for w in w_vec:
                Umax = abs(w) * self.Xmax
                u1, u2, W = gauss_legendre_2d(self.n_gl, Umax)

                h  = np.pi / Umax
                xj = h * u1
                yj = h * u2
                sk = y1_targets / h
                tk = y2_targets / h

                phase = (u1*u1 + u2*u2)/(2.0*w) - w*self.lens.psi_xy(u1/w, u2/w)
                cj    = np.exp(1j*phase) * W

                I  = nufft2d3(xj, yj, cj, sk, tk, isign=-1, eps=self.eps)
                Fw = np.exp(1j*w*(y1_targets**2 + y2_targets**2)/2.0) * I / (1j*w*_TWO_PI)
                F_list.append(Fw)
            return np.stack(F_list, axis=0)