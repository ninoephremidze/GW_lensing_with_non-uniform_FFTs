import os
import json
import math
import pathlib
import hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from multiprocessing import Pool
import time

try:
    import resource
    _HAS_RESOURCE = True
except Exception:
    _HAS_RESOURCE = False
import psutil

_FIFT_GL2D_DIR = os.environ.get("FIFT_GL2D_DIR", "")
_FIFT_GL2D_STRICT = os.environ.get("FIFT_GL2D_STRICT", "0") == "1"

def _ensure_store_dir():
    if not _FIFT_GL2D_DIR:
        return None
    p = pathlib.Path(_FIFT_GL2D_DIR)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return p

def _gl2d_base(n, u_max):
    store = _ensure_store_dir()
    if store is None:
        return None
    base = store / f"gl2d_n{int(n)}_U{int(u_max)}"
    return base

def _gl2d_paths(n, u_max):
    base = _gl2d_base(n, u_max)
    if base is None:
        return None
    return {
        "base": base,
        "u1":   base.with_suffix(".u1.npy"),
        "u2":   base.with_suffix(".u2.npy"),
        "W":    base.with_suffix(".W.npy"),
        "meta": base.with_suffix(".meta.json"),
    }

def _gl2d_exists(n, u_max):
    p = _gl2d_paths(n, u_max)
    if p is None:
        return False
    return all(pathlib.Path(p[k]).exists() for k in ("u1","u2","W","meta"))


def gauss_legendre_2d(n, u_max):
    """
    Build or load 2D Gauss–Legendre nodes/weights on [-u_max, u_max]^2.

    Returns flattened arrays (u1, u2, W) of length n*n.

    If FIFT_GL2D_DIR is set and a matching dataset exists, loads it via memmap.
    If strict mode is enabled (FIFT_GL2D_STRICT=1) and not found, raises.
    Otherwise computes in-process (serial) as a last resort.
    """
    n = int(n); u_max = int(u_max)

    if _FIFT_GL2D_DIR and _gl2d_exists(n, u_max):
        p = _gl2d_paths(n, u_max)
        u1 = np.memmap(p["u1"], dtype=np.float64, mode="r", shape=(n*n,), order="C")
        u2 = np.memmap(p["u2"], dtype=np.float64, mode="r", shape=(n*n,), order="C")
        W  = np.memmap(p["W"],  dtype=np.float64, mode="r", shape=(n*n,), order="C")
        return u1, u2, W

    raise FileNotFoundError(
        f"No precomputed GL2D files for (n={n}, Umax={u_max}). "
        f"Set FIFT_GL2D_DIR and precompute, or unset FIFT_GL2D_STRICT.")

    # # Fallback: compute now (serial) to keep API working.
    # xi, wi = leggauss(n)
    # x = u_max * xi
    # w = u_max * wi
    # u1g, u2g = np.meshgrid(x, x, indexing="xy")
    # W1g, W2g = np.meshgrid(w, w, indexing="xy")
    # u1 = u1g.ravel()
    # u2 = u2g.ravel()
    # W  = (W1g * W2g).ravel()
    return u1, u2, W

# =============================================================================
# Complex interpolation helper (unchanged)
# =============================================================================
def interp_complex_logx(x, xp, fp):
    xr = np.interp(np.log(x), np.log(xp), np.real(fp))
    xi = np.interp(np.log(x), np.log(xp), np.imag(fp))
    return xr + 1j*xi

class CPUTracker:
    """
    Measure average parallelism over a code block:
      effective_cores = CPU_seconds / wall_seconds

    Counts the current process' CPU time, and (if available) also includes
    CPU time spent by child processes that start and finish during the block.
    """
    def __init__(self, include_children=True):
        self.include_children = include_children

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._p0 = time.process_time()
        if _HAS_RESOURCE:
            r_self0   = resource.getrusage(resource.RUSAGE_SELF)
            r_child0  = resource.getrusage(resource.RUSAGE_CHILDREN)
            self._r0s = (r_self0.ru_utime + r_self0.ru_stime,
                         r_child0.ru_utime + r_child0.ru_stime)
        else:
            self._r0s = None
        return self

    def __exit__(self, exc_type, exc, tb):
        self._t1 = time.perf_counter()
        self._p1 = time.process_time()
        wall = max(1e-12, self._t1 - self._t0)

        if self._r0s is not None:
            r_self1  = resource.getrusage(resource.RUSAGE_SELF)
            r_child1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu = ((r_self1.ru_utime + r_self1.ru_stime) - self._r0s[0]) \
                + ((r_child1.ru_utime + r_child1.ru_stime) - self._r0s[1])
        else:
            # Portable fallback: process CPU time only (threads included, no children)
            cpu = max(0.0, self._p1 - self._p0)

        self.wall = wall
        self.cpu  = max(0.0, cpu)
        n = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        self.effective_cores = self.cpu / self.wall
        self.avg_cpu_percent = 100.0 * self.effective_cores / n
        self.n_logical = n

    def report(self, label="[CPU]"):
        return (f"{label} avg {self.effective_cores:.2f} cores over {self.wall:.3f}s "
                f"({self.avg_cpu_percent:.1f}% of {self.n_logical} logical cores; "
                f"CPU sec={self.cpu:.3f})")
