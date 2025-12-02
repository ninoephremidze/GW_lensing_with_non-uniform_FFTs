"""Python wrapper for the cfastfht shared library."""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_LIB_CANDIDATES = []

_env_path = os.environ.get("CFASTFHT_LIB")
if _env_path:
    _LIB_CANDIDATES.append(Path(_env_path))

_repo_root = Path(__file__).resolve().parent.parent
_default_build = _repo_root / "cfastfht" / "build" / "libcfastfht.so"
_LIB_CANDIDATES.append(_default_build)

class _CFHTOptions(ctypes.Structure):
    _fields_ = [
        ("max_levels", ctypes.c_int),
        ("min_dim_prod", ctypes.c_size_t),
        ("z_split", ctypes.c_double),
        ("K_asy", ctypes.c_int),
        ("K_loc", ctypes.c_int),
    ]


def _load_library() -> ctypes.CDLL:
    errors = []
    for candidate in _LIB_CANDIDATES:
        if not candidate:
            continue
        if candidate.suffix != ".so":
            # allow specifying directory
            path = candidate
            if path.is_dir():
                candidate_path = path / "libcfastfht.so"
            else:
                candidate_path = path
        else:
            candidate_path = candidate
        if not candidate_path.exists():
            continue
        try:
            return ctypes.CDLL(str(candidate_path))
        except OSError as exc:
            errors.append(f"{candidate_path}: {exc}")
    found = ctypes.util.find_library("cfastfht")
    if found:
        try:
            return ctypes.CDLL(found)
        except OSError as exc:  # pragma: no cover - propagated later
            errors.append(f"{found}: {exc}")
    msg = "Unable to load libcfastfht.so. "
    if errors:
        msg += " Tried: " + "; ".join(errors)
    raise ImportError(msg)


try:
    _LIB = _load_library()
except Exception as exc:  # pragma: no cover - import-time error
    raise


_LIB.cfastfht_plan_create.restype = ctypes.c_void_p
_LIB.cfastfht_plan_create.argtypes = [
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
    ctypes.c_double,
    ctypes.POINTER(_CFHTOptions),
]

_LIB.cfastfht_plan_destroy.argtypes = [ctypes.c_void_p]
_LIB.cfastfht_plan_destroy.restype = None

_LIB.cfastfht_plan_execute.restype = ctypes.c_int
_LIB.cfastfht_plan_execute.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]

_LIB.cfastfht_plan_execute_batch.restype = ctypes.c_int
_LIB.cfastfht_plan_execute_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
    ctypes.c_size_t,
]

_LIB.cfastfht_last_error.restype = ctypes.c_char_p


def _last_error() -> str:
    msg = _LIB.cfastfht_last_error()
    if not msg:
        return "Unknown cfastfht error"
    return msg.decode("utf-8", errors="ignore")


class CNUFHTPlan:
    """Thin wrapper over cfastfht_plan."""

    def __init__(self,
                 nu: float,
                 rs: np.ndarray,
                 ws: np.ndarray,
                 tol: float,
                 max_levels: Optional[int] = None,
                 min_dim_prod: Optional[int] = None,
                 z_split: Optional[float] = None,
                 K_asy: Optional[int] = None,
                 K_loc: Optional[int] = None) -> None:
        if nu != 0:
            raise ValueError("Current cfastfht backend only supports nu = 0")
        self._rs = np.ascontiguousarray(rs, dtype=np.float64)
        self._ws = np.ascontiguousarray(ws, dtype=np.float64)
        if self._rs.ndim != 1 or self._ws.ndim != 1:
            raise ValueError("rs and ws must be 1-D arrays")
        opts = _CFHTOptions()
        opts.max_levels = -1 if max_levels is None else int(max_levels)
        opts.min_dim_prod = 0 if min_dim_prod is None else int(min_dim_prod)
        opts.z_split = math.nan if z_split is None else float(z_split)
        opts.K_asy = -2 if K_asy is None else int(K_asy)
        opts.K_loc = -2 if K_loc is None else int(K_loc)
        rs_ptr = self._rs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ws_ptr = self._ws.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        options_ptr = ctypes.byref(opts)
        plan_ptr = _LIB.cfastfht_plan_create(
            ctypes.c_double(0.0),
            rs_ptr, ctypes.c_size_t(self._rs.size),
            ws_ptr, ctypes.c_size_t(self._ws.size),
            ctypes.c_double(tol),
            options_ptr,
        )
        if not plan_ptr:
            raise RuntimeError(_last_error())
        self._handle = ctypes.c_void_p(plan_ptr)
        self.n_sources = self._rs.size
        self.n_targets = self._ws.size

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            _LIB.cfastfht_plan_destroy(handle)
            self._handle = None

    def _ensure_open(self) -> ctypes.c_void_p:
        handle = getattr(self, "_handle", None)
        if not handle:
            raise RuntimeError("Plan already closed")
        return handle

    def execute(self, coeffs: np.ndarray) -> np.ndarray:
        coeffs_arr = np.ascontiguousarray(coeffs, dtype=np.float64)
        if coeffs_arr.ndim != 1 or coeffs_arr.size != self.n_sources:
            raise ValueError("coeffs must have shape (n_sources,)")
        out = np.zeros(self.n_targets, dtype=np.float64)
        handle = self._ensure_open()
        rc = _LIB.cfastfht_plan_execute(
            handle,
            coeffs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(_last_error())
        return out

    def execute_batch(self, coeffs: np.ndarray) -> np.ndarray:
        coeffs_arr = np.ascontiguousarray(coeffs, dtype=np.float64)
        if coeffs_arr.ndim != 2 or coeffs_arr.shape[1] != self.n_sources:
            raise ValueError("coeffs must have shape (batch, n_sources)")
        batch_size = coeffs_arr.shape[0]
        out = np.zeros((batch_size, self.n_targets), dtype=np.float64)
        handle = self._ensure_open()
        rc = _LIB.cfastfht_plan_execute_batch(
            handle,
            coeffs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(self.n_sources),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(self.n_targets),
            ctypes.c_size_t(batch_size),
        )
        if rc != 0:
            raise RuntimeError(_last_error())
        return out


class _PlanCache:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, int, int], CNUFHTPlan] = {}

    def get(self, key: Tuple[str, int, int]) -> Optional[CNUFHTPlan]:
        return self._cache.get(key)

    def insert(self, key: Tuple[str, int, int], plan: CNUFHTPlan) -> None:
        self._cache[key] = plan


_PLAN_CACHE = _PlanCache()


def _plan_key(nu: float, rs: np.ndarray, ws: np.ndarray, tol: float) -> Tuple[str, int, int]:
    hasher = hashlib.sha1()
    hasher.update(np.ascontiguousarray(rs, dtype=np.float64).tobytes())
    hasher.update(np.ascontiguousarray(ws, dtype=np.float64).tobytes())
    hasher.update(np.float64(tol).tobytes())
    digest = hasher.hexdigest()
    return (digest, rs.size, ws.size)


def _get_plan(nu: float,
              rs: np.ndarray,
              ws: np.ndarray,
              tol: float) -> CNUFHTPlan:
    key = _plan_key(nu, rs, ws, tol)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = CNUFHTPlan(nu, rs, ws, tol)
        _PLAN_CACHE.insert(key, plan)
    return plan


def fht_batch(nu: float,
              rs: np.ndarray,
              ws: np.ndarray,
              coeff_re: np.ndarray,
              coeff_im: np.ndarray,
              tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    if coeff_re.shape != coeff_im.shape:
        raise ValueError("Coefficient batches must have matching shapes")
    plan = _get_plan(nu, rs, ws, tol)
    g_re = plan.execute_batch(coeff_re)
    g_im = plan.execute_batch(coeff_im)
    return g_re, g_im
