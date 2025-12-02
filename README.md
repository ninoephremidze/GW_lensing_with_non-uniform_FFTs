All user-facing classes are re-exported from `fift.__init__` for convenient
import.

---

## Main features

- **General 2D Fresnel integral (NUFFT)**  
  `FresnelNUFFT3Vec`: batched Fresnel integrals using FINUFFT’s type-3 NUFFT
  and 2D Gauss–Legendre quadrature.

- **Axisymmetric Fresnel integral (NUFHT)**  
  - `FresnelHankelAxisymmetric`: uses **precomputed 1D Gauss–Legendre nodes**
    and **FastHankelTransform.jl** for a zeroth-order Hankel transform.  
  - `FresnelHankelAxisymmetricTrapezoidal`:
    uses a **uniform radial grid + trapezoidal rule** instead of Gauss–Legendre.

- **Lens models**  
  Contains simple axisymmetric lens models: `SIS` and `PointLens`.

- **Precomputed Gauss–Legendre grids**  
  - Utilities for precomputing and storing 1D or 2D GL nodes/weights live in
    the notebook `precompute_GL.ipynb`.
  - Both spherical and symmetric cases load in the precomputed GL quadratures.

- **Comparison helpers**
  - `time_func(fn, ...)`: best-of-N timing helper.  
  - `plot_overlays_ws(...)`: log–log overlay plots for comparing `fift` results vs GLoW.
    - The general 2D FIFT case is compared to the most general GLoW function (`It_MultiContour_C()`).
    - The spherical hankel FIFT case is compared to the fast, axisymmetric GLoW function (`It_SingleIntegral_C()`).

---

## Installation

### Python dependencies

For the 2D NUFFT path (`FresnelNUFFT3Vec`):

- [`finufft`](https://finufft.readthedocs.io)
  
For the axisymmetric Hankel path you now have **two** choices:

1. **Native C backend (preferred when available)**
   - `finufft` Python wheel (ships `libfinufft.so`)
   - C17 compiler + CMake ≥ 3.16

2. **Julia backend (legacy fallback)**
   - `juliacall`
   - Julia packages `FastHankelTransform` + `ForwardDiff`

Install the shared Python requirements:

```bash
pip install numpy psutil threadpoolctl finufft matplotlib juliacall pytest
```

### Building the C NUFHT backend (optional)

```bash
cd cfastfht
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
      -DFINUFFT_LIB_DIR=$(python3 - <<'PY'
import importlib, pathlib
print(pathlib.Path(importlib.import_module("finufft").__file__).parent)
PY
)
cmake --build build
```

Ensure `libcfastfht.so` (the build output) stays in `cfastfht/build/` or add its
location to `CFASTFHT_LIB` before importing `fift`.  The hankel module will load
this backend automatically and fall back to Julia only when the shared library
cannot be found.
