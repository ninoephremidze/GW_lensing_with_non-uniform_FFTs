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

- [`finufft`](https://finufft.readthedocs.io) (Python bindings)

For the axisymmetric Hankel path:

- `juliacall` (Python ↔ Julia bridge)
- Julia packages:
  - `FastHankelTransform`
  - `ForwardDiff` (used inside FastHankelTransform)

Install Python requirements:

```bash
pip install numpy psutil threadpoolctl finufft matplotlib juliacall
