"""
FIFT: Fresnel Integral via Fourier Transforms.

 - 2D nonuniform FFT (NUFFT) for general lenses.
 - Fast Hankel transform (FHT/FFTLog) for circular symmetry, with
   a robust NUFFT fallback when FHT is not available.

Author: Nino Ephremidze
"""
from .lenses import Lens, AxisymmetricLens, SIS, PointLens
from .axisym import FresnelHankel
from .axisym_quad import FresnelHankelQuad
from .spatial import (
    FresnelNUFFT2D, FresnelNUFFT3, FresnelDirect3
)

__all__ = [
    # Lenses
    "Lens", "AxisymmetricLens", "SIS", "PointLens",
    # Axisymmetric solvers
    "FresnelHankel", "FresnelHankelQuad",
    # 2-D solvers
    "FresnelNUFFT2D", "FresnelNUFFT3", "FresnelDirect3",
]

__version__ = "0.1.0"
