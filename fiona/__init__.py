"""
FIONA: Fresnel Integral Optimization via Non-uniform trAnsforms.

Author: Nino Ephremidze
"""
import os
from .lenses import Lens, AxisymmetricLens, SIS, PointLens
from .axisym import FresnelNUFHT
from .general import FresnelNUFFT3
from .utils import CPUTracker

__all__ = [
    # Lenses
    "Lens", "AxisymmetricLens", "SIS", "PointLens",
    # Axisymmetric solvers
    "FresnelNUFHT",
    # 2-D solvers
    "FresnelNUFFT3",
    # controls
    "set_num_threads",
    # CPU Tracker
    "CPUTracker",
]

def set_num_threads(n: int):
    """
    Hint FINUFFT/NumPy/OMP to use up to `n` threads.
    Call this once early in your script.
    """
    n = int(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    try:
        import finufft
        finufft.set_num_threads(n)
    except Exception:
        pass

__version__ = "0.1.1"
