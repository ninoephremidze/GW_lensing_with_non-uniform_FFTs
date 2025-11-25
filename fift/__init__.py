import os

from .lenses import Lens, AxisymmetricLens, SIS, PointLens
from .spatial import FresnelNUFFT3Vec
from .hankel import FresnelHankelAxisymmetric, FresnelHankelAxisymmetricTrapezoidal
from .utils import CPUTracker

__all__ = [
    # lenses
    "Lens", "AxisymmetricLens", "SIS", "PointLens",
    # general 2-D
    "FresnelNUFFT3Vec",
    # axisymmetric (Hankel)
    "FresnelHankelAxisymmetric",
    "FresnelHankelAxisymmetricTrapezoidal"
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
