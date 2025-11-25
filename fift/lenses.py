import numpy as np
from abc import ABC, abstractmethod

class Lens(ABC):
    """Abstract base for a projected lensing potential ψ(x1,x2)."""

    @abstractmethod
    def psi_xy(self, x1, x2):
        """Return ψ(x) on R^2 (broadcast over arrays)."""
        ...

class AxisymmetricLens(Lens):
    """Axisymmetric lenses: ψ(x) = ψ(r), with r = sqrt(x1^2 + x2^2)."""

    @abstractmethod
    def psi_r(self, r):
        ...

    def psi_xy(self, x1, x2):
        r = np.hypot(x1, x2)
        return self.psi_r(r)

class SIS(AxisymmetricLens):
    r"""Singular Isothermal Sphere: ψ(r) = ψ0 * r."""
    def __init__(self, psi0=1.0):
        self.psi0 = float(psi0)

    def psi_r(self, r):
        return self.psi0 * r

class PointLens(AxisymmetricLens):
    r"""Point lens with Plummer softening: ψ(r) = 0.5 ψ0 log(r^2 + x_c^2)."""
    def __init__(self, psi0=1.0, xc=0.0):
        self.psi0 = float(psi0)
        self.xc   = float(xc)

    def psi_r(self, r):
        return 0.5 * self.psi0 * np.log(r*r + self.xc*self.xc)
