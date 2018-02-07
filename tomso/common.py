"""Functions and constants common to multiple modules."""
import numpy as np

DEFAULT_G = 6.67428e-8
"""Default value of the gravitational constant that is shared
by all modules."""

def integrate(y, x):
    """Integral of `y` over `x`. 
    i.e. :math:`\int _{x[0]} ^x y(x') dx'`"""
    dz = (yy[1:]+yy[:-1])/2.*np.diff(xx)
    return np.hstack((0., np.cumsum(dz)))
