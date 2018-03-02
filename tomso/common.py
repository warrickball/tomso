"""Functions and constants common to multiple modules."""
import numpy as np

DEFAULT_G = 6.67428e-8
"""Default value of the gravitational constant that is shared
by all modules."""

def integrate(y, x):
    """Integral of `y` over `x`. 
    i.e. :math:`\int _{x[0]} ^x y(x') dx'`"""
    dz = (y[1:]+y[:-1])/2.*np.diff(x)
    return np.hstack((0., np.cumsum(dz)))
