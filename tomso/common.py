"""Functions and constants common to multiple modules."""
import numpy as np

DEFAULT_G = 6.67428e-8

def integrate(yy, xx):
    """Integral of yy over xx. i.e. \int _x[0] ^x y(x) dx"""
    dz = (yy[1:]+yy[:-1])/2.*np.diff(xx)
    return np.hstack((0., np.cumsum(dz)))
