"""Functions and constants common to multiple modules."""
import numpy as np
import gzip

DEFAULT_G = 6.67428e-8
"""Default value of the gravitational constant that is shared
by all modules."""

def integrate(y, x):
    """Integral of `y` over `x`, computed using the trapezoidal rule. 
    i.e. :math:`\int _{x[0]} ^x y(x') dx'`."""
    dz = (y[1:]+y[:-1])/2.*np.diff(x)
    return np.hstack((0., np.cumsum(dz)))

def complement(y, x):
    """Complement of integral of `y` over `x`, computed using the
    trapezoidal rule.  i.e. :math:`\int _x^{x[-1]}y(x') dx'`."""
    z = integrate(y, x)
    return z[-1] - z


def tomso_open(filename, *args, **kwargs):
    """Wrapper function to open files ending with `.gz` with built-in
    `gzip` module, otherwise use normal open.  Takes the same
    arguments as `open` and `gzip.open` and returns a file object.
    """
    if filename.lower().endswith('.gz'):
        return gzip.open(filename, *args, **kwargs)
    else:
        return open(filename, *args, **kwargs)
