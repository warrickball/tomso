"""Functions and constants common to multiple modules."""
import numpy as np
import gzip
from .constants import sigma_SB

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
    `gzip` module or paths starting with `http` using
    `urllib.request.urlopen`, otherwise use normal open.  `.gz` and
    normal modes take the same arguments as `open` and `gzip.open` and
    return a file object."""
    if filename.startswith('http'):
        from urllib.request import urlopen
        return urlopen(filename)
    elif filename.lower().endswith('.gz'):
        return gzip.open(filename, *args, **kwargs)
    else:
        return open(filename, *args, **kwargs)


def load_mesa_gyre(filename, mesa_or_gyre):
    """Most MESA and GYRE output files both adhere to a similar columned
    ASCII format, so it makes more sense to have one implementation
    for reading them, rather than re-implementing it in each
    submodule.

    """
    with tomso_open(filename, 'rb') as f:
        lines = f.readlines()

    if mesa_or_gyre == 'mesa':
        header = np.genfromtxt(lines[1:3], names=True, dtype=None, encoding='utf-8')
    elif mesa_or_gyre == 'gyre':
        # the GYRE header might be empty
        try:
            header = np.genfromtxt(lines[2:4], names=True, dtype=None, encoding='utf-8')
        except IndexError:
            header = None
    else:
        raise ValueError("mesa_or_gyre must be either 'mesa' or 'gyre', not %s"
                         % mesa_or_gyre)

    data = np.genfromtxt(lines[5:], names=True, dtype=None, encoding='utf-8')

    return header, data


def get_Teff(L, R):
    """Determine the effective temperature `Teff` for a given luminosity
    `L` and radius `R`, both in cgs units."""

    return (L/(4.*np.pi*R**2*sigma_SB))**0.25
