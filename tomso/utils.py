"""Utility functions used by other modules."""
import numpy as np
import gzip
from .constants import sigma_SB

DEFAULT_G = 6.67428e-8
"""Default value of the gravitational constant that is shared
by all modules."""

def integrate(y, x):
    """Integral of `y` over `x`, computed using the trapezoidal rule. 
    i.e. :math:`\\int _{x[0]} ^x y(x') dx'`."""
    dz = (y[1:]+y[:-1])/2.*np.diff(x)
    return np.hstack((0., np.cumsum(dz)))


def complement(y, x):
    """Complement of integral of `y` over `x`, computed using the
    trapezoidal rule.  i.e. :math:`\\int _x^{x[-1]}y(x') dx'`."""
    z = integrate(y, x)
    return z[-1] - z


def regularize(y0=0.0, x0=1e-12):
    def regularizer(f):
        def regularized_f(s):
            with np.errstate(divide='ignore', invalid='ignore'):
                y = f(s)

            y[s.x < x0] = y0
            return y

        return regularized_f

    return regularizer


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


class BaseStellarModel(object):
    """Base stellar model class that defines properties that are
    computed the same way in all stellar model formats."""
    @property
    @regularize()
    def g(self): return self.G*self.m/self.r**2

    @property
    def dP_dr(self): return -self.rho*self.g

    @property
    @regularize(y0=np.inf)
    def Hp(self): return -self.P/self.dP_dr

    @property
    @regularize()
    def drho_dr(self): return -self.rho*(1/self.Gamma_1/self.Hp + self.AA/self.r)

    @property
    @regularize(y0=np.inf)
    def Hrho(self): return -self.rho/self.drho_dr

    @property
    def n_eff(self): return 1/(self.Hrho/self.Hp-1)

    @property
    def cs2(self): return self.Gamma_1*self.P/self.rho

    @property
    def cs(self): return self.cs2**0.5

    @property
    def N(self):
        y = np.full(len(self.x), 0.)
        y[self.N2>0] = self.N2[self.N2>0]**0.5
        return y

    @property
    @regularize(y0=np.inf)
    def S2_1(self): return 2.*self.cs2/self.r**2

    @property
    def S_1(self): return self.S2_1**0.5
