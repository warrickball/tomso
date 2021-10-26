"""Utility functions used by other modules."""
import numpy as np
import gzip
from .constants import sigma_SB
from .constants import GMsun, Rsun, Dnu_sun        # adiabatic
from .constants import Lsun, nu_max_sun, Teff_sun  # full

def integrate(y, x):
    """Integral of `y` over `x`, computed using the trapezoidal rule. 
    i.e. :math:`\\int _{x[0]} ^x y(x') dx'`."""
    dz = 0.5*(y[1:]+y[:-1])*np.diff(x)
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


class AdiabaticStellarModel(object):
    """Base stellar model class that defines properties that are computed
    the same way in all stellar model formats for which only adiabatic
    frequencies can be calculated."""
    def __str__(self):
        return '\n'.join([
            '%s' % type(self),
            'M    %9.3e g      %7.3f Msun' % (self.M, self.G*self.M/GMsun),
            'R    %9.3e cm    %8.3f Rsun' % (self.R, self.R/Rsun),
            'Dnu  %9.1f uHz    %7.3f Dnu_sun' % (self.Dnu, self.Dnu_factor)
        ])


    @property
    def Dnu_factor(self):
        return (self.G*self.M/GMsun/self.R**3*Rsun**3)**0.5

    @property
    def Dnu(self): return self.Dnu_factor*Dnu_sun

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


class FullStellarModel(AdiabaticStellarModel):
    """Base stellar model class that defines properties that are computed
    the same way in all stellar model formats for which both adiabatic
    and non-adiabatic frequencies can be calculated."""
    def __str__(self):
        return super(FullStellarModel, self).__str__() + '\n' + \
            '\n'.join([
                'L    %9.3e erg/s %8.3f Lsun' % (self.L, self.L/Lsun),
                'Teff   %7i K      %7.3f Teff_sun' % (self.Teff, self.Teff/Teff_sun),
                'nu_max %7i uHz    %7.3f nu_max_sun' % (self.nu_max, self.nu_max_factor)])

    @property
    def Teff(self): return get_Teff(self.L, self.R)

    @property
    def nu_max_factor(self):
        return self.G*self.M/GMsun/(self.R/Rsun)**2/(self.Teff/Teff_sun)**0.5

    @property
    def nu_max(self):
        return self.nu_max_factor*nu_max_sun

    @property
    def Gamma_2(self): return 1.0/(1.0-self.grad_a)

    @property
    def Gamma_3(self): return 1. + (1.-1./self.Gamma_2)*self.Gamma_1

    @property
    def grad_r(self): return 3*self.kappa*self.P*self.L_r/(64.*np.pi*sigma_SB*self.G*self.m*self.T**4)
