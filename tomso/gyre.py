# -*- coding: utf-8 -*-

"""
Functions for manipulating `GYRE`_ input and output files.

.. _GYRE: https://gyre.readthedocs.io/
"""

import numpy as np
import warnings
from .constants import G_DEFAULT
from .utils import tomso_open, load_mesa_gyre
from .utils import integrate, regularize
from .utils import FullStellarModel


def load_summary(filename, return_object=True):
    """Reads a GYRE summary file and returns the global data and mode data
    in two structured arrays.  Uses builtin `gzip` module to read
    files ending with `.gz`.

    If `return_object` is `True`, instead returns an `GYRELog` object.
    This is the default behaviour as of v0.0.12.  The old
    behaviour will be dropped completely from v0.1.0.

    Parameters
    ----------
    filename: str
        Filename of the GYRE summary file to load.

    Returns
    -------
    header: structured array
        Global data for the frequency calculation. e.g. initial parameters.
        The keys for the array are the GYRE variable names as in
        the ``&output`` namelist in the GYRE input file.

    data: structured array
        Mode data for the frequency calculation. e.g. mode frequencies.
        The keys for the array are the GYRE variable names as in
        the ``&output`` namelist in the GYRE input file.

    """

    header, data = load_mesa_gyre(filename, 'gyre')
    if return_object:
        return GYRELog(header, data)
    else:
        warnings.warn("From tomso 0.1.0+, `gyre.load_summary` will only "
                      "return a `GYRELog` object: use `return_object=True` "
                      "to mimic future behaviour",
                      FutureWarning)
        return header, data


def load_mode(filename, return_object=True):
    """Reads a GYRE mode file and returns the global data and mode profile
    data in two structured arrays.  Uses builtin `gzip` module to read
    files ending with `.gz`.

    If `return_object` is `True`, instead returns an `GYRELog` object.
    This is the default behaviour as of v0.0.12.  The old
    behaviour will be dropped completely from v0.1.0.

    Parameters
    ----------
    filename: str
        Filename of the GYRE mode file to load.

    Returns
    -------
    header: structured array
        Global data for the frequency calculation. e.g. initial parameters.
        The keys for the array are the GYRE variable names as in
        the ``&output`` namelist in the GYRE input file.

    data: structured array
        Mode data for the frequency calculation. e.g. mode frequencies.
        The keys for the array are the GYRE variable names as in
        the ``&output`` namelist in the GYRE input file.

    """

    header, data = load_mesa_gyre(filename, 'gyre')
    if return_object:
        return GYRELog(header, data)
    else:
        warnings.warn("From tomso 0.1.0+, `gyre.load_summary` will only "
                      "return a `GYRELog` object: use `return_object=True` "
                      "to mimic future behaviour",
                      FutureWarning)
        return header, data


def load_gyre(filename, return_object=True):
    """Reads a GYRE stellar model file and returns the global data and
    point-wise data in a pair of NumPy record arrays.  Uses builtin
    `gzip` module to read files ending with `.gz`.

    If `return_object` is `True`, instead returns a
    :py:class:`GYREStellarModel` object.  This is the default
    behaviour as of v0.0.12.  The old behaviour will be dropped
    completely from v0.1.0.

    Parameters
    ----------
    filename: str
        Filename of the GYRE file.

    Returns
    -------
    header: structured array
        Global data for the stellar model. e.g. total mass, luminosity.

    data: structured array
        Profile data for the stellar model. e.g. radius, pressure.

    """
    def replace(s):
        # handles annoying Fortran formatting
        t = s[:]
        t = t.replace(b'D', b'E')
        t = t.replace(b'+', b'E+')
        t = t.replace(b'-', b'E-')
        t = t.replace(b'EE', b'E')
        t = t.replace(b' E-', b' -')
        return t

    with tomso_open(filename, 'rb') as f:
        lines = [replace(line) for line in f.readlines()]

    header_length = len(lines[0].split())
    if header_length == 4:
        version = 1
    elif header_length == 5:
        version = int(lines[0].split()[-1])
    else:
        raise ValueError("header should have 4 or 5 components but "
                         "it appears to have %i" % header_length)

    header = np.loadtxt(lines[:1], dtype=gyre_header_dtypes[version])
    data = np.loadtxt(lines[1:], dtype=gyre_data_dtypes[version])

    if return_object:
        return GYREStellarModel(header, data)
    else:
        warnings.warn("From tomso 0.1.0+, `gyre.load_gyre` will only return "
                      "a `GYREStellarModel` object: use `return_object=True` "
                      "to mimic future behaviour",
                      FutureWarning)
        return header, data


def save_gyre(filename, header, data):
    """Given the global data and point-wise data for a stellar model (as
    returned by :py:meth:`load_gyre`), saves the data to a target file
    in the GYRE format.

    Parameters
    ----------
    filename: str
        Filename of the GYRE file.

    header: structured array
        Global data for the stellar model. e.g. total mass, luminosity.

    data: structured array
        Profile data for the stellar model. e.g. radius, pressure.

    """
    with open(filename, 'wt') as f:
        header_length = len(list(header[()]))
        # if header_length == 4:
        #     fmt = ''.join(['%6i','%26.16E'*3,'\n'])
        # elif header_length == 5:
        #     fmt = ''.join(['%6i','%26.16E'*3,'%6i\n'])
        # else:
        #     raise ValueError("header should have 4 or 5 components but "
        #                      "it appears to have %i" % header_length)
        if not 'version' in header.dtype.names:
            fmt = ''.join(['%6i','%26.16E'*3,'\n'])
        else:
            fmt = ''.join(['%6i','%26.16E'*3,'%6i\n'])

        f.writelines([fmt % tuple(header[()])])

        N = len(data[0])-1
        fmt = ''.join(['%6i',' %26.16E'*N,'\n'])
        for row in data:
            f.writelines([fmt % tuple(row)])


class GYRELog(object):
    """A dict-like class that contains the data for a GYRE summary or mode
    file.  Variables in the header or the body can be accessed by the
    appropriate key, as interpreted by ``numpy.genfromtxt``, so the
    fields ``Re(x)`` become ``Rex``. e.g. ``GYRELog['Refreq']``
    returns the `Re(freq)` column.

    This object will normally be instantiated using
    :py:meth:`gyre.load_summary` or :py:meth:`gyre.load_mode`.

    Parameters
    ----------
    header: structured array
        Header data for the GYRE summary or mode file. i.e. data for
        which there is only one value in the file.
    data: structured array
        Columned data for the summary or mode file. i.e. data for
        which there are multiple values (one per timestep or mesh
        point).

    """
    def __init__(self, header, data):
        self.header = header
        self.data = data

    def __len__(self):
        return len(self.data)

    def __str__(self):
        s = ['%s\n' % type(self)]
        if self.header is None:
            s.append('Header:\n    empty\n')
        else:
            s.append('Header:\n')
            for name in self.header.dtype.names:
                s.append('%26s = %s\n' % (name, self.header[name]))

        s.append('Column names:\n')
        N = max([len(name) for name in self.data.dtype.names])+1
        cols = 80//N
        for i, name in enumerate(self.data.dtype.names):
            s.append(name.rjust(N))
            if (i+1)%cols==0:
                s.append('\n')

        return ''.join(s)

    def __repr__(self):
        with np.printoptions(threshold=10):
            return('GYRELog(\nheader=\n%s,\ndata=\n%s)' % (self.header, self.data))

    def __getitem__(self, key):
        if isinstance(key, str):
            for source in [self.data, self.header]:
                names = source.dtype.names
                if key in names:
                    return source[key]
            else:
                raise KeyError(key)
        else:
            # assume we're trying to slice the data array
            return GYRELog(self.header, self.data[key])


class GYREStellarModel(FullStellarModel):
    """A class that contains and allows one to manipulate the data stored
    a plain-text GYRE Stellar Model.
    This will usually be provided from a file by using
    :py:meth:`load_gyre` but an object can be constructed from any
    similarly structured arrays.

    The main attributes are the **header** and **data** record arrays,
    which store the data that's written in the text file.  The data in
    these arrays can be accessed via the attributes with more
    physically-meaningful names (e.g. the speed of sound is
    ``GYREStellarModel.cs``).

    Some of these values can also be set via the attributes if doing
    so is unambiguous. For example, the fractional radius **x** is not
    a member of the **data** array but setting **x** will assign
    the actual radius **r** to the corresponding values.  Values that
    are settable are indicated in the list of parameters.

    Parameters
    ----------
    header: structured array
        Global data for the stellar model. e.g. total mass, luminosity.

    data: structured array
        Profile data for the stellar model. e.g. radius, pressure.

    G: float, optional
        Value for the gravitational constant, in cgs units.  If not
        given (which is the default behaviour), we use the module-wise
        default value.

    Attributes
    ----------
    version: int
        file version number
    M: float, settable
        total mass
    R: float, settable
        photospheric radius
    L: float, settable
        total luminosity
    Teff: float
        effective temperature, derived from luminosity and radius
    k: NumPy array
        mesh point number
    r: NumPy array, settable
        radius co-ordinate
    T: NumPy array, settable
        temperature
    P: NumPy array, settable
        pressure
    rho: NumPy array, settable
        density
    L_r: NumPy array, settable
        luminosity at radius **r**
    kappa: NumPy array, settable
        Rosseland mean opacity
    eps: NumPy array, settable
        specific energy generation rate
    Gamma_1: NumPy array
        first adiabatic index
    AA: NumPy array
        Ledoux discriminant
    x: NumPy array, settable
        fractional radius co-ordinate
    q: NumPy array, settable
        fractional mass co-ordinate
    m: NumPy array, settable
        mass co-ordinate
    w: NumPy array
        former fractional mass depth (w=m/(M-m))
    g: NumPy array
        local gravitational acceleration
    Hp: NumPy array
        pressure scale height
    Hrho: NumPy array
        density scale height
    N2: NumPy array
        squared Brunt–Väisälä (angular) frequency
    cs2: NumPy array
        squared adiabatic sound speed
    cs: NumPy array
        adiabatic sound speed
    U: NumPy array
        homology invariant *dlnm/dlnr*
    V: NumPy array
        homology invariant *dlnP/dlnr*
    Vg: NumPy array
        homology invariant *V/Gamma_1*
    tau: NumPy array
        acoustic depth
    """
    def __init__(self, header, data, G=G_DEFAULT):
        self.header = header
        self.data = data
        self.G = G

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        with np.printoptions(threshold=10):
            return('GYREStellarModel(\nheader=\n%s,\ndata=\n%s\n)' % (self.header, self.data))

    def to_file(self, filename):
        """Save the model to a file.

        Parameters
        ----------
        filename: str
            Filename to which the data is written.
        """
        self.header['n'] = self.n
        save_gyre(filename, self.header, self.data)

    def to_fgong(self, reverse=True, ivers=1300):
        """Convert the model to an ``FGONG`` object.

        Parameters
        ----------
        reverse: bool, optional
            If ``True`` (the default), store the FGONG data ordered
            from the surface to the centre.  Otherwise, store the
            FGONG data ordered from the centre to the surface.
        ivers: int, optional
            The integer indicating the version number of the file.
            (default=1300)

        """
        from .fgong import FGONG

        glob = np.zeros(15)
        glob[0] = self.M
        glob[1] = self.R
        glob[2] = self.L
        glob[14] = self.G

        var = np.zeros((len(self.data), 40))
        var[:,0] = self.r
        var[:,1] = self.lnq
        var[:,2] = self.T
        var[:,3] = self.P
        var[:,4] = self.rho
        var[:,6] = self.L_r
        var[:,7] = self.kappa
        var[:,9] = self.Gamma_1
        var[:,10] = self.grad_a
        var[:,14] = self.AA

        if reverse:
            return FGONG(glob, var[::-1], ivers=ivers, G=self.G)
        else:
            return FGONG(glob, var, ivers=ivers, G=self.G)

    def to_amdl(self):
        """Convert the model to an ``ADIPLSStellarModel`` object."""
        from .adipls import ADIPLSStellarModel

        ioff = (0 if self.r[0] < 1e6 else 1) # mimic ADIPLS's FGONG to AMDL script
        A = np.zeros((len(self.data) + ioff, 6))

        # we can safely ignore division by 0 here
        with np.errstate(divide='ignore', invalid='ignore'):
            A[ioff:,0] = self.x
            A[ioff:,1] = self.q/self.x**3
            A[ioff:,2] = self.Vg
            A[ioff:,3] = self.Gamma_1
            A[ioff:,4] = self.AA
            A[ioff:,5] = self.U

        A[0,0] = 0.
        A[0,1] = 4.*np.pi/3.*self.rho[0]*self.R**3/self.M
        A[0,2] = 0.
        A[0,3] = self.Gamma_1[0]
        A[0,4] = 0.
        A[0,5] = 3.

        D = np.zeros(8)
        D[0] = self.M
        D[1] = self.R
        D[2] = self.P[0]
        D[3] = self.rho[0]
        D[4] = 4.*np.pi/3.*self.G*(self.rho[0]*self.R)**2/(self.P[0]*self.Gamma_1[0])
        D[5] = D[4]
        D[6] = -1.0
        D[7] = 0.0

        return ADIPLSStellarModel(D, A, G=self.G)

    @property
    def version(self):
        if 'version' in self.header.dtype.names:
            return self.header['version']
        else:
            return 1

    @property
    def n(self):
        return len(self.data)

    # Various properties for easier access to the data in `header` and
    # `data`.

    @property
    def M(self): return self.header['M']

    @M.setter
    def M(self, val): self.header['M'] = val

    @property
    def R(self): return self.header['R']

    @R.setter
    def R(self, val): self.header['R'] = val

    @property
    def L(self): return self.header['L']

    @L.setter
    def L(self, val): self.header['L'] = val

    @property
    def k(self): return self.data['k']

    @property
    def r(self): return self.data['r']

    @r.setter
    def r(self, val): self.data['r'] = val

    @property
    def L_r(self): return self.data['L_r']

    @L_r.setter
    def L_r(self, val): self.data['L_r'] = val

    @property
    def P(self): return self.data['P']

    @P.setter
    def P(self, val): self.data['P'] = val

    @property
    def T(self): return self.data['T']

    @T.setter
    def T(self, val): self.data['T'] = val

    @property
    def rho(self): return self.data['rho']

    @rho.setter
    def rho(self, val): self.data['rho'] = val

    @property
    def nabla(self): return self.data['nabla']

    @property
    def N2(self): return self.data['N2']

    @N2.setter
    def N2(self, val): self.data['N2'] = val

    @property
    def kappa(self): return self.data['kappa']

    @kappa.setter
    def kappa(self, val): self.data['kappa'] = val

    @property
    def grad_a(self): return self.data['nabla_ad']

    # Some properties have definitions that depend on the GYRE file
    # version.
    @property
    def w(self):
        if self.version in [1, 19]:
            return self.data['w']
        else:
            return self.data['m']/(self.header['M']-self.data['m'])

    @property
    def m(self):
        if self.version in [1, 19]:
            return self.data['w']*self.header['M']/(self.data['w']+1)
        else:
            return self.data['m']

    @m.setter
    def m(self, val):
        if self.version in [1, 19]:
            self.data['w'] = val/(self.M-w)
        else:
            self.data['m'] = val

    @property
    def Gamma_1(self):
        if self.version == 1:
            return self.data['c_P']/self.data['c_V']
        else:
            return self.data['Gamma_1']

    @Gamma_1.setter
    def Gamma_1(self, val):
        if self.version == 1:
            raise ValueError
        else:
            self.data['Gamma_1'] = val

    @property
    def eps(self):
        if self.version in [1, 19, 100]:
            return self.data['eps_tot']
        else:
            return self.data['eps']

    @property
    def Omega(self):
        if self.version == 1:
            return np.zeros_like(self.data['k'])
        else:
            return self.data['Omega']

    # Some convenient quantities derived from data in `header` and
    # `data` arrays.
    @property
    def x(self): return self.r/self.R

    @x.setter
    def x(self, val): self.r = val*self.R

    @property
    def q(self): return self.m/self.M

    @q.setter
    def q(self, val): self.m = val*self.M

    @property
    @regularize(y0=-np.inf, x0=1e-308)
    def lnq(self): return np.log(self.q)

    @lnq.setter
    def lnq(self, val): self.q = np.exp(val)

    @property
    @regularize()
    def AA(self): return self.N2*self.r/self.g

    @property
    @regularize(y0=3.0)
    def U(self): return 4.*np.pi*self.rho*self.r**3/self.m

    @property
    @regularize()
    def V(self): return self.G*self.m*self.rho/self.P/self.r

    @property
    def Vg(self): return self.V/self.Gamma_1

    @property
    def tau(self):
        tau = integrate(1./self.cs[::-1], self.r[::-1])[::-1]
        return np.max(tau)-tau

gyre_header_dtypes = {1: [('n','int'), ('M','float'), ('R','float'),
                          ('L','float')],
                      19: [('n','int'), ('M','float'), ('R','float'),
                           ('L','float'), ('version','int')],
                      100: [('n','int'), ('M','float'), ('R','float'),
                            ('L','float'), ('version','int')],
                      101: [('n','int'), ('M','float'), ('R','float'),
                            ('L','float'), ('version','int')]}

gyre_data_dtypes = {1: [('k','int'), ('r','float'), ('w','float'),
                        ('L_r','float'), ('P','float'), ('T','float'),
                        ('rho','float'), ('nabla','float'),
                        ('N2','float'), ('c_V','float'), ('c_P','float'),
                        ('chi_T','float'), ('chi_rho','float'),
                        ('kappa','float'), ('kappa_T','float'),
                        ('kappa_rho','float'), ('eps_tot','float'),
                        ('eps_eps_T','float'), ('eps_eps_rho','float')],
                    19: [('k','int'), ('r','float'), ('w','float'),
                         ('L_r','float'), ('P','float'), ('T','float'),
                         ('rho','float'), ('nabla','float'),
                         ('N2','float'), ('Gamma_1','float'),
                         ('nabla_ad','float'), ('delta','float'),
                         ('kappa','float'), ('kappa_T','float'),
                         ('kappa_rho','float'), ('eps_tot','float'),
                         ('eps_eps_T','float'), ('eps_eps_rho','float'),
                         ('Omega','float')],
                    100: [('k','int'), ('r','float'), ('m','float'),
                          ('L_r','float'), ('P','float'), ('T','float'),
                          ('rho','float'), ('nabla','float'),
                          ('N2','float'), ('Gamma_1','float'),
                          ('nabla_ad','float'), ('delta','float'),
                          ('kappa','float'), ('kappa_kappa_T','float'),
                          ('kappa_kappa_rho','float'), ('eps_tot','float'),
                          ('eps_eps_T','float'), ('eps_eps_rho','float'),
                          ('Omega','float')],
                    101: [('k','int'), ('r','float'), ('m','float'),
                          ('L_r','float'), ('P','float'), ('T','float'),
                          ('rho','float'), ('nabla','float'),
                          ('N2','float'), ('Gamma_1','float'),
                          ('nabla_ad','float'), ('delta','float'),
                          ('kappa','float'), ('kappa_kappa_T','float'),
                          ('kappa_kappa_rho','float'), ('eps','float'),
                          ('eps_eps_T','float'), ('eps_eps_rho','float'),
                          ('Omega','float')]}
