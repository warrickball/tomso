# -*- coding: utf-8 -*-

"""Functions for manipulating FGONG files.  These are provided through
the **FGONG** object and a module function to read an **FGONG** object
from a file.

"""

import numpy as np
import warnings
from .constants import G_DEFAULT
from .utils import integrate, tomso_open, regularize
from .utils import FullStellarModel


def load_fgong(filename, fmt='ivers', G=None):
    """Given an FGONG file, returns a :py:class:`FGONG` that contains
    NumPy arrays ``glob`` and ``var`` that
    correspond to the scalar and point-wise variables, as specified
    in the `FGONG format`_.

    .. _FGONG format: https://www.astro.up.pt/corot/ntools/docs/CoRoT_ESTA_Files.pdf

    Also returns the first four lines of the file as a ``comment``, if
    desired.  The data can be accessed via properties like ``x`` or ``P``
    for fractional radius and pressure.

    The version number ``ivers`` is used to infer the format of floats
    if ``fmt='ivers'``.

    Parameters
    ----------
    filename: str
        Name of the FGONG file to read.
    fmt: str, optional
        Format string for floats in `glob` and `var`.  If ``'ivers'``,
        uses ``%16.9E`` if the file's ``ivers < 1000`` or ``%26.18E3` if
        ``ivers >= 1000``.  If ``'auto'``, tries to guess the size of each
        float. (default: ``'ivers'``)

    Returns
    -------
    f: :py:class:`FGONG`
        The scalar (or global) variables for the stellar model

    """
    with tomso_open(filename, 'rb') as f:
        comment = [f.readline().decode('utf-8').strip() for i in range(4)]
        nn, iconst, ivar, ivers = [int(i) for i in f.readline().decode('utf-8').split()]
        # lines = f.readlines()
        lines = [line.decode('utf-8').lower().replace('d', 'e')
                 for line in f.readlines()]

    tmp = []

    if fmt == 'ivers':
        if ivers < 1000:
            N = 16
        else:
            N = 27
    # try to guess the length of each float in the data
    elif fmt == 'auto':
        N = len(lines[0])//5
    else:
        N = len(fmt % -1.111)

    for line in lines:
        for i in range(len(line)//N):
            s = line[i*N:i*N+N]
            # print(s)
            if s[-9:] == '-Infinity':
                s = '-Inf'
            elif s[-9:] == ' Infinity':
                s = 'Inf'
            elif s.lower().endswith('nan'):
                s = 'nan'
            elif 'd' in s.lower():
                s = s.lower().replace('d','e')

            tmp.append(float(s))

    glob = np.array(tmp[:iconst])
    var = np.array(tmp[iconst:]).reshape((-1, ivar))

    return FGONG(glob, var, ivers=ivers, G=G,
                 description=comment)


class FGONG(FullStellarModel):
    """A class that contains and allows one to manipulate the data in a
    stellar model stored in the `FGONG format`_.

    .. _FGONG format: https://www.astro.up.pt/corot/ntools/docs/CoRoT_ESTA_Files.pdf

    The main attributes are the **glob** and **var** arrays, which
    follow the definitions in the FGONG standard.  The data in these
    arrays can be accessed via the attributes with more
    physically-meaningful names (e.g. the radius is ``FGONG.r``).

    Some of these values can also be set via the attributes if doing
    so is unambiguous. For example, the fractional radius **x** is not a
    member of the **var** array but setting **x** will assign the actual
    radius **r**, which is the first column of **var**.  Values that are
    settable are indicated in the list of parameters.

    Parameters
    ----------
    glob: NumPy array
        The global variables for the stellar model.
    var: NumPy array
        The point-wise variables for the stellar model. i.e. things
        that vary through the star like temperature, density, etc.
    ivers: int, optional
        The integer indicating the version number of the file.
        (default=0)
    G: float, optional
        Value for the gravitational constant.  If not given (which is
        the default behaviour), we use ``glob[14]`` if it exists and
        is close to the module-wide default value.  Otherwise, we use
        the module-wide default value.
    description: list of 4 strs, optional
        The first four lines of the FGONG file, which usually contain
        notes about the stellar model.

    Attributes
    ----------
    iconst: int
        number of global data entries (i.e. length of **glob**)
    nn: int
        number of points in stellar model (i.e. number of rows in **var**)
    ivar: int
        number of variables recorded at each point in stellar model
        (i.e. number of columns in **var**)
    M: float, settable
        total mass
    R: float, settable
        photospheric radius
    L: float, settable
        total luminosity
    Teff: float
        effective temperature, derived from luminosity and radius
    r: NumPy array, settable
        radius co-ordinate
    lnq: NumPy array, settable
        natural logarithm of the fractional mass co-ordinate
    T: NumPy array, settable
        temperature
    P: NumPy array, settable
        pressure
    rho: NumPy array, settable
        density
    X: NumPy array, settable
        fractional hydrogen abundance (by mass)
    L_r: NumPy array, settable
        luminosity at radius **r**
    kappa: NumPy array, settable
        Rosseland mean opacity
    epsilon: NumPy array, settable
        specific energy generation rate
    Gamma_1: NumPy array, settable
        first adiabatic index, aliased by **G1**
    G1: NumPy array, settable
        first adiabatic index, alias of **Gamma_1**
    cp: NumPy array, settable
        specific heat capacity
    AA: NumPy array, settable
        Ledoux discriminant
    Z: NumPy array, settable
        metal abundance
    x: NumPy array, settable
        fractional radius co-ordinate
    q: NumPy array, settable
        fractional mass co-ordinate
    m: NumPy array, settable
        mass co-ordinate
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

    def __init__(self, glob, var, ivers=300, G=None,
                 description=['', '', '', '']):
        self.ivers = ivers
        self.glob = glob
        self.var = var
        self.description = description

        # if G is None, use glob[14] if it exists and looks like a
        # reasonable value of G
        if G is None:
            if len(glob) >= 14 and np.isclose(glob[14], G_DEFAULT,
                                              rtol=1e-3, atol=0.01e-8):
                self.G = glob[14]
            else:
                self.G = G_DEFAULT
        else:
            self.G = G

    def __len__(self):
        return len(self.var)

    def __repr__(self):
        with np.printoptions(threshold=10):
            return('FGONG(\nglob=\n%s,\nvar=\n%s,\ndescription=\n%s)' % (self.glob, self.var, '\n'.join(self.description)))

    def to_file(self, filename, float_formatter='ivers'):
        """Save the model to an FGONG file.

        Parameters
        ----------
        filename: str
            Filename to which the data is written.
        float_formatter: str or function
            Determines how floating point numbers are formatted.  If
            ``'ivers'`` (the default), use the standard formats
            ``%16.9E`` if ``ivers < 1000`` or ``%26.18E3`` if ``ivers
            >= 1000``.  If a Python format specifier
            (e.g. ``'%16.9E'``), pass floats into that like
            ``float_formatter % float``.  Otherwise, must be a
            function that takes a float as an argument and returns a
            string.  In most circumstances you'll want to control the
            output by changing the value of ``'ivers'``.
        """
        nn, ivar = self.var.shape
        iconst = len(self.glob)

        if float_formatter == 'ivers':
            if self.ivers < 1000:
                def ff(x):
                    if not np.isfinite(x):
                        return '%16s' % x

                    s = np.format_float_scientific(x, precision=9, unique=False, exp_digits=2, sign=True)
                    if s[0] == '+':
                        s = ' ' + s[1:]

                    return s
            else:
                def ff(x):
                    if not np.isfinite(x):
                        return '%27s' % x

                    s = np.format_float_scientific(x, precision=18, unique=False, exp_digits=3, sign=True)
                    if s[0] == '+':
                        s = ' ' + s[1:]

                    return ' ' + s
        else:
            try:
                float_formatter % 1.111
                ff = lambda x: float_formatter % x
            except TypeError:
                ff = float_formatter

        with open(filename, 'wt') as f:
            f.write('\n'.join(self.description) + '\n')

            line = '%10i'*4 % (nn, iconst, ivar, self.ivers) + '\n'
            f.write(line)

            for i, val in enumerate(self.glob):
                f.write(ff(val))
                if i % 5 == 4:
                    f.write('\n')

            if i % 5 != 4:
                f.write('\n')

            for row in self.var:
                for i, val in enumerate(row):
                    f.write(ff(val))
                    if i % 5 == 4:
                        f.write('\n')

            if i % 5 != 4:
                f.write('\n')


    def to_amdl(self):
        """Convert the model to an :py:class:`ADIPLSStellarModel` object.

        The output should be identical (to within a few times machine
        error) to the output of ``fgong-amdl.d`` tool distributed with
        ADIPLS.
        """
        from .adipls import ADIPLSStellarModel

        M, R = self.glob[:2]
        r, P, rho, G1, AA = self.var[::-1,[0,3,4,9,14]].T
        m = np.exp(self.var[::-1,1])*M

        ioff = (0 if r[0] < 1e6 else 1)
        nn = len(self.var) + ioff

        # convert profile
        A = np.zeros((nn, 6))

        # we can safely ignore division by 0 here
        with np.errstate(divide='ignore', invalid='ignore'):
            A[ioff:,0] = r/R
            A[ioff:,1] = m/M/(r/R)**3
            A[ioff:,2] = self.G*m*rho/(G1*P*r)
            A[ioff:,3] = G1
            A[ioff:,4] = AA
            A[ioff:,5] = 4.*np.pi*rho*r**3/m

        A[0,0] = 0.
        A[0,1] = 4.*np.pi/3.*rho[0]*R**3/M
        A[0,2] = 0.
        A[0,3] = G1[0]
        A[0,4] = 0.
        A[0,5] = 3.

        # convert header
        D = np.zeros(8)
        D[0] = M
        D[1] = R
        D[2] = P[0]
        D[3] = rho[0]

        # second derivatives at centre are given
        if self.glob[10] < 0.:
            D[4] = -self.glob[10]/G1[0]
            D[5] = -self.glob[11]
        else:
            D[4] = 4.*np.pi/3.*self.G*(rho[0]*R)**2/(P[0]*G1[0])
            # D[5] = np.nanmax((A[1:,4]/A[1:,0]**2)[A[1:,0]<0.05])
            # D[5] = np.max((D[5], 0.))+D[4]
            D[5] = D[4]

        D[6] = -1.
        D[7] = 0.

        if A[-1,4] <= 10.:
            # chop off outermost point
            A = A[:-1]
            nn -= 1

        return ADIPLSStellarModel(D, A, G=self.G)

    def to_gyre(self, version=None):
        """Convert the model to a :py:class:`GYREStellarModel` object.

        Parameters
        ----------
        version: int, optional
            Specify GYRE format version number times 100. i.e.,
            ``version=101`` produce a file with data version 1.01.  If
            ``None`` (the default), the latest version available in
            TOMSO is used.
       """
        from .gyre import gyre_header_dtypes, gyre_data_dtypes, GYREStellarModel

        if version is None:
            version = max([k for k in gyre_header_dtypes.keys()])

        header = np.zeros(1, gyre_header_dtypes[version])
        header['M'] = self.glob[0]
        header['R'] = self.glob[1]
        header['L'] = self.glob[2]

        if version > 1:
            header['version'] = version

        data = np.zeros(self.nn, gyre_data_dtypes[version])
        # data['r'] = self.var[:,0]
        # data['T'] = self.var[:,2]
        # data['P'] = self.var[:,3]
        # data['rho'] = self.var[:,4]

        # if np.all(np.diff(data['r']) <= 0):
        #     return GYREStellarModel(header, data[::-1], G=self.G)
        # else:
        #     return GYREStellarModel(header, data, G=self.G)

        g = GYREStellarModel(header[0], data, G=self.G)

        g.r = self.r
        g.m = self.m
        g.T = self.T
        g.P = self.P
        g.rho = self.rho
        g.Gamma_1 = self.Gamma_1
        g.N2 = self.N2
        g.kappa = self.kappa
        g.L_r = self.L_r
        g.data['nabla_ad'] = self.var[:,10]
        g.data['delta'] = self.var[:,11]

        # The definitions of epsilon in FGONG and GYRE formats might
        # be different.  Compute non-adiabatic modes at your peril!
        if version < 101:
            g.data['eps_tot'] = self.epsilon
        else:
            g.data['eps'] = self.epsilon

        if np.all(np.diff(g.r) <= 0):
            g.data = g.data[::-1]

        g.data['k'] = np.arange(self.nn) + 1
        return g

    # FGONG parameters that can be derived from data
    @property
    def iconst(self): return len(self.glob)

    @property
    def nn(self): return self.var.shape[0]

    @property
    def ivar(self): return self.var.shape[1]

    # Various properties for easier access to the data in `glob` and
    # `var`.

    @property
    def M(self): return self.glob[0]

    @M.setter
    def M(self, val): self.glob[0] = val

    @property
    def R(self): return self.glob[1]

    @R.setter
    def R(self, val): self.glob[1] = val

    @property
    def L(self): return self.glob[2]

    @L.setter
    def L(self, val): self.glob[2] = val

    @property
    def r(self): return self.var[:,0]

    @r.setter
    def r(self, val):
        self.var[:,0] = val
        self.var[:,17] = self.R-val

    @property
    def lnq(self): return self.var[:,1]

    @lnq.setter
    def lnq(self, val): self.var[:,1] = val

    @property
    def T(self): return self.var[:,2]

    @T.setter
    def T(self, val): self.var[:,2] = val

    @property
    def P(self): return self.var[:,3]

    @P.setter
    def P(self, val): self.var[:,3] = val

    @property
    def rho(self): return self.var[:,4]

    @rho.setter
    def rho(self, val): self.var[:,4] = val

    @property
    def X(self): return self.var[:,5]

    @X.setter
    def X(self, val): self.var[:,5] = val

    @property
    def L_r(self): return self.var[:,6]

    @L_r.setter
    def L_r(self, val): self.var[:,6] = val

    @property
    def kappa(self): return self.var[:,7]

    @kappa.setter
    def kappa(self): self.var[:,7] = val

    @property
    def epsilon(self): return self.var[:,8]

    @epsilon.setter
    def epsilon(self, val): self.var[:,8] = val

    @property
    def Gamma_1(self): return self.var[:,9]

    @Gamma_1.setter
    def Gamma_1(self, val): self.var[:,9] = val

    @property
    def G1(self): return self.var[:,9]

    @G1.setter
    def G1(self, val): self.var[:,9] = val

    @property
    def grad_a(self): return self.var[:,10]

    @grad_a.setter
    def grad_a(self, val): self.var[:,10] = val

    @property
    def cp(self): return self.var[:,12]

    @cp.setter
    def cp(self, val): self.var[:,12] = val

    @property
    def AA(self): return self.var[:,14]

    @AA.setter
    def AA(self, val): self.var[:,14] = val

    @property
    def Z(self): return self.var[:,16]

    @Z.setter
    def Z(self, val): self.var[:,16] = val

    # Some convenient quantities derived from `glob` and `var`.
    @property
    def x(self): return self.r/self.R

    @x.setter
    def x(self, val): self.r = val*self.R

    @property
    def q(self): return np.exp(self.lnq)

    @q.setter
    def q(self, val): self.lnq = np.log(val)

    @property
    def m(self): return self.q*self.M

    @m.setter
    def m(self, val): self.q = val/self.M

    @property
    @regularize()
    def N2(self): return self.AA*self.g/self.r

    @property
    @regularize(y0=3)
    def U(self): return 4.*np.pi*self.rho*self.r**3/self.m

    @property
    @regularize()
    def V(self): return self.G*self.m*self.rho/self.P/self.r

    @property
    def Vg(self): return self.V/self.Gamma_1

    @property
    def tau(self):
        if np.all(np.diff(self.x) < 0):
            return -integrate(1./self.cs, self.r)
        else:
            tau = integrate(1./self.cs[::-1], self.r[::-1])[::-1]
            return np.max(tau)-tau
