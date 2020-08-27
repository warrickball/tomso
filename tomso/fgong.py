# -*- coding: utf-8 -*-

"""Functions for manipulating FGONG files.  These are provided through
the **FGONG** object and a module function to read an **FGONG** object
from a file.

"""

import numpy as np
import warnings
from .adipls import fgong_to_amdl
from .utils import DEFAULT_G
from .utils import integrate, tomso_open, get_Teff, regularize
from .utils import BaseStellarModel


def load_fgong(filename, fmt='ivers', return_comment=False,
               return_object=True, G=None):
    """Given an FGONG file, returns NumPy arrays `glob` and `var` that
    correspond to the scalar and point-wise variables, as specified
    in the `FGONG format`_.

    .. _FGONG format: https://www.astro.up.pt/corot/ntools/docs/CoRoT_ESTA_Files.pdf

    Also returns the first four lines of the file as a `comment`, if
    desired.

    The version number `ivers` is used to infer the format of floats
    if `fmt='ivers'`.

    If `return_object` is `True`, instead returns an :py:class:`FGONG`
    object.  This will become default behaviour from v0.0.12.  The old
    behaviour will be dropped completely from v0.1.0.

    Parameters
    ----------
    filename: str
        Name of the FGONG file to read.
    fmt: str, optional
        Format string for floats in `glob` and `var`.  If `'ivers'`,
        uses `%16.9E' if the file's `ivers < 1000` or `%26.18E3` if
        `ivers >= 1000`.  If `'auto'`, tries to guess the size of each
        float. (default: 'ivers')
    return_comment: bool, optional
        If ``True``, return the first four lines of the FGONG file.
        These are comments that are not used in any calculations.

    Returns
    -------
    glob: NumPy array
        The scalar (or global) variables for the stellar model
    var: NumPy array
        The point-wise variables for the stellar model. i.e. things
        that vary through the star like temperature, density, etc.
    comment: list of strs, optional
        The first four lines of the FGONG file.  These are comments
        that are not used in any calculations.  Only returned if
        ``return_comment=True``.

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

    if return_object:
        return FGONG(glob, var, ivers=ivers, G=G,
                     description=comment)
    else:
        warnings.warn("From tomso 0.1.0+, `fgong.load_fgong` will only "
                      "return an `FGONG` object: use `return_object=True` "
                      "to mimic future behaviour",
                      FutureWarning)
        if return_comment:
            return glob, var, comment
        else:
            return glob, var


def save_fgong(filename, glob, var, ivers=1300, comment=['','','',''],
               float_formatter='ivers'):
    """Given data for an FGONG file in the format returned by
    :py:meth:`~tomso.fgong.load_fgong` (i.e. two NumPy arrays and a
    possible header), writes the data to a file.

    This function will be dropped from v0.1.0 in favour of the `to_file`
    function of the :py:class:`FGONG` object.

    Parameters
    ----------
    filename: str
        Filename to which FGONG data is written.
    glob: NumPy array
        The global variables for the stellar model.
    var: NumPy array
        The point-wise variables for the stellar model. i.e. things
        that vary through the star like temperature, density, etc.
    ivers: int, optional
        The integer indicating the version number of the file.
        (default=0)
    comment: list of strs, optional
        The first four lines of the FGONG file, which usually contain
        notes about the stellar model.

    """
    nn, ivar = var.shape
    iconst = len(glob)

    if float_formatter == 'ivers':
        if ivers < 1000:
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
        f.write('\n'.join(comment) + '\n')

        line = '%10i'*4 % (nn, iconst, ivar, ivers) + '\n'
        f.write(line)

        for i, val in enumerate(glob):
            f.write(ff(val))
            if i % 5 == 4:
                f.write('\n')

        if i % 5 != 4:
            f.write('\n')

        for row in var:
            for i, val in enumerate(row):
                f.write(ff(val))
                if i % 5 == 4:
                    f.write('\n')

        if i % 5 != 4:
            f.write('\n')


def fgong_get(key_or_keys, glob, var, reverse=False, G=DEFAULT_G):
    """Retrieves physical properties of a FGONG model from the ``glob`` and
    ``var`` arrays.

    This function will be dropped from v0.1.0 in favour of the
    attributes of the :py:class:`FGONG` object.

    Parameters
    ----------
    key_or_keys: str or list of strs
        The desired variable or a list of desired variables.  Current
        options are:

        - ``M``: total mass (float)
        - ``R``: photospheric radius (float)
        - ``L``: total luminosity (float)
        - ``r``: radius (array)
        - ``x``: fractional radius (array)
        - ``m``: mass co-ordinate (array)
        - ``q``: fractional mass co-ordinate (array)
        - ``g``: gravity (array)
        - ``rho``: density (array)
        - ``P``: pressure (array)
        - ``AA``: Ledoux discriminant (array)
        - ``Hp``: pressure scale height (array)
        - ``Hrho``: density scale height (array)
        - ``G1``: first adiabatic index (array)
        - ``T``: temperature (array)
        - ``X``: hydrogen abundance (array)
        - ``L_r``: luminosity at radius ``r`` (array)
        - ``kappa``: opacity (array)
        - ``epsilon``: specific energy generation rate (array)
        - ``cp``: specific heat capacity (array)
        - ``cs2``: sound speed squared (array)
        - ``cs``: sound speed (array)
        - ``tau``: acoustic depth (array)

        For example, if ``glob`` and ``var`` have been returned from
        :py:meth:`~tomso.fgong.load_fgong`, you could use

        >>> M, m = fgong.fgong_get(['M', 'm'], glob, var)

        to get the total mass and mass co-ordinate.  If you only want
        one variable, you don't need to use a list.  The return type
        is just the one corresponding float or array.  So, to get a
        single variable you could use either

        >>> x, = fgong.fgong_get(['x'], glob, var)

        or

        >>> x = fgong.fgong_get('x', glob, var)

    glob: NumPy array
        The scalar (or global) variables for the stellar model
    var: NumPy array
        The point-wise variables for the stellar model. i.e. things
        that vary through the star like temperature, density, etc.
    reverse: bool (optional)
        If ``True``, reverse the arrays so that the first element is
        the centre.
    G: float (optional)
        Value of the gravitational constant.

    Returns
    -------
    output: list of floats and arrays
        A list returning the floats or arrays in the order requested
        by the parameter ``keys``.

    """
    M, R, L = glob[:3]
    r, lnq, T, P, rho, X, L_r, kappa, epsilon, G1 = var[:,:10].T
    cp = var[:,12]
    AA = var[:,14]

    x = r/R
    q = np.exp(lnq)
    m = q*M
    g = G*m/r**2
    Hp = P/(rho*g)
    Hrho = 1/(1/G1/Hp + AA/r)
    cs2 = G1*P/rho                    # square of the sound speed
    cs = np.sqrt(cs2)
    if np.all(np.diff(x) < 0):
        tau = -integrate(1./cs, r)      # acoustic depth
    else:
        tau = integrate(1./cs[::-1], r[::-1])[::-1]
        tau = np.max(tau)-tau

    if type(key_or_keys) == str:
        keys = [key_or_keys]
        just_one = True
    else:
        keys = key_or_keys
        just_one = False

    I = np.arange(len(var), dtype=int)
    if reverse:
        I = I[::-1]

    output = []
    for key in keys:
        if key == 'M': output.append(M)
        elif key == 'R': output.append(R)
        elif key == 'L': output.append(L)
        elif key == 'r': output.append(r[I])
        elif key == 'x': output.append(x[I])
        elif key == 'm': output.append(m[I])
        elif key == 'q': output.append(q[I])
        elif key == 'g': output.append(g[I])
        elif key == 'rho': output.append(rho[I])
        elif key == 'P': output.append(P[I])
        elif key == 'AA': output.append(AA[I])
        elif key == 'Hp': output.append(Hp[I])
        elif key == 'Hrho': output.append(Hrho[I])
        elif key == 'G1': output.append(G1[I])
        elif key == 'T': output.append(T[I])
        elif key == 'X': output.append(X[I])
        elif key == 'L_r': output.append(L_r[I])
        elif key == 'kappa': output.append(kappa[I])
        elif key == 'epsilon': output.append(epsilon[I])
        elif key == 'cp': output.append(cp[I])
        elif key == 'cs2': output.append(cs2[I])
        elif key == 'cs': output.append(cs[I])
        elif key == 'tau': output.append(tau[I])
        else: raise ValueError('%s is not a valid key for fgong.fgong_get' % key)

    if just_one:
        assert(len(output) == 1)
        return output[0]
    else:
        return output


class FGONG(BaseStellarModel):
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
        is close to the module-wise default value.  Otherwise, we use
        the module-wise default value.
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
            if len(glob) >= 14 and np.isclose(glob[14], DEFAULT_G,
                                              rtol=1e-3, atol=0.01e-8):
                self.G = glob[14]
            else:
                self.G = DEFAULT_G
        else:
            self.G = G

    def __len__(self):
        return len(self.var)

    def __repr__(self):
        return('FGONG(\nglob=\n%s,\nvar=\n%s,\ndescription=\n%s' % (self.glob, self.var, '\n'.join(self.description)))

    def to_file(self, filename, float_formatter='ivers'):
        """Save the model to an FGONG file.

        Parameters
        ----------
        filename: str
            Filename to which the data is written.
        fmt: str, optional
            Format string for floating point numbers in the **glob**
            and **var** arrays.
        """
        save_fgong(filename, self.glob, self.var,
                   ivers=self.ivers, comment=self.description,
                   float_formatter=float_formatter)

    def to_amdl(self):
        """Convert the model to an ``ADIPLSStellarModel`` object."""
        from .adipls import ADIPLSStellarModel

        return ADIPLSStellarModel(
            *fgong_to_amdl(self.glob, self.var, G=self.G), G=self.G)

    def to_gyre(self, version=None):
        """Convert the model to a ``GYREStellarModel`` object.

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
    def Teff(self): return get_Teff(self.L, self.R)

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

    # - ``G1``: first adiabatic index (array)
