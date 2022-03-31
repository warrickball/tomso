# -*- coding: utf-8 -*-

"""Functions and classes for reading and writing ADIPLS binary output.  Many
return or contain what I call ``cs`` arrays.  These are defined in Section 8.2 of
the `ADIPLS documentation`_.  They are structured arrays containing
various scalar results from the frequency calculation.

    .. _ADIPLS documentation: https://github.com/MESAHub/mesa/raw/r21.12.1/adipls/adipack.c/notes/adiab.prg.c.pdf
"""
import numpy as np
import warnings
from .constants import G_DEFAULT
from .utils import integrate, complement, regularize
from .utils import AdiabaticStellarModel


def read_one_cs(f):
    """Utility function to parse one ``cs`` array from a binary file
    handle ``f``."""
    cs = np.fromfile(f, dtype=cs_floats, count=1)
    cs = cs.astype(cs_dtypes, copy=False)
    return cs


def load_pointwise_data(filename, ncols):
    """Utility function for common structure of ADIPLS data that has a
    value at each point in a stellar model. e.g. eigenfunction or
    kernel files.

    Parameters
    ----------
    filename: str
        Name of the file to be read.
    ncols: int
        Number of columns in the data.

    Returns
    -------
    css: structured array
        The ``cs`` arrays for each mode.
    data: list of arrays
        The point-wise data arrays for each mode.

    """
    css = []
    data = []

    with open(filename, 'rb') as f:
        while True:
            if not f.read(4): break
            css.append(read_one_cs(f))
            nnw = np.fromfile(f, dtype='i', count=1)[0]
            row = np.fromfile(f, dtype='d', count=ncols*nnw).reshape((-1, ncols))
            data.append(row)
            f.read(4)

    return np.squeeze(css), np.squeeze(data)


def load_agsm(filename):
    """Reads an ADIPLS grand summary file and returns an
    :py:class:`ADIPLSGrandSummary` object.

    Parameters
    ----------
    filename: str
        Name of the grand summary file, usually starting or ending
        with ``agsm``.

    Returns
    -------
    agsm: :py:class:`ADIPLSGrandSummary` object
        Contains data from the ``cs`` arrays for each mode.
    """

    css = []

    with open(filename, 'rb') as f:
        while True:
            if not f.read(4): break
            css.append(read_one_cs(f))
            f.read(4)

    return ADIPLSGrandSummary(np.squeeze(css))


def load_amde(filename, nfmode=1):
    """Reads an ADIPLS eigenfunction file written with the specified value
    of ``nfmode`` in the input file (either 1, 2 or 3)
    and returns an :py:class:`ADIPLSEigenfunctions` object.

    Parameters
    ----------
    filename: str
        Name of the eigenfunction file, usually starting or ending
        with ``amde``.
    nfmode: int, optional
        ADIPLS's ``nfmode`` parameter, which determines the format of
        the eigenfunction data.  See Section 8.4 of the `ADIPLS
        documentation`_ for details of the output.

    Returns
    -------
    amde: :py:class:`ADIPLSEigenfunctions`
        The eigenfunction data for the modes.
    """

    if nfmode == 1:
        css, data = load_pointwise_data(filename, 7)
        x = data[0,:,0]
    elif nfmode == 2 or nfmode == 3:
        # thanks to Vincent Boening for this
        ncols = 2
        css = []
        data = []
        with open(filename, 'rb') as f:
            f.read(4)
            nnw = np.fromfile(f, dtype='i', count=1)[0]
            x = np.fromfile(f, dtype='d', count=nnw)
            f.read(4)

            while True:
                if not f.read(4): break
                css.append(read_one_cs(f))
                row = np.fromfile(f, dtype='d', count=ncols*nnw).reshape((-1, ncols))
                data.append(row)
                f.read(4)
    else:
        raise ValueError('nfmode must be 1, 2 or 3 but got %i' % nfmode)

    return ADIPLSEigenfunctions(np.squeeze(css), np.squeeze(data), x=x, nfmode=nfmode)


def load_amdl(filename, live_dangerously=False, G=G_DEFAULT):
    """Reads an ADIPLS model file and returns an
    :py:class:`ADIPLSStellarModel` object.  See Section 5 of the
    `ADIPLS documentation`_ for details.

    Parameters
    ----------
    filename: str
        Name of the model file, usually starting or ending with ``amdl``.
    live_dangerously: bool, optional
        If ``True``, load the file even if it looks like it might be
        too large for an AMDL file (i.e. has more than a million points).
    G: float, optional
        Value for the gravitational constant, in cgs units.  If not
        given (which is the default behaviour), we use the module-wise
        default value.

    Returns
    -------
    amdl: :py:class:`ADIPLSStellarModel` object
        Contains the model data.

    """

    with open(filename, 'rb') as f:
        f.read(4)
        nmod = np.fromfile(f, dtype='i', count=1)[0]
        nn = np.fromfile(f, dtype='i', count=1)[0]
        if not live_dangerously and nn > 1000000:
            raise IOError("Model appears to have %i points; "
                          "it probably isn't an AMDL file. "
                          "If you're sure that it is, try again "
                          "with live_dangerously=True" % nn)

        D = np.fromfile(f, dtype='d', count=8)
        A = np.fromfile(f, dtype='d', count=6*nn).reshape((-1,6))
        f.read(4)
        # check that this is the end of the file

    return ADIPLSStellarModel(D, A, nmod=nmod, G=G)


def load_rkr(filename):
    """Reads an ADIPLS rotational kernel file and returns an
    :py:class:`ADIPLSRotationKernels` object.

    Parameters
    ----------
    filename: str
        Name of the kernel file, usually starting or ending with
        ``rkr``.

    Returns
    -------
    rkr: :py:class:`ADIPLSRotationKernels` object
        The kernel arrays for each mode.

    """

    return ADIPLSRotationKernels(*load_pointwise_data(filename, 2))


def kernels(cs, eig, D, A, G=G_DEFAULT, alpha=None):
    """Returns the density and squared sound speed kernels.  I have tried
    to make this as notationally similar to Gough & Thompson (1991) as
    possible.  The kernels are normalized to have unit integrals over
    the radius *r*.

    Parameters
    ----------
    cs: structured array
        The ``cs`` array for the mode.
    eig: np.array, shape(N,7)
        Eigenfunction data for the mode, as returned by
        :py:meth:`load_amde`.
    D: 1-d array
        Global data, as defined by eq. (5.2) of the `ADIPLS
        documentation`_ and returned by :py:meth:`load_amdl`.
    A: 2-d array
        Point-wise data, as defined by eq. (5.1) of the `ADIPLS
        documentation`_ and returned by :py:meth:`load_amdl`.
    G: float, optional
        Value for the gravitational constant, in cgs units.  If not
        given (which is the default behaviour), we use the module-wise
        default value.
    alpha: float, optional
        Coefficient of the complementary function.  If ``None``, computed
        as in Michael Thompson's kernel code.

    Returns
    -------
    K_cs2: np.array, length N
        The sound speed squared structure kernel.
    K_rho: np.array, length N
        The density structure kernel.

    """

    l = cs['l']
    M, R, P_c, rho_c = D[:4]                # mass and radius from FGONG
    y = eig.T
    sigma2 = cs['sigma2']
    omega = np.sqrt(sigma2*G*M/R**3)        # convert to angular frequency
    L2 = l*(l+1)

    x = A[:,0]
    r = x*R                                 # radial co-ordinate
    m = A[:,1]*x**3*M                       # mass co-ordinate
    g = G*m/r**2                            # gravity
    g[x==0] = 0.
    rho = A[:,1]*A[:,5]*M/4./np.pi/R**3     # density
    rho[x==0] = rho_c
    G1 = A[:,3]                             # first adiabatic index
    P = G*m*rho/G1/r/A[:,2]                 # pressure
    P[x==0] = P_c
    cs2 = G1*P/rho                          # square of the sound speed
    A1 = A[:,1]
    A2 = A[:,2]
    Vg = A2[:]
    drho_dr = -(A[:,4]+A[:,2])*rho/r        # density gradient
    drho_dr[x==0] = 0.

    xi_r = y[1]*R

    if l == 0:
        xi_h = 0.*xi_r  # radial modes have zero horizontal component
        chi = Vg/x*(y[1]-sigma2/A1/x*y[2])
        dxi_r_dr = chi - 2.*y[1]/x
        dPhi_dr = -4.*np.pi*G*rho*xi_r
        Phi = -complement(dPhi_dr, r)  # but actually you don't even need it
    elif l > 0:
        xi_h = y[2]*R/L2
        eta = L2*A1/sigma2
        chi = Vg/x*(y[1]-y[2]/eta-y[3])
        dxi_r_dr = chi - 2.*y[1]/x + y[2]/x
        dPhi_dr = -g/x*(y[3] + y[4]) - y[3]*R*(4.*np.pi*G*rho - 2.*g/r)
        Phi = -g*R*y[3]
    else:
        raise ValueError('l must be non-negative')

    chi[x==0] = 0.
    dxi_r_dr[x==0] = 0.
    dPhi_dr[x==0] = 0.
    Phi_r = Phi/r
    Phi_r[x==0] = 0.

    S = np.trapz((xi_r**2 + L2*xi_h**2)*rho*r**2, r)

    K_cs2 = rho*cs2*chi**2*r**2  # c.f. equation (60)
    K_cs2 = K_cs2/S/omega**2/2.

    # following InversionKit (103)
    K_rho = cs2*chi**2 - omega**2*(xi_r**2+L2*xi_h**2) \
        - 2.*g*xi_r*(chi - dxi_r_dr) \
        + 4.*np.pi*G*rho*xi_r**2 \
        - 4.*np.pi*G*complement((2.*rho*chi+xi_r*drho_dr)*xi_r, r) \
        + 2.*(xi_r*dPhi_dr + L2*xi_h*Phi_r)
    K_rho = K_rho*rho*r**2/2./S/omega**2

    comp = rho*r**2
    if alpha is None:
        alpha = np.trapz(K_rho*comp, r)/np.trapz(comp*comp, r)

    K_rho = K_rho - alpha*comp

    return K_cs2, K_rho


cs_dtypes = [('xmod','float'), ('M','float'), ('R','float'),
             ('P_c','float'), ('rho_c','float'), ('D_5','float'),
             ('D_6','float'), ('D_7','float'), ('D_8','float'),
             ('A_2(x_s)','float'), ('A_5(x_s)','float'),
             ('x_1','float'), ('sigma2_Omega','float'),
             ('x_f','float'), ('fctsbc','int'), ('fcttbc','int'),
             ('lambda','float'), ('l','int'), ('n','int'),
             ('sigma2','float'), ('sigma2_c','float'),
             ('y_1,max','float'), ('x_max', 'float'), ('E','float'),
             ('Pi_E','float'), ('Pi_V','float'), ('nu_V','float'),
             ('ddsig','float'), ('ddsol','float'),
             ('y_1(x_s)','float'), ('y_2(x_s)','float'),
             ('y_3(x_s)','float'), ('y_4(x_s)','float'),
             ('z_1,max','float'), ('xhat_max','float'),
             ('beta_nl','float'), ('nu_Ri','float'), ('m','int')]

for i in range(len(cs_dtypes), 50):
    cs_dtypes.append(('col%i' % i, 'float'))

cs_floats = [(k, 'float') for (k,v) in cs_dtypes]


class ADIPLSStellarModel(AdiabaticStellarModel):
    """A class that contains and allows one to manipulate the data in a
    stellar model stored in ADIPLS's internal binary model format.
    See Section 5 of the `ADIPLS documentation`_ for details.

    This will usually be provided from a file by using
    :py:meth:`load_amdl` but an object can be constructed from any
    similarly structured arrays.

    The main attributes are the **D** and **A** arrays, which follow
    the definitions in the ADIPLS documentation.  The data in these
    arrays can be accessed via the attributes with more
    physically-meaningful names (e.g. the radius is
    ``ADIPLSStellarModel.r``).

    Some of these values can also be set via the attributes if doing
    so is unambiguous. For example, the fractional radius **x** is not a
    member of the **var** array but setting **x** will assign the actual
    radius **r**, which is the first column of **var**.  Values that are
    settable are indicated in the list of parameters.

    Parameters
    ----------
    D: 1-d array
        Global data, as defined by eq. (5.2) of the ADIPLS
        documentation.
    A: 2-d array
        Point-wise data, as defined by eq. (5.1) of the ADIPLS
        documentation.
    nmod: int, optional
        The model number.  I'm not sure what it's used for but it
        doesn't seem to matter.
    G: float, optional
        Value for the gravitational constant, in cgs units.  If not
        given (which is the default behaviour), we use the module-wise
        default value.

    Attributes
    ----------
    nn: int
        number of points in stellar model (i.e. number of rows in **A**)
    M: float, settable
        total mass
    R: float, settable
        photospheric radius
    P_c: float, settable
        central pressure
    rho_c: float, settable
        central density
    x: NumPy array, settable
        fractional radius co-ordinate
    q: NumPy array, settable
        fractional mass co-ordinate
    lnq: NumPy array, settable
        natural logarithm of the fractional mass co-ordinate
    Vg: NumPy array
        homology invariant *V/Gamma_1*
    Gamma_1: NumPy array, settable
        first adiabatic index, aliased by **G1**
    G1: NumPy array, settable
        first adiabatic index, alias of **Gamma_1**
    AA: NumPy array, settable
        Ledoux discriminant
    U: NumPy array
        homology invariant *dlnm/dlnr*
    V: NumPy array
        homology invariant *dlnP/dlnr*
    r: NumPy array, settable
        radius co-ordinate
    m: NumPy array, settable
        mass co-ordinate
    P: NumPy array
        pressure
    rho: NumPy array
        density
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
    tau: NumPy array
        acoustic depth
    """
    def __init__(self, D, A, nmod=0, G=G_DEFAULT):
        self.D = D
        self.A = A
        self.nmod = nmod
        self.G = G

    def __len__(self):
        return len(self.A)

    def __repr__(self):
        with np.printoptions(threshold=10):
            return('ADIPLSStellarModel(\nD=\n%s,\nA=\n%s,\nG=%.15g,\nnmod=%s\n)' % (self.D, self.A, self.G, self.nmod))

    def to_file(self, filename):
        """Save the model to an ADIPLS binary stellar model file (usually
        either starting or ending with `amdl`).

        Parameters
        ----------
        filename: str
            Filename to which the data is written.
        """
        nn = len(self.A)
        length = np.array(8*(1+8+6*nn), dtype=np.int32)
        with open(filename, 'wb') as f:
            length.tofile(f)
            np.array((self.nmod,), dtype=np.int32).tofile(f)
            np.array((nn,), dtype=np.int32).tofile(f)
            self.D.tofile(f)
            self.A.tofile(f)
            length.tofile(f)

    def to_fgong(self, reverse=True, ivers=1300):
        """Convert the model to an :py:class:`~tomso.fgong.FGONG` object.

        Note that the ADIPLS binary format only has the data necessary
        to compute adiabiatic stellar oscillations, so the FGONG will
        be missing some data (e.g. temperature, luminosity).

        Parameters
        ----------
        reverse: bool, optional
            If ``True`` (the default), store the FGONG data ordered
            from the surface to the centre.  Otherwise, store the
            FGONG data ordered from the centre to the surface.
        """
        from .fgong import FGONG

        M, R = self.D[:2]

        glob = np.zeros(15)
        var = np.zeros((len(self.A), 40))

        r = self.A[:,0]*R
        q = self.A[:,1]*self.A[:,0]**3
        m = q*M
        G1 = self.A[:,3]
        AA = self.A[:,4]

        # we can safely ignore division by 0 here
        with np.errstate(divide='ignore', invalid='ignore'):
            lnq = np.log(q)
            rho = self.A[:,5]*m/(4.*np.pi*r**3)
            P = self.G*m*rho/(G1*r*self.A[:,2])

        P[0] = self.D[2]
        rho[0] = self.D[3]

        var[:,0] = r
        var[:,1] = lnq
        var[:,3] = P
        var[:,4] = rho
        var[:,9] = G1
        var[:,14] = AA

        glob[0] = M
        glob[1] = R
        glob[10] = -self.D[4]*G1[0]
        glob[11] = -self.D[5]

        if reverse:
            var = var[::-1]

        return FGONG(glob, var, G=self.G, ivers=ivers)


    def to_gyre(self, version=None):
        """Convert the model to an :py:class:`~tomso.gyre.GYREStellarModel`
        object.

        Note that the ADIPLS binary format only has the data necessary
        to compute adiabiatic stellar oscillations, so the GYRE
        stellar model will be missing some data (e.g. temperature,
        luminosity).

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
        header['M'] = self.D[0]
        header['R'] = self.D[1]
        header['L'] = 42.0
        header['version'] = version

        data = np.ones(self.nn, gyre_data_dtypes[version])
        g = GYREStellarModel(header[0], data, G=self.G)

        g.r = self.r
        g.m = self.m
        g.P = self.P
        g.rho = self.rho
        g.Gamma_1 = self.Gamma_1
        g.N2 = self.N2

        # GYRE doesn't know if it's doing adiabatic or non-adiabatic
        # modes when it reads the file, so it does some calculations
        # expecting meaningful data.  We fudge this so we don't get
        # FPEs.
        g.kappa = 42.0
        g.L_r = 42.0
        if version < 101:
            g.data['eps_tot'] = 42.0
        else:
            g.data['eps'] = 42.0

        g.data['k'] = np.arange(self.nn) + 1
        return g


    # AMDL parameters that can be derived from data
    @property
    def nn(self): return len(self.A)

    # Various properties for easier access to the data in `glob` and
    # `var`.

    @property
    def M(self): return self.D[0]

    @M.setter
    def M(self, val): self.D[0] = val

    @property
    def R(self): return self.D[1]

    @R.setter
    def R(self, val): self.D[1] = val

    @property
    def P_c(self): return self.D[2]

    @P_c.setter
    def P_c(self, val): self.D[2] = val

    @property
    def rho_c(self): return self.D[3]

    @rho_c.setter
    def rho_c(self, val): self.D[3] = val

    @property
    def x(self): return self.A[:,0]

    @x.setter
    def x(self, val): self.A[:,0] = val

    @property
    def q(self): return self.A[:,1]*self.x**3

    @q.setter
    def q(self, val): self.A[:,1] = val/self.x**3

    @property
    def Vg(self): return self.A[:,2]

    @Vg.setter
    def Vg(self, val): self.A[:,2] = val

    @property
    def Gamma_1(self): return self.A[:,3]

    @Gamma_1.setter
    def Gamma_1(self, val): self.A[:,3] = val

    @property
    def G1(self): return self.A[:,3]

    @G1.setter
    def G1(self, val): self.A[:,3] = val

    @property
    def AA(self): return self.A[:,4]

    @AA.setter
    def AA(self, val): self.A[:,4] = val

    @property
    def U(self): return self.A[:,5]

    @U.setter
    def U(self, val): self.A[:,5] = val

    @property
    def V(self): return self.Vg*self.Gamma_1

    @V.setter
    def V(self, val): self.Vg = val/self.Gamma_1

    @property
    def r(self): return self.x*self.R

    @r.setter
    def r(self, val): self.x = val/self.R

    @property
    def m(self): return self.q*self.M

    @m.setter
    def m(self, val): self.q = val/self.M

    @property
    @regularize(y0=-np.inf, x0=1e-308)
    def lnq(self): return np.log(self.q)

    @lnq.setter
    def lnq(self, val): self.q = np.exp(val)

    @property
    def P(self):
        with np.errstate(invalid='ignore'):
            val = self.G*self.m*self.rho/(self.Gamma_1*self.r*self.A[:,2])

        val[self.x==0] = self.P_c
        return val

    @property
    def rho(self):
        with np.errstate(invalid='ignore'):
            val = self.A[:,5]*self.m/self.r**3/4./np.pi

        val[self.x==0] = self.rho_c
        return val

    @property
    @regularize()
    def N2(self): return self.AA*self.g/self.r

    @property
    def tau(self):
        tau = integrate(1./self.cs[::-1], self.r[::-1])[::-1]
        return np.max(tau)-tau


class ADIPLSGrandSummary(object):
    """A class that represents the information for a set of mode
    frequencies, loaded from an ADIPLS grand summary file (often
    starting or ending with ``agsm``).  The main data is stored in the
    ``css`` attribute, which is a structured array.  This will usually
    be provided from a file by using :py:meth:`load_agsm` but an object can be
    constructed from any similarly structured array.

    A subset of the information in the ``css`` array is made available
    through attributes.

    Parameters
    ----------
    css: structured NumPy array
        The ``cs`` arrays for each mode.

    Attributes
    ----------
    G: float
        gravitational constant
    M: float
        total mass
    R: float
        photospheric radius
    l: NumPy array of ints
        angular degrees
    n: NumPy array of ints
        radial orders
    sigma2: NumPy array of floats
        square of the dimensionless angular eigenfrequency
    sigma2_c: NumPy array of floats
        square of the dimensionless angular eigenfrequency corrected
        for the Cowling approximation
    Pi_E: NumPy array of floats
        eigenperiod, in seconds
    Pi_V: NumPy array of floats
        variational period, in seconds
    nu_Ri: NumPy array of floats
        cyclic eigenfrequency corrected using Richardson
        extrapolation, in Hz
    nu_V: NumPy array of floats
        variational cyclic frequency, in Hz
    nu_E: NumPy array of floats
        cyclic eigenfrequency, in seconds
    nu_c: NumPy array of floats
        cyclic eigenfrequency corrected for the Cowling approximation,
        in Hz
    nu: NumPy array of floats
        alias of ``nu_c``
    E: NumPy array of floats
        Normalised mode inertia (see eq. (4.3) of ADIPLS notes).  Note
        that ADIPLS's definition is smaller than GYRE's by a factor of
        4π.
    beta: NumPy array of floats
        Weight for rotation kernel (see eq. (4.7) of ADIPLS notes or
        (8.43) of JCD's oscillation notes).

    """
    def __init__(self, css):
        self.css = css

    def __len__(self):
        return len(self.css)

    def __str__(self):
        return '\n'.join([
            '%s' % type(self),
            'G    %11.6g cm³/g/s²' % self.G,
            'M    %9.3e g    %7.3f Msun' % (self.M, self.M/1.98841e33),
            'R    %9.3e cm   %7.3f Rsun' % (self.R, self.R/695.7e8)])

    def __repr__(self):
        with np.printoptions(threshold=10):
            return('ADIPLSGrandSummary(\ncss=\n%s)' % self.css)

    @property
    def G(self): return self.R**3/self.M/self.sigma2[0]*(2.*np.pi/self.Pi_E[0])**2

    @property
    def M(self): return self.css[0]['M']

    @property
    def R(self): return self.css[0]['R']

    @property
    def l(self): return self.css['l']

    @property
    def n(self): return self.css['n']

    @property
    def sigma2(self): return self.css['sigma2']

    @property
    def sigma2_c(self): return self.css['sigma2_c']

    @property
    def Pi_E(self): return self.css['Pi_E']*60.0

    @property
    def Pi_V(self): return self.css['Pi_V']*60.0

    @property
    def nu_Ri(self): return self.css['nu_Ri']/1e3

    @property
    def nu_V(self): return self.css['nu_V']/1e3

    @property
    def nu_E(self): return 1/self.Pi_E

    @property
    def nu_c(self): return np.sqrt(self.sigma2_c/self.sigma2)/self.Pi_E

    @property
    def nu(self): return self.nu_c

    @property
    def E(self): return self.css['E']

    @property
    def beta(self): return self.css['beta_nl']

    def index_ln(self, l, n):
        """Returns the index of mode with angular degree *l*
        and radial order *n*."""
        return np.where((self.l==l)&(self.n==n))[0][0]

    def index_nl(self, n, l):
        """Returns the index of mode with radial order *n*
        and angular degree *l*."""
        return self.index_ln(l, n)


class ADIPLSEigenfunctions(ADIPLSGrandSummary):
    """A class that represents the information for a set of eigenfunction
    data kernels produced by ADIPLS.  This will usually be provided
    from a file by using :py:meth:`load_amde` but an object can be
    constructed from any similarly structured array.

    Parameters
    ----------
    css: structured NumPy array
        The ``cs`` arrays for each mode.
    eigs: 3-d NumPy array
        The eigenfunction arrays for each mode.  The nth element of
        the array has the eigenfunction data for the nth mode, in
        the same order as the summary data in *css*.  The number of
        rows in the array for a given mode is the number of meshpoints
        in the model.  The number of columns is either 6 or 2,
        depending on **nfmode**.
    nfmode: int
        The output mode used by ADIPLS' when the data was stored.
    x: NumPy array, optional
        If **nfmode** is 2 or 3, the fractional radius must be
        provided separately.  If **nfmode** is 1, it will be inferred
        from the eigenfunction data if not explicitly provided.


    This class has all the attributes of
    :py:class:`ADIPLSGrandSummary` as well as the following extras.

    Attributes
    ----------
    x: NumPy array
        fractional radius co-ordinate
    eigs: list of NumPy arrays
        The nth row is the eigenfunction data for the nth mode, in
        the same order as the summary data in *css*.

    """
    # TODO: add some derived properties: xi_r, xi_h
    # TODO: add access to eigenfunctions by n and l, like rotation kernels
    def __init__(self, css, eigs, nfmode=1, x=None):
        ADIPLSGrandSummary.__init__(self, css)

        self.nfmode = nfmode
        if nfmode == 1:
            if x is None:
                self.x = eigs[0,:,0]
            else:
                self.x = x
        elif nfmode == 2 or nfmode == 3:
            self.x = x
        else:
            raise ValueError('nfmode must be 1, 2 or 3, not %i' % nfmode)

        self.eigs = eigs

    def __str__(self):
        return super(ADIPLSEigenfunctions, self).__str__()

    def __repr__(self):
        with np.printoptions(threshold=10):
            return('ADIPLSEigenfunctions(nfmode=%i,\ncss=\n%s,\neigs=\n%s)' % (self.nfmode, self.css, self.eigs))

    def eig_ln(self, l, n):
        "Load eigenfunction by *l* and *n*."
        return self.eigs[(self.l==l)&(self.n==n)][0]

    def eig_nl(self, n, l):
        "Load eigenfunction by *n* and *l*."
        return self.eig_ln(l, n)


class ADIPLSRotationKernels(ADIPLSGrandSummary):
    """A class that represents the information for a set of rotational
    kernels produced by ADIPLS.  This will usually be provided from a
    file by using :py:meth:`load_rkr` but an object can be constructed
    from any similarly structured array.

    Parameters
    ----------
    css: structured NumPy array
        The ``cs`` arrays for each mode.
    rkrs: list of arrays
        The kernel arrays for each mode.  Each array has two columns:
        the fractional radius :math:`x` and the kernel :math:`K(x)`.


    This class has all the attributes of
    :py:class:`ADIPLSGrandSummary` as well as the following extras.

    Attributes
    ----------
    x: NumPy array
        fractional radius co-ordinate
    K: list of NumPy arrays
        The nth row is the rotation kernel for the nth mode, in
        the same order as the summary data in *css*.  The mode with
        radial order *n* and angular degree *l* can be accessed by the
        functions ``K_ln(l,n)`` or ``K_nl(n,l)``.

    """
    def __init__(self, css, rkr):
        ADIPLSGrandSummary.__init__(self, css)
        self.x = rkr[0,:,0]
        self.K = rkr[:,:,1]

    def __str__(self):
        return super(ADIPLSRotationKernels, self).__str__()

    def __repr__(self):
        with np.printoptions(threshold=10):
            return('ADIPLSRotationKernels(\ncss=\n%s,\nx=%s,\nK=\n%s)' % (self.css, self.x, self.K))

    def K_ln(self, l, n):
        "Load kernel by angular degree *l* and radial order *n*."
        return self.K[(self.l==l)&(self.n==n)][0]

    def K_nl(self, n, l):
        "Load kernel by radial order *n* and angular degree *l*."
        return self.K_ln(l, n)
