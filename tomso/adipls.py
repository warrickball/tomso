"""Functions for reading and writing ADIPLS binary output.  Many
return what I call ``cs`` arrays.  These are defined in Section 8.2 of
the `ADIPLS documentation`_.  They are structured arrays containing
various scalar results from the frequency calculation.

    .. _ADIPLS documentation: https://sourceforge.net/p/mesa/code/HEAD/tree/trunk/adipls/adipack.c/notes/adiab.prg.c.pdf
"""
import numpy as np
from tomso.common import integrate, DEFAULT_G

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
    """Reads an ADIPLS grand summary file.

    Parameters
    ----------
    filename: str
        Name of the grand summary file, usually starting or ending
        with ``agsm``.

    Returns
    -------
    css: structured array
        The ``cs`` arrays for each mode.

    """

    css = []

    with open(filename, 'rb') as f:
        while True:
            if not f.read(4): break
            css.append(read_one_cs(f))
            f.read(4)

    return np.squeeze(css)


def load_amde(filename, nfmode=1):
    """Reads an ADIPLS eigenfunction file written with the specified value
    of ``nfmode`` in the input file (either 1, 2 or 3).

    Parameters
    ----------
    filename: str
        Name of the eigenfunction file, usually starting or ending
        with ``amde``.
    nfmode: int, optional
        ADIPLS's ``nfmode`` parameter, which determines the format of
        the eigenfunction data.  See Section 8.4 of the `ADIPLS
        documentation`_ for details of the output.  Note that for
        ``nfmode=2`` or ``3``, the fractional radius is returned as an
        extra (third) component.

    Returns
    -------
    css: structured array
        The ``cs`` arrays for each mode.
    eigs: list of arrays
        The eigenfunction arrays for each mode.  Each array has seven
        columns: the fractional radius :math:`x` and six columns of
        ADIPLS's :math:`y` matrix.  The first four columns of
        :math:`y` are defined by equation (2.5) of the documentation
        and the last two by equations (4.4) and (4.6).
    x: array, optional
        Fractional radius co-ordinate ``x`` of the eigenfunctions.
        Only returned for ``nfmode=2`` or ``3``, in which case the
        radial co-ordinate isn't a part of ``eigs``.

    """

    if nfmode == 1:
        return load_pointwise_data(filename, 7)
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
                
        return np.squeeze(css), np.squeeze(data), x
    else:
        raise ValueError('nfmode must be 1, 2 or 3 but got %i' % nfmode)


def load_amdl(filename, return_nmod=False, live_dangerously=False):
    """Reads an ADIPLS model file.  See Section 5 of the `ADIPLS
    documentation`_ for details.

    Parameters
    ----------
    filename: str
        Name of the model file, usually starting or ending with ``amdl``.
    return_nmod: bool, optional
        If `True`, return the ``nmod`` parameter in the file.
    live_dangerously: bool, optional
        If `True`, load the file even if it looks like it might be
        too large for an AMDL file (i.e. has more than a million points).

    Returns
    -------
    D: 1-d array
        Global data, as defined by eq. (5.2) of the ADIPLS
        documentation.
    A: 2-d array
        Point-wise data, as defined by eq. (5.1) of the ADIPLS
        documentation.
    nmod: int, optional
        The model number.  I'm not sure what it's used for but it
        doesn't seem to matter.  Only returned if `return_nmod=True`.

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

    if return_nmod:
        return D, A, int(nmod)
    else:
        return D, A


def load_rkr(filename):
    """Reads an ADIPLS rotational kernel file.

    Parameters
    ----------
    filename: str
        Name of the kernel file, usually starting or ending with
        ``rkr``.

    Returns
    -------
    css: structured array
        The ``cs`` arrays for each mode.
    rkrs: list of arrays
        The kernel arrays for each mode.  Each array has two columns:
        the fractional radius :math:`x` and the kernel :math:`K(x)`.

    """

    return load_pointwise_data(filename, 2)


def save_amdl(filename, D, A, nmod=0):
    """Writes an ADIPLS model file, given data in the same form as
    returned by :py:meth:`~tomso.adipls.load_amdl`.  See Section 5 of
    the `ADIPLS documentation`_ for details.

    Parameters
    ----------
    filename: str
        Name of the model file, usually starting or ending with amdl.
    D: 1-d array
        Global data, as defined by eq. (5.2) of the `ADIPLS
        documentation`_.
    A: 2-d array
        Point-wise data, as defined by eq. (5.1) of the `ADIPLS
        documentation`_.
    nmod: int, optional
        The model number.  I'm not sure what it's used for but it
        doesn't seem to matter.

    """
    nn = len(A)
    length = np.array(8*(1+8+6*nn), dtype=np.int32)
    with open(filename, 'wb') as f:
        length.tofile(f)
        np.array((nmod,), dtype=np.int32).tofile(f)
        np.array((nn,), dtype=np.int32).tofile(f)
        D.tofile(f)
        A.tofile(f)
        length.tofile(f)


def amdl_get(key_or_keys, D, A, G=DEFAULT_G):
    """Retrieves physical properties of an AMDL model from the ``D`` and
    ``A`` arrays.

    Parameters
    ----------
    keys: list of strs
        A desired variable or a list of desired variables.  Current
        options are:

        - ``M``: total mass (float)
        - ``R``: photospheric radius (float)
        - ``P_c``: central pressure (float)
        - ``rho_c``: central density (float)
        - ``r``: radius (array)
        - ``x``: fractional radius (array)
        - ``m``: mass co-ordinate (array)
        - ``q``: fractional mass co-ordinate (array)
        - ``g``: gravity (array)
        - ``rho``: density (array)
        - ``P``: pressure (array)
        - ``Hp``: pressure scale height (array)
        - ``G1``: first adiabatic index (array)
        - ``cs2``: sound speed squared (array)
        - ``cs``: sound speed (array)
        - ``tau``: acoustic depth

        For example, if ``D`` and ``A`` have been returned from
        :py:meth:`~tomso.adipls.load_amdl`, you could use

        >>> M, m = adipls.amdl_get(['M', 'm'], D, A)

        to get the total mass and mass co-ordinate.  If you only want
        one variable, you don't need to use a list.  The return type
        is just the one corresponding float or array.  So, to get a
        single variable you could use either

        >>> x, = adipls.amdl_get(['x'], D, A)

        or

        >>> x = adipls.amdl_get('x', D, A)

    D: 1-d array
        Global data, as defined by eq. (5.2) of the `ADIPLS
        documentation`_ and returned by
        :py:meth:`~tomso.adipls.load_amdl`.
    A: 2-d array
        Point-wise data, as defined by eq. (5.1) of the `ADIPLS
        documentation`_ and returned by
        :py:meth:`~tomso.adipls.load_amdl`.
    
    Returns
    -------
    output: list of floats and arrays
        A list returning the floats or arrays in the order requested
        by the parameter ``keys``.

    """
    M, R, P_c, rho_c = D[:4]
    x = A[:,0]
    q = A[:,1]*x**3
    m = q*M
    r = x*R
    g = G*m/r**2
    rho = A[:,5]*m/r**3/4./np.pi
    rho[x==0] = rho_c
    G1 = A[:,3]  # first adiabatic index
    P = G*m*rho/G1/r/A[:,2]  # pressure
    P[x==0] = P_c
    Hp = P/(rho*g)  # pressure scale height
    cs2 = G1*P/rho  # square of the sound speed
    cs = np.sqrt(cs2)
    tau = -integrate(1./cs[::-1], r[::-1])[::-1]      # acoustic depth
    
    if type(key_or_keys) == str:
        keys = [key_or_keys]
        just_one = True
    else:
        keys = key_or_keys
        just_one = False

    output = []
    for key in keys:
        if key == 'M': output.append(M)
        elif key == 'R': output.append(R)
        elif key == 'P_c': output.append(P_c)
        elif key == 'rho_c': output.append(rho_c)
        elif key == 'r': output.append(r)
        elif key == 'x': output.append(x)
        elif key == 'm': output.append(m)
        elif key == 'q': output.append(q)
        elif key == 'g': output.append(g)
        elif key == 'rho': output.append(rho)
        elif key == 'P': output.append(P)
        elif key == 'Hp': output.append(Hp)
        elif key == 'G1': output.append(G1)
        elif key == 'cs2': output.append(cs2)
        elif key == 'cs': output.append(cs)
        elif key == 'tau': output.append(tau)
        else: raise ValueError('%s is not a valid key for adipls.amdl_get' % key)

    if just_one:
        assert(len(output) == 1)
        return output[0]
    else:
        return output


def kernels(ell, cs, eig, D, A, G=DEFAULT_G,
           alpha=None):
    """Returns the density and squared sound speed kernels.  I have tried
    to make this as notationally similar to Gough & Thompson (1991) as
    possible.  The kernels are normalized to have unit integrals over
    the radius *r*.

    Parameters
    ----------
    ell: int
        The angular degree of the mode.
    cs: structured array
        The ``cs`` array for the mode.
    eig: np.array, shape(N,7)
        Eigenfunction data for the mode, as returned by
        :py:meth:`~tomso.adipls.load_amde`.
    D: 1-d array
        Global data, as defined by eq. (5.2) of the `ADIPLS
        documentation`_ and returned by
        :py:meth:`~tomso.adipls.load_amdl`.
    A: 2-d array
        Point-wise data, as defined by eq. (5.1) of the `ADIPLS
        documentation`_ and returned by
        :py:meth:`~tomso.adipls.load_amdl`.
    G: float, optional
        The gravitational constant.
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

    def integrate(yy, xx):
        """\int _x[0] ^x y(x) dx"""
        dz = (yy[1:]+yy[:-1])/2.*np.diff(xx)
        return np.hstack((0., np.cumsum(dz)))

    def complement(yy, xx):
        """\int _x ^x[-1] y(x) dx"""
        zz = integrate(yy, xx)
        return zz[-1] - zz

    M, R, P_c, rho_c = D[:4]                # mass and radius from FGONG
    y = eig.T
    sigma2 = cs['sigma2']
    omega = np.sqrt(sigma2*G*M/R**3)        # convert to angular frequency
    L2 = ell*(ell+1)

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

    if ell == 0:
        xi_h = 0.*xi_r  # radial modes have zero horizontal component
        chi = Vg/x*(y[1]-sigma2/A1/x*y[2])
        dxi_r_dr = chi - 2.*y[1]/x
        dPhi_dr = -4.*np.pi*G*rho*xi_r
        Phi = -complement(dPhi_dr, r)  # but actually you don't even need it
    elif ell > 0:
        xi_h = y[2]*R/L2
        eta = L2*A1/sigma2
        chi = Vg/x*(y[1]-y[2]/eta-y[3])
        dxi_r_dr = chi - 2.*y[1]/x + y[2]/x
        dPhi_dr = -g/x*(y[3] + y[4]) - y[3]*R*(4.*np.pi*G*rho - 2.*g/r)
        Phi = -g*R*y[3]
    else:
        raise ValueError('ell must be non-negative')

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


def fgong_to_amdl(glob, var, G=DEFAULT_G):
    """Converts FGONG data (in the form of `glob` and `var`, as returned
    by :py:meth:`~tomso.fgong.load_fgong` into ADIPLS binary data, which
    can be saved using :py:meth:`~tomso.adipls.save_amdl`.

    The output should be identical (to within a few times machine
    error) to the output of ``fgong-amdl.d`` tool distributed with
    ADIPLS.

    Parameters
    ----------
    glob: NumPy array
        The scalar (or global) variables for the stellar model
    var: NumPy array
        The point-wise variables for the stellar model. i.e. things
        that vary through the star like temperature, density, etc.
    G: float, optional
        Newton's gravitational constant in cgs units.

    Returns
    -------
    D: 1-d array
        Global data, as defined by eq. (5.2) of the `ADIPLS
        documentation`_.
    A: 2-d array
        Point-wise data, as defined by eq. (5.1) of the `ADIPLS
        documentation`_.

    """
    M, R = glob[:2]
    r, P, rho, G1, AA = var[::-1,[0,3,4,9,14]].T
    m = np.exp(var[::-1,1])*M
    N2 = G*m/r**3*AA

    ioff = (0 if r[0] < 1e6 else 1)
    nn = len(var) + ioff

    # convert profile
    A = np.zeros((nn, 6))
    A[ioff:,0] = r/R
    A[ioff:,1] = m/M/(r/R)**3
    A[ioff:,2] = G*m*rho/(G1*P*r)
    A[ioff:,3] = G1
    A[ioff:,4] = N2*r**3/(G*m)
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
    if glob[10] < 0.:
        D[4] = -glob[10]/G1[0]
        D[5] = -glob[11]
    else:
        D[4] = 4.*np.pi/3.*G*(rho[0]*R)**2/(P[0]*G1[0])
        # D[5] = np.nanmax((A[1:,4]/A[1:,0]**2)[A[1:,0]<0.05])
        # D[5] = np.max((D[5], 0.))+D[4]
        D[5] = D[4]

    D[6] = -1.
    D[7] = 0.

    if A[-1,4] <= 10.:
        # chop off outermost point
        A = A[:-1]
        nn -= 1

    return D, A


cs_dtypes = [('xmod','float'), ('M','float'), ('R','float'),
             ('P_c','float'), ('rho_c','float'), ('D_5','float'),
             ('D_6','float'), ('D_7','float'), ('D_8','float'),
             ('A_2(x_s)','float'), ('A_5(x_s)','float'),
             ('x_1','float'), ('sigma2_Omega','float'),
             ('x_f','float'), ('fctsbc','int'), ('fcttbc','int'),
             ('lambda','float'), ('ell','int'), ('enn','int'),
             ('sigma2','float'), ('sigma2_c','float'),
             ('y_1,max','float'), ('x_max', 'float'), ('E','float'),
             ('Pi_E','float'), ('Pi_V','float'), ('nu_V','float'),
             ('ddsig','float'), ('ddsol','float'),
             ('y_1(x_s)','float'), ('y_2(x_s)','float'),
             ('y_3(x_s)','float'), ('y_4(x_s)','float'),
             ('z_1,max','float'), ('xhat_max','float'),
             ('beta_nl','float'), ('nu_Ri','float'), ('emm','int')]

for i in range(len(cs_dtypes), 50):
    cs_dtypes.append(('col%i' % i, 'float'))

cs_floats = [(k, 'float') for (k,v) in cs_dtypes]
