"""
Functions for general I/O, not specific to a particular code.
"""

import numpy as np
from tomso.common import integrate, DEFAULT_G


def load_fgong(filename, N=-1, return_comment=False):
    """Given an FGONG file, returns NumPy arrays `glob` and `var` that
    correspond to the scalar and point-wise variables, as specified
    in the `FGONG format`_.

    .. _FGONG format: https://www.astro.up.pt/corot/ntools/docs/CoRoT_ESTA_Files.pdf

    Also returns the first four lines of the file as a `comment`, if
    desired.

    The version number `ivers` is not implemented.

    Parameters
    ----------
    filename: str
        Name of the FGONG file to read.
    N: integer, optional
        Number of characters in each float.  If negative, the function
        tries to guess the size of each float. (default: -1)
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
    f = open(filename, 'r')

    comment = [f.readline() for i in range(4)]

    nn, iconst, ivar, ivers = [int(i) for i in f.readline().split()]

    lines = f.readlines()
    tmp = []

    # try to guess the length of each float in the data
    if N < 0: N = len(lines[0])//5

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
                s.lower().replace('d','e')

            tmp.append(float(s))

    glob = np.array(tmp[:iconst])
    var = np.array(tmp[iconst:]).reshape((-1, ivar))

    f.close()

    if return_comment:
        return glob, var, comment
    else:
        return glob, var


def save_fgong(filename, glob, var, fmt='%16.9E', ivers=0,
               comment=['\n','\n','\n','\n']):
    """Given data for an FGONG file in the format returned by
    :py:meth:`~tomso.io.load_fgong` (i.e. two NumPy arrays and a
    possible header), writes the data to a file.

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

    with open(filename, 'w') as f:
        f.writelines(comment)

        line = '%10i'*4 % (nn, iconst, ivar, ivers)
        f.writelines([line + '\n'])

        for i in range(0, iconst, 5):
            N = np.mod(i+4, 5)+1  # number of floats in this row
            line = fmt*N % tuple(glob[i:i+5])
            f.writelines([line + '\n'])

        for row in var:
            for i in range(0, ivar, 5):
                N = np.mod(i+4, 5)+1  # number of floats in this row
                line = fmt*N % tuple(row[i:i+5])
                f.writelines([line + '\n'])


def fgong_get(keys, glob, var, G=DEFAULT_G):
    """Retrieves physical properties of a FGONG model from the ``glob`` and
    ``var`` arrays.

    Parameters
    ----------
    keys: list of strs
        A list of desired variables.  Current options are:

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
        - ``Hp``: pressure scale height (array)
        - ``G1``: first adiabatic index (array)
        - ``T``: temperature (array)
        - ``X``: hydrogen abundance (array)
        - ``L_r``: luminosity at radius ``r`` (array)
        - ``kappa``: opacity (array)
        - ``epsilon``: specific energy generation rate (array)
        - ``cs2``: sound speed squared (array)
        - ``cs``: sound speed (array)
        - ``tau``: acoustic depth

        For example, if ``glob`` and ``var`` have been returned from
        :py:meth:`~tomso.io.load_fgong`, you could use

        >>> M, m = io.fgong_get(['M', 'm'], glob, var)

        to get the total mass and mass co-ordinate.  If you only want one variable,
        remember that you get a length 1 list, not the single item, so use

        >>> x, = io.fgong_get(['x'], glob, var)

        rather than

        >>> x = io.fgong_get(['x'], glob, var)

    glob: NumPy array
        The scalar (or global) variables for the stellar model
    var: NumPy array
        The point-wise variables for the stellar model. i.e. things
        that vary through the star like temperature, density, etc.
    
    Returns
    -------
    output: list of floats and arrays
        A list returning the floats or arrays in the order requested
        by the parameter ``keys``.

    """
    M, R, L = glob[:3]
    r, lnq, T, P, rho, X, L_r, kappa, epsilon, G1 = var[:,:10].T
    x = r/R
    q = np.exp(lnq)
    m = q*M
    g = G*m/r**2
    Hp = P/(rho*g)
    cs2 = G1*P/rho                    # square of the sound speed
    cs = np.sqrt(cs2)
    tau = -integrate(1./cs[::-1], r[::-1])[::-1]      # acoustic depth

    output = []
    for key in keys:
        if key == 'M': output.append(M)
        elif key == 'R': output.append(R)
        elif key == 'L': output.append(L)
        elif key == 'r': output.append(r)
        elif key == 'x': output.append(x)
        elif key == 'm': output.append(m)
        elif key == 'q': output.append(q)
        elif key == 'g': output.append(g)
        elif key == 'rho': output.append(rho)
        elif key == 'P': output.append(P)
        elif key == 'Hp': output.append(Hp)
        elif key == 'G1': output.append(G1)
        elif key == 'T': output.append(T)
        elif key == 'X': output.append(X)
        elif key == 'L_r': output.append(L_r)
        elif key == 'kappa': output.append(kappa)
        elif key == 'epsilon': output.append(epsilon)
        elif key == 'cs2': output.append(cs2)
        elif key == 'cs': output.append(cs)
        elif key == 'tau': output.append(tau)
        else: raise ValueError('invalid key for adipls.amdl_get')

    return output


def load_gyre(filename):
    """Reads a GYRE stellar model file and returns the global data and
    point-wise data in a pair of NumPy record arrays.

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
    with open(filename, 'r') as f:
        lines = [line.replace('D','E') for line in f.readlines()]

    header = np.loadtxt(lines[:1], dtype=gyre_header_dtypes)
    data = np.loadtxt(lines[1:], dtype=gyre_data_dtypes)

    return header, data


def save_gyre(filename, header, data):
    """Given the global data and point-wise data for a stellar model (as
    returned by :py:meth:`~tomso.io.load_gyre`), saves the data to a
    target file in the GYRE format.

    Parameters
    ----------
    filename: str
        Filename of the GYRE file.

    header: structured array
        Global data for the stellar model. e.g. total mass, luminosity.

    data: structured array
        Profile data for the stellar model. e.g. radius, pressure.

    """
    with open(filename, 'w') as f:
        fmt = ''.join(['%6i','%26.16E'*3,'%6i\n'])
        f.writelines([fmt % tuple(header[()])])

        N = len(data[0])-1
        fmt = ''.join(['%6i',' %26.16E'*N,'\n'])
        for row in data:
            f.writelines([fmt % tuple(row)])


gyre_header_dtypes = [('n','int'), ('M','float'), ('R','float'),
                      ('L','float'), ('version','int')]
gyre_data_dtypes = [('k','int'), ('r','float'), ('m','float'),
                    ('L_r','float'), ('p','float'), ('T','float'),
                    ('rho','float'), ('nabla','float'),
                    ('N2','float'), ('Gamma_1','float'),
                    ('nabla_ad','float'), ('delta','float'),
                    ('kappa','float'), ('kappa_T','float'),
                    ('kappa_rho','float'), ('eps','float'),
                    ('eps_T','float'), ('eps_rho','float'),
                    ('omega','float')]
