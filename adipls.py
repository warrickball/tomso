"""
Functions for reading ADIPLS binary output.
"""

import struct
import numpy as np


def parse_one_cs(f):
    N = 50  # number of elements in each cs array
    fmt = '<' + N*'d'
    size = struct.calcsize(fmt)
    dtypes = [('xmod','float'), ('M','float'), ('R','float'),
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

    for i in range(len(dtypes), N): dtypes.append(('col%i' % i, 'float'))

    return np.array(struct.unpack(fmt, f.read(size)), dtype=dtypes)


def load_agsm(filename):
    """Reads an ADIPLS grand summary file."

    Parameters
    ----------
    filename: str
        Name of the grand summary file, usually starting or ending with agsm.


    Returns
    -------
    css: structured array
        The cs arrays for each mode.
    """

    css = []

    with open(filename, 'rb') as f:
        while True:
            if not f.read(4): break  # read returns false at EOF
            css.append(parse_one_cs(f))
            f.read(4)

    return np.squeeze(np.vstack(css))


def load_amde(filename):
    """Reads an ADIPLS eigenfunction file, written with nmode=1.

    Parameters
    ----------
    filename: str
        Name of the eigenfunction file, usually starting or ending with amde


    Returns
    -------
    css: structured array
        The cs arrays for each mode.
    eigs: list of arrays
        The eigenfunction arrays for each mode.
    """

    css = []
    eigs = []

    with open(filename, 'rb') as f:
        while True:
            if not f.read(4): break  # read returns false at EOF

            css.append(parse_one_cs(f))

            fmt = '<i'
            size = struct.calcsize(fmt)
            nnw = int(struct.unpack(fmt, f.read(size))[0])

            fmt = '<' + 7*nnw*'d'
            size = struct.calcsize(fmt)
            eig = np.array(struct.unpack(fmt, f.read(size))).reshape((-1,7))
            eigs.append(eig)

            f.read(4)

    return np.squeeze(np.vstack(css)), eigs


def load_amdl(amdl):
    """Reads an ADIPLS model file.

    Parameters
    ----------
    amdl: str
        Name of the model file, usually starting or ending with amdl


    Returns
    -------
        nmod: int
        nn: int
        D: 1D array
        A: 2D array
    """

    with open(amdl, 'rb') as f:
        f.read(4)

        fmt = '<i'
        size = struct.calcsize(fmt)
        nmod = int(struct.unpack(fmt, f.read(size))[0])

        nn = int(struct.unpack(fmt, f.read(size))[0])

        fmt = '<8d'
        size = struct.calcsize(fmt)
        D = np.array(struct.unpack(fmt, f.read(size)))

        fmt = '<' + 6*nn*'d'
        size = struct.calcsize(fmt)
        A = np.array(struct.unpack(fmt, f.read(size))).reshape((-1,6))

    return nmod, nn, D, A


def load_rkr(filename):
    """Reads an ADIPLS rotational kernel file.

    Parameters
    ----------
    filename: str
        Name of the kernel file, usually starting or ending with rkr


    Returns
    -------
    css: structured array
        The cs arrays for each mode.
    rkrs: list of arrays
        The kernel arrays for each mode.
    """

    css = []
    rkrs = []

    with open(filename, 'rb') as f:
        while True:
            if not f.read(4): break  # read returns false at EOF

            css.append(parse_one_cs(f))

            fmt = '<i'
            size = struct.calcsize(fmt)
            nnw = int(struct.unpack(fmt, f.read(size))[0])

            fmt = '<' + 2*nnw*'d'
            size = struct.calcsize(fmt)
            rkr = np.array(struct.unpack(fmt, f.read(size))).reshape((-1,2))
            rkrs.append(rkr)

            f.read(4)

    return np.squeeze(np.vstack(css)), rkrs
