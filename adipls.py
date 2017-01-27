"""
Functions for reading ADIPLS binary output.
"""

import struct
import numpy as np


def read_one_cs(f):
    cs = np.fromfile(f, dtype=cs_floats, count=1)
    cs = cs.astype(cs_dtypes, copy=False)
    return cs


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
            if not f.read(4): break
            css.append(read_one_cs(f))
            f.read(4) 

    return np.squeeze(css)


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
            if not f.read(4): break
            css.append(read_one_cs(f))
            nnw = np.fromfile(f, dtype='i', count=1)[0]
            eig = np.fromfile(f, dtype='d', count=7*nnw).reshape((-1,7))
            eigs.append(eig)
            f.read(4)

    return np.squeeze(css), np.squeeze(eigs)


def load_amdl(filename):
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

    with open(filename, 'rb') as f:
        f.read(4)
        nmod = np.fromfile(f, dtype='i', count=1)[0]
        nn = np.fromfile(f, dtype='i', count=1)[0]
        D = np.fromfile(f, dtype='d', count=8)
        A = np.fromfile(f, dtype='d', count=6*nn).reshape((-1,6))
        f.read(4)
        # check that this is the end of the file

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
            if not np.fromfile(f, dtype='i', count=1): break
            css.append(read_one_cs(f))
            nnw = np.fromfile(f, dtype='i', count=1)[0]
            rkr = np.fromfile(f, dtype='d', count=2*nnw).reshape((-1,2))
            rkrs.append(rkr)
            f.read(4)

    return np.squeeze(css), np.squeeze(rkrs)


def save_amdl(filename, nmod, nn, D, A):
    length = np.array(8*(nmod+nn+8+6*nn), dtype=np.int32)
    with open(filename, 'wb') as f:
        length.tofile(f)
        nmod.tofile(f)
        nn.tofile(f)
        D.tofile(f)
        A.tofile(f)
        length.tofile(f)


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
