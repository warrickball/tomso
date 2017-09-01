"""
Functions for reading and writing ADIPLS binary output.
"""
import numpy as np


def read_one_cs(f):
    """Utility function to parse one `cs` array from a binary file."""
    cs = np.fromfile(f, dtype=cs_floats, count=1)
    cs = cs.astype(cs_dtypes, copy=False)
    return cs


def load_pointwise_data(filename, ncols):
    """Utility function for common structure of ADIPLS data that has a
    value at each point in a stellar model. e.g. eigenfunction and
    kernel files.
    """
    css = []
    eigs = []

    with open(filename, 'rb') as f:
        while True:
            if not f.read(4): break
            css.append(read_one_cs(f))
            nnw = np.fromfile(f, dtype='i', count=1)[0]
            eig = np.fromfile(f, dtype='d', count=ncols*nnw).reshape((-1, ncols))
            eigs.append(eig)
            f.read(4)

    return np.squeeze(css), np.squeeze(eigs)


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

    return load_pointwise_data(filename, 7)


def load_amdl(filename):
    """Reads an ADIPLS model file.  See Section 5 of the ADIPLS
    documentation for details.

    Parameters
    ----------
    filename: str
        Name of the model file, usually starting or ending with amdl.

    Returns
    -------
    nmod: int
        The model number.  I'm not sure what it's used for but it
        doesn't seem to matter.

    nn: int
        The number of points in the model.

    D: 1-d array
        Global data, as defined by eq. (5.2) of the ADIPLS
        documentation.

    A: 2-d array
        Point-wise data, as defined by eq. (5.1) of the ADIPLS
        documentation.

    """

    with open(filename, 'rb') as f:
        f.read(4)
        nmod = np.fromfile(f, dtype='i', count=1)[0]
        nn = np.fromfile(f, dtype='i', count=1)[0]
        D = np.fromfile(f, dtype='d', count=8)
        A = np.fromfile(f, dtype='d', count=6*nn).reshape((-1,6))
        f.read(4)
        # check that this is the end of the file

    return int(nmod), int(nn), D, A


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

    return load_pointwise_data(filename, 2)


def save_amdl(filename, nmod, nn, D, A):
    """Writes an ADIPLS model file, given data in the same form as
    returned by load_amdl.  See Section 5 of the ADIPLS documentation
    for details.

    Parameters
    ----------
    filename: str
        Name of the model file, usually starting or ending with amdl.

    nmod: int
        The model number.  I'm not sure what it's used for but it
        doesn't seem to matter.

    nn: int
        The number of points in the model.

    D: 1-d array
        Global data, as defined by eq. (5.2) of the ADIPLS
        documentation.

    A: 2-d array
        Point-wise data, as defined by eq. (5.1) of the ADIPLS
        documentation.

    """
    length = np.array(8*(1+8+6*nn), dtype=np.int32)
    with open(filename, 'wb') as f:
        length.tofile(f)
        np.array((nmod,), dtype=np.int32).tofile(f)
        np.array((nn,), dtype=np.int32).tofile(f)
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
