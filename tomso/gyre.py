"""
Functions for manipulating `GYRE`_ input and output files.

.. _GYRE: https://bitbucket.org/rhdtownsend/gyre/wiki/Home
"""

import numpy as np


def load_summary(filename):
    """Reads a GYRE summary file and returns the global data and mode data
    in two structured arrays.

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

    with open(filename, 'r') as f:
        lines = [line.encode('utf-8') for line in f.readlines()]

    # catch case of no global data
    # just try to load header and give up on failure
    try:
        header = np.genfromtxt(lines[2:4], names=True)
    except IndexError:
        header = None
        
    data = np.genfromtxt(lines[5:], names=True)

    return header, data


def load_mode(filename):
    """Reads a GYRE mode file and returns the global data and mode profile
    data in two structured arrays.

    Parameters
    ----------
    filename: str
        Filename of the GYRE mode file to load.

    Returns
    -------
    header: structured array
        Global data for the frequency calculation. e.g. initial parameters.
        The keys for the array are the GYRE variable names as in
        the &output namelist in the GYRE input file.

    data: structured array
        Mode data for the frequency calculation. e.g. mode frequencies.
        The keys for the array are the GYRE variable names as in
        the &output namelist in the GYRE input file.

    """
    with open(filename, 'r') as f:
        lines = [line.encode('utf-8') for line in f.readlines()]

    # catch case of no global data
    if lines[1] == '\n':
        header = []
    else:
        header = np.genfromtxt(lines[2:4], names=True)

    data = np.genfromtxt(lines[5:], names=True)

    return header, data


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
    returned by :py:meth:`~tomso.gyre.load_gyre`), saves the data to a
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
