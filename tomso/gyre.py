"""
Functions for manipulating `GYRE`_ input and output files.

.. _GYRE: https://bitbucket.org/rhdtownsend/gyre/wiki/Home
"""

import numpy as np
from tomso.common import tomso_open


def load_summary(filename):
    """Reads a GYRE summary file and returns the global data and mode data
    in two structured arrays.  Uses builtin `gzip` module to read
    files ending with `.gz`.

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

    with tomso_open(filename, 'rb') as f:
        lines = f.readlines()

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
    data in two structured arrays.  Uses builtin `gzip` module to read
    files ending with `.gz`.

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
    with tomso_open(filename, 'rb') as f:
        lines = f.readlines()

    # catch case of no global data
    if lines[1] == '\n':
        header = []
    else:
        header = np.genfromtxt(lines[2:4], names=True)

    data = np.genfromtxt(lines[5:], names=True)

    return header, data


def load_gyre(filename):
    """Reads a GYRE stellar model file and returns the global data and
    point-wise data in a pair of NumPy record arrays.  Uses builtin
    `gzip` module to read files ending with `.gz`.

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
    with open(filename, 'wt') as f:
        header_length = len(list(header[()]))
        if header_length == 4:
            fmt = ''.join(['%6i','%26.16E'*3,'\n'])
        elif header_length == 5:
            fmt = ''.join(['%6i','%26.16E'*3,'%6i\n'])
        else:
            raise ValueError("header should have 4 or 5 components but "
                             "it appears to have %i" % header_length)
            
        f.writelines([fmt % tuple(header[()])])

        N = len(data[0])-1
        fmt = ''.join(['%6i',' %26.16E'*N,'\n'])
        for row in data:
            f.writelines([fmt % tuple(row)])


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
