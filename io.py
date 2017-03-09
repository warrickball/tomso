"""
Functions for general I/O, not specific to a particular code.
"""
import numpy as np


def load_fgong(filename, N=-1):
    """Given an FGONG file, returns a Python dictionary containing
    NumPy arrays that correspond to the structures in the
    specification of the FGONG format:

    https://www.astro.up.pt/corot/ntools/docs/CoRoT_ESTA_Files.pdf

    That is, the dictionary has arrays indexed by 'nn', 'iconst',
    'ivar', 'ivers', 'glob' and 'var'.

    Parameters
    ----------
    filename: str
        Name of the FGONG file to read.
    N: integer, optional
        Number of characters in each float.  If negative, the function
        tries to guess the size of each float. (default: -1)

    Returns
    -------
    fgong: dict
        Dictionary with scalars and arrays for FGONG data.  The keys are

        * 'nn': integer, number of points in model
        * 'iconst': integer, number of scalar values in global parameters ('glob')
        * 'ivar': integer, number of variables in model profile ('var')
        * 'ivers': integer, version number (not used in calculation)
        * 'glob': float array, shape=(iconst,); scalar parameters for stellar model (e.g. mass, radius)
        * 'var': float array, shape=(nn, ivar); stellar model profile

    """
    f = open(filename, 'r')

    fgong = {'header':[]}
    for i in range(4):
        fgong['header'].append(f.readline())

    tmp = [int(i) for i in f.readline().split()]
    fgong['nn'] = tmp[0]
    fgong['iconst'] = tmp[1]
    fgong['ivar'] = tmp[2]
    fgong['ivers'] = tmp[3]

    lines = f.readlines()
    tmp = []

    # try to guess the length of each float in the data
    if N < 0: N = len(lines[0])/5

    for line in lines:
        for i in range(len(line)/N):
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

    fgong['glob'] = np.array(tmp[:fgong['iconst']])
    fgong['var'] = np.array(tmp[fgong['iconst']:]).reshape((-1,fgong['ivar']))

    f.close()

    return fgong


def save_fgong(filename, fgong, fmt='%16.9E'):
    """Given data for an FGONG file in the format returned by
    `load_fgong`, writes the data to a file.

    Parameters
    ----------
    filename: str
        Filename to which FGONG data is written.
    fgong: dict
        FGONG data, formatted as in `load_fgong`.
    fmt: str
        Format of the floating-point data (scalars and model profile).
    """
    with open(filename, 'w') as f:
        f.writelines(fgong['header'])

        line = '%10i'*4 % (fgong['nn'], fgong['iconst'],
                           fgong['ivar'], fgong['ivers'])
        f.writelines([line + '\n'])

        for i in range(0, fgong['iconst'], 5):
            N = np.mod(i+4, 5)+1  # number of floats in this row
            line = fmt*N % tuple(fgong['glob'][i:i+5])
            f.writelines([line + '\n'])

        for row in fgong['var']:
            for i in range(0, fgong['ivar'], 5):
                N = np.mod(i+4, 5)+1  # number of floats in this row
                line = fmt*N % tuple(row[i:i+5])
                f.writelines([line + '\n'])


def load_gyre(filename):
    with open(filename, 'r') as f:
        lines = [line.replace('D','E') for line in f.readlines()]

    header = np.genfromtxt(lines[:1], dtype=gyre_header_dtypes)
    data = np.genfromtxt(lines[1:], dtype=gyre_data_dtypes)

    return header, data


def save_gyre(filename, header, data):
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
