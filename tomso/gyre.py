"""
Functions for manipulating GYRE input and output files.
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
