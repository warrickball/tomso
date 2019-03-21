"""
Functions for manipulating `MESA`_ input and output files.

.. _MESA: http://mesa.sourceforge.net/
"""

import numpy as np
from tomso.common import tomso_open, load_mesa_gyre


def load_history(filename):
    """Reads a MESA history file and returns the global data and history
    data in two structured arrays.  Uses builtin `gzip` module to read
    files ending with `.gz`.

    Parameters
    ----------
    filename: str
        Filename of the MESA history file to load.

    Returns
    -------
    header: structured array
        Global data for the evolutionary run. e.g. initial parameters.
        The keys for the array are the MESA variable names as in
        `history.columns`.

    data: structured array
        History data for the run. e.g. age, effective temperature.
        The keys for the array are the MESA variable names as in
        `history.columns`.
    """
    
    return load_mesa_gyre(filename, 'mesa')
    

def load_profile(filename):
    """Reads a MESA profile and returns the global data and profile
    data in two structured arrays.  Uses builtin `gzip` module to read
    files ending with `.gz`.

    Parameters
    ----------
    filename: str
        Filename of the MESA profile to load.

    Returns
    -------
    header: structured array
        Global data for the stellar model. e.g. total mass, luminosity.
        The keys for the array are the MESA variable names as in
        `profile.columns`.

    data: structured array
        Profile data for the stellar model. e.g. radius, pressure.
        The keys for the array are the MESA variable names as in
        `profile.columns`.
    """

    return load_mesa_gyre(filename, 'mesa')


def load_astero_results(filename):
    """Reads a set of MESA results from one of the optimization routines
    in the `astero` module.

    Parameters
    ----------
    filename: str
        Filename of the file containing the results.

    Returns
    -------
    data: structured array
        Array with all the results.
    """
    with tomso_open(filename, 'rb') as f:
        lines = [line.replace(b'D', b'E') for line in f.readlines()]

    # the last column results for `search_type = simplex` fits have a
    # nameless column that says what kind of simplex step was taken.
    # we have to give it a name ourselves
    names = [name.decode('utf-8') for name in lines[1].split()]
    N_columns = len(lines[2].split())
    if len(names) == N_columns - 1:
        names.append('step_type')
        
    data = np.genfromtxt(lines[2:-4], dtype=None, names=names,
                         encoding='utf-8')

    return data


def load_sample(filename):
    """Reads a MESA sample file that describes a model from one of the
    optimization routines in the `astero` module.

    Parameters
    ----------
    filename: str
        Filename of the file containing the result.

    Returns
    -------
    d: dict
        A dictionary containing all the results.

    """
    with tomso_open(filename, 'rb') as f:
        # lines = [line.split() for line in f.read().decode('utf-8').split('\n')
        #          if line.strip()]
        lines = [line.decode('utf-8').split() for line in f.readlines() if line.strip()]

    table_dtype = [('n', int), ('chi2term', float), ('freq', float), ('corr', float),
                   ('obs', float), ('sigma', float), ('logE', float)]
    d = {'l%i' % ell: np.zeros(0, dtype=table_dtype) for ell in range(4)}
    # d = {'l0': np.zeros(0, dtype=table_dtype),
    #      'l1': np.zeros(0, dtype=table_dtype),
    #      'l2': np.zeros(0, dtype=table_dtype),
    #      'l3': np.zeros(0, dtype=table_dtype)}
    ell = 0

    for line in lines:
        if line[0][:2] == 'l=':
            ell = int(line[0][-1])
        elif len(line) == 7:
            # I'm not quite sure why this hideous construction is
            # necessary but it seems that the recarray construction
            # depends on whether it gets a tuple or a list
            row = np.array(tuple([int(line[0])] + list(map(float, line[1:]))),
                           dtype=table_dtype)
            d['l%i' % ell] = np.append(d['l%i' % ell], row)
            # d['l%i' % ell]['n'].append(int(line[0]))
            # d['l%i' % ell]['chi2'].append(float(line[1]))
            # d['l%i' % ell]['mdl'].append(float(line[2]))
            # d['l%i' % ell]['cor'].append(float(line[3]))
            # d['l%i' % ell]['obs'].append(float(line[4]))
            # d['l%i' % ell]['err'].append(float(line[5]))
            # d['l%i' % ell]['logE'].append(float(line[6]))
        else:
            key = ''.join([word + ' ' for word in line[:-1]])[:-1]
            # if key == 'mass/Msun':
            #     key = 'initial mass'

            value = float(line[-1].lower().replace('d', 'e'))
            d[key] = value

    return d


# update_inlist, string_where and replace_value all ported from
# mesaface.  still need testing!
def update_inlist(inlist, d):
    """Updates parameter values in a MESA inlist file.  The function
    searches the whole file for the parameter key.  An ``IndexError``
    usually means that one of the keys in dict `d` wasn't found in
    `inlist`.

    Parameters
    ----------
    inlist: str
        Filename of the inlist file that will be updated.
    d: dict
        Dictionary containing the parameter names and their new
        values. e.g. `{'initial_mass': 1.0}` or
        `{'use_Ledoux_criterion': True}`.

    """
    with open(inlist, 'r') as f: lines = f.readlines()

    # don't search comments
    search_lines = [line.split('!', 1)[0] for line in lines]

    for key, value in d.items():
        i = string_where(search_lines, key)[0]
        lines[i] = replace_value(lines[i], value)

    with open(inlist, 'wt') as f:
        f.writelines(lines)


def string_where(lines, expr):
    "Returns list of indices of the lines in `lines` containing `expr`."
    return [i for i in range(len(lines)) if expr in lines[i].split()]


def replace_value(line, value):
    """Replaces the parameter `value` in the given `line` of a MESA
    inlist.  Format is inferred from the type of value: `float`,
    `str`, `int` or `bool`.

    """
    equals = line.index('=')+1
    if type(value) == float:
        return '%s %.20e\n' % (line[:equals], value)
    elif type(value) == str:
        return '%s %s\n' % (line[:equals], value)
    elif type(value) == int:
        return '%s %i\n' % (line[:equals], value)
    elif type(value) == bool:
        if value:
            return '%s .true.\n' % line[:equals]
        else:
            return '%s .false.\n' % line[:equals]
    else:
        raise ValueError('Value in mesa.replace_value() is not a valid type!')
