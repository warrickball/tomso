"""
Functions for manipulating `MESA`_ input and output files.

.. _MESA: http://mesa.sourceforge.net/
"""

import numpy as np
import warnings
from tomso.common import tomso_open, load_mesa_gyre


def load_history(filename, prune=False, return_object=False):
    """Reads a MESA history file and returns the global data and history
    data in two structured arrays.  Uses builtin `gzip` module to read
    files ending with `.gz`.

    If `return_object` is `True`, instead returns an `MESALog` object.
    This will become default behaviour from v0.0.12.  The old
    behaviour will be dropped completely from v0.1.0.

    Parameters
    ----------
    filename: str
        Filename of the MESA history file to load.
    prune: bool, optional
        If `True`, make the model number monotonic by only using the
        last model of with any given model number and restrict models
        to those with model number less than that of the last model.
        Useful for removing apparent reversals in time or model number
        because of backups and retries, and for models that finished
        with fewer models following a restart.

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

    header, data = load_mesa_gyre(filename, 'mesa')
    if prune:
        data = data[data['model_number'] <= data['model_number'][-1]]
        I = np.unique(data['model_number'][::-1], return_index=True)[1][::-1]
        data = data[len(data) - I - 1][::-1]

    if return_object:
        return MESALog(header, data)
    else:
        warnings.warn("From tomso 0.1.0+, `mesa.load_history` will only "
                      "return a `MESALog` object: use `return_object=True` "
                      "to mimic future behaviour",
                      FutureWarning)
        return header, data


def load_profile(filename, return_object=False):
    """Reads a MESA profile and returns the global data and profile
    data in two structured arrays.  Uses builtin `gzip` module to read
    files ending with `.gz`.

    If `return_object` is `True`, instead returns an `MESALog` object.
    This will become default behaviour from v0.0.12.  The old
    behaviour will be dropped completely from v0.1.0.

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

    header, data = load_mesa_gyre(filename, 'mesa')
    if return_object:
        return MESALog(header, data)
    else:
        warnings.warn("From tomso 0.1.0+, `mesa.load_profile` will only "
                      "return a `MESALog` object: use `return_object=True` "
                      "to mimic future behaviour",
                      FutureWarning)
        return header, data


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


class MESALog(object):
    """A dict-like class that contains the data for a MESA history or
    profile.  Variables in the header or the body can be accessed by
    the appropriate key. e.g. ``MESALog['star_age']`` returns the
    `star_age` column.

    This class also converts from (and to) logarithmic data if it is
    (not) stored in that form. e.g. if a history contains ``log_dt``,
    you can still access ``dt`` with ``MESALog['dt']``.

    This object will normally be instantiated using
    :py:meth:`mesa.load_history` or :py:meth:`mesa.load_profile`.

    Parameters
    ----------
    header: structured array
        Header data for the MESA history or profile. i.e. data for
        which there is only one value in the file.
    data: structured array
        Columned data for the history or profile. i.e. data for which
        there are multiple values (one per timestep or mesh point).

    """
    def __init__(self, header, data):
        self.header = header
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, str):
            for source in [self.data, self.header]:
                names = source.dtype.names
                if key in names:
                    return source[key]
                elif ('log_' + key) in names:
                    return 10.**source['log_' + key]
                elif ('log' + key) in names:
                    return 10.**source['log' + key]
                elif key.startswith('log_') and key[4:] in names:
                    return np.log10(source[key[4:]])
                elif key.startswith('log') and key[3:] in names:
                    return np.log10(source[key[3:]])
                else:
                    raise KeyError
        else:
            # assume we're trying to slice the data array
            return MESALog(self.header, self.data[key])
