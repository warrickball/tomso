"""
Functions for loading output from the `Cambridge STARS`_ code.

.. _Cambridge STARS: http://www.ast.cam.ac.uk/~stars/index.htm
"""

import numpy as np


def load_out(filename):
    """Reads a STARS `out` file and returns (part of) the summaries and
    profiles in two structured arrays.

    Parameters
    ----------
    filename: str
        Filename of the STARS output file to load.

    Returns
    -------
    summaries: 2-d structured array
        Summaries of each model in the run, similar to MESA's
        `history` files.

    profiles: 3-d structured array
        Model profiles produced at regular intervals during the run.
        The first index of the array is the profile number.
    """
    with open(filename, 'r') as f:
        line = f.readline()

        K = int(line.split()[0])  # number of points in profiles

        summaries = []
        profiles = []

        while line:
            line = f.readline()

            if len(line.split()) < 2:
                continue
            elif line.split()[0] == 'K':
                # found a profile
                dtypes = [('k','int')] + \
                         [(name, 'float') for name in line.split()[1:]]
                data = [f.readline() for i in range(K+(K-1)//10)]
                data = [tuple(map(float, row.split()))
                        for row in data if row != '\n']
                profiles.append(np.array(data, dtype=dtypes))
            elif line.split()[1] == 'dt/age/MH/MHe':
                # found a summary
                # have to do some basic manipulation to make parsing stable
                summary = [int(line.split()[0])]
                for i in range(2):
                    line = f.readline()
                    summary.extend(line.replace('-',' -').replace('E -','E-').split()[1:-1])

                summaries.append(np.array(tuple(summary),
                                          dtype=summary_dtypes[:len(summary)]))

    return np.squeeze(np.vstack(summaries)), np.vstack((profiles))


def load_plot(filename):
    """Reads a STARS `plot` file into a structured array.

    Parameters
    ----------
    filename: str
        Filename of the STARS plot file to load.

    Returns
    -------
    plot: 2-d structured array
        Data for evolutionary track, with one row per model along the
        track. e.g. age, effective temperature, etc.

    """
    return np.loadtxt(filename, dtype=plot_dtypes)


summary_dtypes = [('n','int'), ('dt', 'float'),
                  ('tn', 'float'), ('P', 'float'),
                  ('LH', 'float'), ('Lth', 'float'), ('H1_cntr','float'),
                  ('He4_cntr', 'float'), ('C12_cntr', 'float'),
                  ('N14_cntr', 'float'), ('O16_cntr','float'),
                  ('Ne20_cntr', 'float'), ('He3_cntr','float'),
                  ('psi_cntr', 'float'), ('dens_cntr','float'),
                  ('temp_cntr', 'float'), ('age','float'),
                  ('tKH', 'float'),('rlf', 'float'),
                  ('LHe','float'),('Lnu', 'float'), ('H1_srfc', 'float'),
                  ('He4_srfc', 'float'), ('C12_srfc', 'float'),
                  ('N14_srfc', 'float'), ('O16_srfc', 'float'),
                  ('Ne20_srfc', 'float'), ('He3_srfc', 'float'),
                  ('psi_srfc', 'float'), ('dens_srfc', 'float'),
                  ('temp_srfc', 'float'), ('MH', 'float'), ('MHe','float'),
                  ('Mb', 'float'), ('dM', 'float'), ('LC','float'),
                  ('m', 'float'), ('H1_Tmax', 'float'),
                  ('He4_Tmax', 'float'), ('C12_Tmax', 'float'),
                  ('N14_Tmax', 'float'), ('O16_Tmax', 'float'),
                  ('Ne20_Tmax', 'float'), ('He3_Tmax', 'float'),
                  ('psi_Tmax', 'float'), ('dens_Tmax', 'float'),
                  ('temp_Tmax', 'float')]


plot_dtypes = [('n','int'), ('age', 'float'), ('logR', 'float'),
               ('logTeff', 'float'), ('logL', 'float'), ('M', 'float'),
               ('M_H','float'), ('M_He', 'float'), ('L_H', 'float'),
               ('L_He', 'float'), ('L_C', 'float'), ('mconv1', 'float'),
               ('m_conv2','float'), ('m_conv3','float'), ('m_conv4','float'),
               ('m_conv5','float'), ('m_conv6','float'), ('m_conv7','float'),
               ('m_conv8','float'), ('m_conv9','float'), ('m_conv10','float'),
               ('m_conv11','float'), ('m_conv12','float'),
               ('m_eps_H_max', 'float'), ('m_eps_He_max', 'float'),
               ('logkappa', 'float'), ('dt', 'float'),
               ('X_Hs', 'float'), ('X_Hes', 'float'), ('H_Cs', 'float'),
               ('X_Ns','float'), ('X_Os', 'float'), ('X_He3s', 'float'),
               ('R/R_RL', 'float'), ('J_1', 'float'), ('P', 'float'),
               ('R_sep', 'float'), ('M1+M2','float'), ('J_orb', 'float'),
               ('J_1+J_2', 'float'), ('J_tot','float'), ('omega_orb', 'float'),
               ('omega_1', 'float'), ('I_1','float'), ('I_orb', 'float'),
               ('Mdot', 'float'), ('m_shell1','float'), ('m_shell2','float'),
               ('m_shell3','float'), ('m_shell4','float'),
               ('m_shell5','float'), ('m_shell6','float'),
               ('m_shell7','float'), ('m_shell8','float'),
               ('m_shell9','float'), ('m_shell10','float'),
               ('m_shell11','float'), ('m_shell12','float'),
               ('m_th1','float'), ('m_th2','float'), ('m_th3','float'),
               ('m_th4','float'), ('m_th5','float'), ('m_th6','float'),
               ('m_th7','float'), ('m_th8','float'), ('m_th9','float'),
               ('m_th10','float'), ('m_th11','float'), ('m_th12','float'),
               ('M_ce', 'float'), ('R_ce', 'float'),
               ('logRho_c', 'float'), ('logT_c', 'float')]
