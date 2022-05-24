#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pl
from tomso import fgong

S = fgong.load_fgong('https://users-phys.au.dk/jcd/solar_models/fgong.l5bi.d.15c', G=6.67232e-8)
pl.plot(S.tau, np.gradient(S.cs, S.tau)/1e4)
pl.xlabel("τ (sec)")
pl.ylabel("dc/dτ (10⁴ cm/s²)")
pl.axis([100., 3000., 0., 2.5])
pl.savefig('modelS_dcs_dtau.png')
