"""Physical constants."""
import numpy as np

k_B = 1.380649e-16
"""Exact value of the Boltzmann constant in cgs units."""

h = 6.62607015e-27
"""Exact value of Planck's constant in cgs units."""

c_light = 2.99792458e10
"""Exact value of the speed of light in cgs units."""

sigma_SB = 2.*np.pi**5*k_B**4/(15.*h**3*c_light**2)
"""Exact value of the Stefan-Boltzmann constant in cgs units."""

# IAU B3
GMsun = 1.3271244e26
"""Solar standard gravitational parameter in cgs units (IAU B3)."""

Rsun = 695700e5
"""Solar radius in cgs units (IAU B3)."""

Lsun = 3.828e33
"""Solar luminosity in cgs units (IAU B3)."""

Teff_sun = (Lsun/(4.*np.pi*Rsun**2*sigma_SB))**0.25
"""Solar effective temperature in K (IAU B3)."""

# G (uncertainty)
G_CODATA_2018 = 6.67430e-8 # (15) # MESA since r12934
G_CODATA_2014 = 6.67408e-8 # (31)
G_CODATA_2010 = 6.67384e-8 # (80)
G_CODATA_2006 = 6.67428e-8 # (67) # MESA until r12934
G_CODATA_2002 = 6.6742e-8  # (10)
G_CODATA_1998 = 6.673e-8   # (10)
G_CODATA_1986 = 6.67259e-8 # (85)
G_CODATA_1973 = 6.6720e-8  # (41)
G_MODEL_S = 6.67232e-8
G_DEFAULT = G_CODATA_2018

Msun = GMsun/G_DEFAULT
