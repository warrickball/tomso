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
