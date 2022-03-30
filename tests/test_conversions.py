import numpy as np
from tomso.adipls import load_amdl
from tomso.fgong import load_fgong
from tomso.gyre import load_gyre
from math import frexp
import unittest

# We should be able to make various round trips between different
# format and preserve the data relevant for adiabatic oscillations.
# Conversion to AMDL might modify the data because we mimic ADIPLS's
# own FGONG-to-AMDL script.

mech_scalars = ['R', 'M', 'G']
mech_vectors = ['r', 'q', 'lnq', 'm', 'rho', 'P', 'Gamma_1', 'N2',
                'U', 'V', 'tau']

thermo_scalars = ['L', 'Teff']
thermo_vectors = ['L_r', 'T', 'Gamma_2', 'grad_r', 'grad_a']

class TestConversionFunctions(unittest.TestCase):
    def compare_models(self, m0, m1, thermo=False):
        self.assertEqual(len(m0), len(m1))

        scalars = mech_scalars + thermo_scalars if thermo else mech_scalars
        vectors = mech_vectors + thermo_vectors if thermo else mech_vectors

        for attr in scalars:
            np.testing.assert_allclose(getattr(m0, attr), getattr(m1, attr),
                                       rtol=1e-14, atol=1e-14, err_msg=attr)

        for attr in vectors:
            np.testing.assert_allclose(getattr(m0, attr), getattr(m1, attr),
                                       rtol=1e-14, atol=1e-14, err_msg=attr)

    def test_fgong_to_fgong(self):
        f = load_fgong('data/modelS.fgong')
        self.compare_models(f, f.to_gyre().to_fgong(), thermo=True)

    def test_gyre_to_gyre(self):
        g = load_gyre('data/mesa.gyre')
        self.compare_models(g, g.to_fgong().to_gyre(), thermo=True)

    def test_amdl_to_amdl(self):
        a = load_amdl('data/modelS.amdl')
        self.compare_models(a, a.to_fgong().to_amdl())
        self.compare_models(a, a.to_gyre().to_amdl())
        self.compare_models(a, a.to_fgong().to_gyre().to_amdl())
        self.compare_models(a, a.to_gyre().to_fgong().to_amdl())

if __name__ == '__main__':
    unittest.main()
