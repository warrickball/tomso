from tomso.adipls import load_amdl
from tomso.fgong import load_fgong
from tomso.gyre import load_gyre
from math import frexp
import unittest

# We should be able to make various round trips between different
# format and preserve the data relevant for adiabatic oscillations.
# Conversion to AMDL might modify the data because we mimic ADIPLS's
# own FGONG-to-AMDL script.

scalars = ['R', 'M', 'G']
vectors = ['r', 'q', 'm', 'rho', 'P', 'Gamma_1', 'N2', 'U', 'V']

class TestConversionFunctions(unittest.TestCase):
    def compare_floats(self, x, y, attr='', index=0, places=12):
        self.assertAlmostEqual(frexp(x)[0], frexp(y)[0], places=places,
                               msg='%s at k=%i' % (attr, index))

    def compare_models(self, m0, m1):
        self.assertEqual(len(m0), len(m1))

        for attr in scalars:
            self.compare_floats(getattr(m0, attr), getattr(m1, attr),
                                attr=attr)

        for attr in vectors:
            x0 = getattr(m0, attr)
            x1 = getattr(m1, attr)
            for i in range(len(m0)):
                self.compare_floats(x0[i], x1[i], attr=attr, index=i)

    def test_fgong_to_fgong(self):
        f = load_fgong('data/modelS.fgong', return_object=True)
        self.compare_models(f, f.to_gyre().to_fgong())

    def test_gyre_to_gyre(self):
        g = load_gyre('data/mesa.gyre', return_object=True)
        self.compare_models(g, g.to_fgong().to_gyre())

    def test_amdl_to_amdl(self):
        a = load_amdl('data/modelS.amdl', return_object=True)
        self.compare_models(a, a.to_fgong().to_amdl())
        self.compare_models(a, a.to_gyre().to_amdl())
        self.compare_models(a, a.to_fgong().to_gyre().to_amdl())
        self.compare_models(a, a.to_gyre().to_fgong().to_amdl())

if __name__ == '__main__':
    unittest.main()
