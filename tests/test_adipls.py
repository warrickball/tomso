from tomso import adipls, io
import numpy as np
import unittest

tmpfile = 'data/tmpfile'


class TestADIPLSFunctions(unittest.TestCase):

    def test_load_mesa_amdl(self):
        D, A, nmod = adipls.load_amdl('data/mesa.amdl', return_nmod=True)
        self.assertEqual(nmod, 1)
        self.assertEqual(len(A), 601)
        self.assertAlmostEqual(D[0], 1.988205400E+33)
        self.assertAlmostEqual(D[1], 6.204550713E+10)

    def test_load_modelS_agsm(self):
        css = adipls.load_agsm('data/modelS.agsm')
        for cs in css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 69599062580.0)

    def test_load_modelS_amde(self):
        css, eigs = adipls.load_amde('data/modelS.amde')
        for cs in css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 69599062580.0)

    def test_load_modelS_amdl(self):
        D, A, nmod = adipls.load_amdl('data/modelS.amdl', return_nmod=True)
        self.assertEqual(nmod, 1)
        self.assertEqual(len(A), 2482)
        self.assertAlmostEqual(D[0], 1.989e33)
        self.assertAlmostEqual(D[1], 69599062580.0)

    def test_load_modelS_rkr(self):
        css, rkrs = adipls.load_rkr('data/modelS.rkr')
        for cs in css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 69599062580.0)

    def test_save_mesa_amdl(self):
        D1, A1, nmod1 = adipls.load_amdl('data/mesa.amdl', return_nmod=True)
        adipls.save_amdl(tmpfile, D1, A1, nmod=nmod1)
        D2, A2, nmod2 = adipls.load_amdl(tmpfile, return_nmod=True)
        self.assertTrue(np.all(nmod1 == nmod2))
        self.assertTrue(np.all(len(A1) == len(A2)))
        self.assertTrue(np.all(D1 == D2))
        self.assertTrue(np.all(A1 == A2))

    def test_save_modelS_amdl(self):
        D1, A1, nmod1 = adipls.load_amdl('data/modelS.amdl', return_nmod=True)
        adipls.save_amdl(tmpfile, D1, A1, nmod=nmod1)
        D2, A2, nmod2 = adipls.load_amdl(tmpfile, return_nmod=True)
        self.assertTrue(np.all(nmod1 == nmod2))
        self.assertTrue(np.all(len(A1) == len(A2)))
        self.assertTrue(np.all(D1 == D2))
        self.assertTrue(np.all(A1 == A2))

    def test_fgong_to_amdl_modelS(self):
        D1, A1 = adipls.load_amdl('data/modelS.amdl')
        glob, var = io.load_fgong('data/modelS.fgong')
        D2, A2 = adipls.fgong_to_amdl(glob, var, G=6.67232e-8)
        for (x,y) in zip(D1, D2):
            self.assertAlmostEqual(x,y)

        for i in range(len(A1)):
            for (x,y) in zip(A1[i], A2[i]):
                self.assertAlmostEqual(x,y)

    def test_fgong_to_amdl_mesa(self):
        D1, A1 = adipls.load_amdl('data/mesa.amdl')
        glob, var = io.load_fgong('data/mesa.fgong')
        D2, A2 = adipls.fgong_to_amdl(glob, var, G=6.67232e-8)
        for (x,y) in zip(D1, D2):
            self.assertAlmostEqual(x,y)

        for i in range(len(A1)):
            for (x,y) in zip(A1[i], A2[i]):
                self.assertAlmostEqual(x,y)

    def cross_check_css(self):
        css_agsm = adipls.load_agsm('data/mesa.agsm')
        css_amdl = adipls.load_agsm('data/mesa.amdl')[0]
        css_amde = adipls.load_agsm('data/mesa.amde')[0]
        css_rkr = adipls.load_agsm('data/mesa.rkr')[0]
        self.assertTrue(np.all(css_agsm == css_amdl))
        self.assertTrue(np.all(css_agsm == css_amde))
        self.assertTrue(np.all(css_agsm == css_rkr))


if __name__ == '__main__':
    unittest.main()
