from tomso import adipls
import numpy as np
import unittest

tmpfile = 'data/tmpfile'


class TestADIPLSFunctions(unittest.TestCase):

    def test_load_mesa_amdl(self):
        nmod, nn, D, A = adipls.load_amdl('data/mesa.amdl')
        self.assertEqual(nmod, 1)
        self.assertEqual(nn, 599)
        self.assertAlmostEqual(D[0], 1.989200045e33)
        self.assertAlmostEqual(D[1], 61888348160.0)

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
        nmod, nn, D, A = adipls.load_amdl('data/modelS.amdl')
        self.assertEqual(nmod, 1)
        self.assertEqual(nn, 2482)
        self.assertAlmostEqual(D[0], 1.989e33)
        self.assertAlmostEqual(D[1], 69599062580.0)

    def test_load_modelS_rkr(self):
        css, rkrs = adipls.load_rkr('data/modelS.rkr')
        for cs in css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 69599062580.0)

    def test_save_mesa_amdl(self):
        nmod1, nn1, D1, A1 = adipls.load_amdl('data/mesa.amdl')
        adipls.save_amdl(tmpfile, nmod1, nn1, D1, A1)
        nmod2, nn2, D2, A2 = adipls.load_amdl(tmpfile)
        self.assertTrue(np.all(nmod1 == nmod2))
        self.assertTrue(np.all(nn1 == nn2))
        self.assertTrue(np.all(D1 == D2))
        self.assertTrue(np.all(A1 == A2))

    def test_save_modelS_amdl(self):
        nmod1, nn1, D1, A1 = adipls.load_amdl('data/modelS.amdl')
        adipls.save_amdl(tmpfile, nmod1, nn1, D1, A1)
        nmod2, nn2, D2, A2 = adipls.load_amdl(tmpfile)
        self.assertTrue(np.all(nmod1 == nmod2))
        self.assertTrue(np.all(nn1 == nn2))
        self.assertTrue(np.all(D1 == D2))
        self.assertTrue(np.all(A1 == A2))

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
