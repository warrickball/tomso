from tomso import adipls, fgong
import numpy as np
import unittest

tmpfile = 'data/tmpfile'
EPS = np.finfo(float).eps

class TestADIPLSFunctions(unittest.TestCase):

    def test_load_mesa_amdl(self):
        a = adipls.load_amdl('data/mesa.amdl')
        self.assertEqual(a.nmod, 1)
        self.assertEqual(len(a.A), 601)
        self.assertAlmostEqual(a.D[0], 1.988205400E+33)
        self.assertAlmostEqual(a.D[1], 6.204550713E+10)

    def test_load_mesa_amdl_fail(self):
        np.savetxt(tmpfile, np.random.rand(100,100))

        with self.assertRaises(IOError):
            a = adipls.load_amdl(tmpfile)

    def test_amdl_get(self):
        m = adipls.load_amdl('data/mesa.amdl')
        self.assertEqual(m.M, m.D[0])
        self.assertEqual(m.R, m.D[1])

        np.testing.assert_equal(m.x, m.A[:,0])
        np.testing.assert_equal(m.Gamma_1, m.A[:,3])
        np.testing.assert_equal(m.G1, m.A[:,3])
        np.testing.assert_equal(m.G1, m.Gamma_1)
        np.testing.assert_allclose(m.cs**2, m.cs2)
        np.testing.assert_allclose(m.S2_1, m.S_1**2)
        np.testing.assert_allclose(np.maximum(m.N2, 0), m.N**2)

    def test_amdl_get_cross_check(self):
        m = adipls.load_amdl('data/mesa.amdl')
        np.testing.assert_allclose(m.q, m.m/m.M, rtol=4*EPS)
        np.testing.assert_allclose(m.x, m.r/m.R, rtol=4*EPS)
        np.testing.assert_allclose(m.Hp, m.P/(m.rho*m.g), rtol=4*EPS, equal_nan=True)
        np.testing.assert_allclose(m.cs2, m.Gamma_1*m.P/m.rho, rtol=4*EPS)

        s = '%r' % m

    def test_load_modelS_agsm(self):
        agsm = adipls.load_agsm('data/modelS.agsm')
        self.assertEqual(len(agsm), 3)
        for cs in agsm.css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 6.959894677e10)

        s = '%s' % agsm
        s = '%r' % agsm

        self.assertAlmostEqual(agsm.M, 1.989e33)
        self.assertAlmostEqual(agsm.R, 6.959894677e10)
        np.testing.assert_allclose(agsm.nu_V, 1/agsm.Pi_V)
        np.testing.assert_allclose(agsm.nu_E, 1/agsm.Pi_E)

        for i in range(len(agsm.css)):
            self.assertEqual(i, agsm.index_ln(agsm.l[i], agsm.n[i]))
            self.assertEqual(i, agsm.index_nl(agsm.n[i], agsm.l[i]))

    def test_load_modelS_amde(self):
        amde1 = adipls.load_amde('data/modelS_nfmode1.amde')
        amde2 = adipls.load_amde('data/modelS_nfmode2.amde', nfmode=2)
        amde3 = adipls.load_amde('data/modelS_nfmode3.amde', nfmode=3)

        np.testing.assert_equal(amde1.eigs[:,:,1:3], amde2.eigs)
        np.testing.assert_equal(amde1.x, amde2.x)
        np.testing.assert_equal(amde1.x, amde3.x)
        np.testing.assert_equal(amde1.css, amde2.css)
        np.testing.assert_equal(amde1.css, amde3.css)
        np.testing.assert_equal(amde1.eigs[0], amde1.eig_ln(amde1.l[0], amde1.n[0]))
        np.testing.assert_equal(amde1.eigs[0], amde1.eig_nl(amde1.n[0], amde1.l[0]))

        s = '%s' % amde1
        s = '%r' % amde2

    def test_load_amde_nfmode_value_error(self):
        self.assertRaises(ValueError, adipls.load_amde, 'data/modelS_nfmode1.amde', nfmode=4)

    def test_load_modelS_amdl(self):
        a = adipls.load_amdl('data/modelS.amdl')
        self.assertEqual(a.nmod, 1)
        self.assertEqual(len(a.A), 2482)
        self.assertAlmostEqual(a.D[0], 1.989e33)
        self.assertAlmostEqual(a.D[1], 6.959894677e10)

    def test_load_modelS_rkr(self):
        rkr = adipls.load_rkr('data/modelS.rkr')

        for cs in rkr.css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 6.959894677e10)

        np.testing.assert_equal(rkr.K[0], rkr.K_ln(rkr.l[0], rkr.n[0]))
        np.testing.assert_equal(rkr.K[0], rkr.K_nl(rkr.n[0], rkr.l[0]))

        s = '%s' % rkr
        s = '%r' % rkr

    def test_save_mesa_amdl(self):
        a1 = adipls.load_amdl('data/mesa.amdl')
        a1.to_file(tmpfile)
        a2 = adipls.load_amdl(tmpfile)
        np.testing.assert_equal(a1.nmod, a2.nmod)
        np.testing.assert_equal(a1.nn, a2.nn)
        np.testing.assert_equal(a1.D, a2.D)
        np.testing.assert_equal(a1.A, a2.A)

    def test_save_modelS_amdl(self):
        a1 = adipls.load_amdl('data/modelS.amdl')
        a1.to_file(tmpfile)
        a2 = adipls.load_amdl(tmpfile)
        np.testing.assert_equal(a1.nmod, a2.nmod)
        np.testing.assert_equal(a1.nn, a2.nn)
        np.testing.assert_equal(a1.D, a1.D)
        np.testing.assert_equal(a1.A, a1.A)

    def test_fgong_to_amdl_modelS(self):
        a1 = adipls.load_amdl('data/modelS.amdl', G=6.67232e-8)
        m2 = fgong.load_fgong('data/modelS.fgong', G=6.67232e-8)
        a2 = m2.to_amdl()

        np.testing.assert_allclose(a1.D, a2.D)
        np.testing.assert_allclose(a1.A, a2.A)

    def test_fgong_to_amdl_mesa(self):
        a1 = adipls.load_amdl('data/mesa.amdl', G=6.67428e-8)
        f = fgong.load_fgong('data/mesa.fgong', G=6.67428e-8)
        a2 = f.to_amdl()

        np.testing.assert_allclose(a1.D, a2.D)
        np.testing.assert_allclose(a1.A, a2.A)

    def test_amdl_to_fgong_modelS(self):
        I = [0,3,4,9,14] # only these columns can be restored from AMDL format

        m1 = fgong.load_fgong('data/modelS.fgong', G=6.67232e-8)
        a2 = adipls.load_amdl('data/modelS.amdl', G=6.67232e-8)
        m2 = a2.to_fgong()

        np.testing.assert_allclose(m1.glob[:2], m2.glob[:2])
        np.testing.assert_allclose(m1.var[:-1,I], m2.var[:-1,I])

    def test_amdl_to_fgong_mesa(self):
        I = [0,3,4,9,14] # only these columns can be restored from AMDL format

        m1 = fgong.load_fgong('data/mesa.fgong', G=6.67428e-8)
        a2 = adipls.load_amdl('data/mesa.amdl', G=6.67428e-8)
        m2 = a2.to_fgong()

        np.testing.assert_allclose(m1.glob[:2], m2.glob[:2])
        np.testing.assert_allclose(m1.var[:,I], m2.var[:,I])

    def test_cross_check_css(self):
        agsm = adipls.load_agsm('data/modelS.agsm')
        amde = adipls.load_amde('data/modelS_nfmode1.amde')
        rkr = adipls.load_rkr('data/modelS.rkr')
        np.testing.assert_equal(agsm.css, amde.css)
        np.testing.assert_equal(agsm.css, rkr.css)

if __name__ == '__main__':
    unittest.main()
