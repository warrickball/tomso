from tomso import adipls, fgong
import numpy as np
import unittest

tmpfile = 'data/tmpfile'
EPS = np.finfo(float).eps

class TestADIPLSFunctions(unittest.TestCase):

    def test_load_mesa_amdl(self):
        D, A, nmod = adipls.load_amdl('data/mesa.amdl', return_nmod=True,
                                      return_object=False)
        self.assertEqual(nmod, 1)
        self.assertEqual(len(A), 601)
        self.assertAlmostEqual(D[0], 1.988205400E+33)
        self.assertAlmostEqual(D[1], 6.204550713E+10)

        a = adipls.load_amdl('data/mesa.amdl', return_object=True)
        self.assertEqual(a.nmod, 1)
        self.assertEqual(len(a.A), 601)
        self.assertAlmostEqual(a.D[0], 1.988205400E+33)
        self.assertAlmostEqual(a.D[1], 6.204550713E+10)

    def test_load_mesa_amdl_fail(self):
        np.savetxt(tmpfile, np.random.rand(100,100))

        with self.assertRaises(IOError):
            D, A = adipls.load_amdl(tmpfile, return_object=False)

        with self.assertRaises(IOError):
            a = adipls.load_amdl(tmpfile, return_object=True)

    def test_amdl_get(self):
        D, A = adipls.load_amdl('data/mesa.amdl', return_object=False)
        M, R, x, G1 = adipls.amdl_get(['M','R','x','G1'], D, A)
        self.assertEqual(M, D[0])
        self.assertEqual(R, D[1])

        np.testing.assert_equal(x, A[:,0])
        np.testing.assert_equal(G1, A[:,3])

        cs, = adipls.amdl_get(['cs'], D, A)
        cs2 = adipls.amdl_get('cs2', D, A)
        np.testing.assert_allclose(cs**2, cs2)

        m = adipls.load_amdl('data/mesa.amdl', return_object=True)
        self.assertEqual(m.M, m.D[0])
        self.assertEqual(m.R, m.D[1])

        np.testing.assert_equal(m.x, m.A[:,0])
        np.testing.assert_equal(m.Gamma_1, m.A[:,3])
        np.testing.assert_equal(m.G1, m.A[:,3])
        np.testing.assert_equal(m.G1, m.Gamma_1)
        np.testing.assert_allclose(m.cs**2, m.cs2)

    def test_amdl_get_cross_check(self):
        D, A = adipls.load_amdl('data/mesa.amdl', return_object=False)
        M, R, P_c, rho_c, r, x, m, q, g, rho, P, Hp, G1, cs2, cs, tau \
            = adipls.amdl_get(['M','R', 'P_c', 'rho_c', 'r', 'x', 'm',
                                'q', 'g', 'rho', 'P', 'Hp', 'G1', 'cs2',
                                'cs', 'tau'], D, A)
        np.testing.assert_allclose(q, m/M, rtol=4*EPS)
        np.testing.assert_allclose(x, r/R, rtol=4*EPS)
        np.testing.assert_allclose(Hp, P/(rho*g), rtol=4*EPS, equal_nan=True)
        np.testing.assert_allclose(cs2, G1*P/rho, rtol=4*EPS)

        m = adipls.load_amdl('data/mesa.amdl', return_object=True)
        np.testing.assert_allclose(m.q, m.m/m.M, rtol=4*EPS)
        np.testing.assert_allclose(m.x, m.r/m.R, rtol=4*EPS)
        np.testing.assert_allclose(m.Hp, m.P/(m.rho*m.g), rtol=4*EPS, equal_nan=True)
        np.testing.assert_allclose(m.cs2, m.Gamma_1*m.P/m.rho, rtol=4*EPS)

    def test_load_modelS_agsm(self):
        css = adipls.load_agsm('data/modelS.agsm', return_object=False)
        for cs in css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 6.959894677e10)

        agsm = adipls.load_agsm('data/modelS.agsm', return_object=True)
        for cs in agsm.css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 6.959894677e10)

        self.assertAlmostEqual(agsm.M, 1.989e33)
        self.assertAlmostEqual(agsm.R, 6.959894677e10)
        np.testing.assert_allclose(agsm.nu_V, 1/agsm.Pi_V)
        np.testing.assert_allclose(agsm.nu_E, 1/agsm.Pi_E)

        for i in range(len(agsm.css)):
            self.assertEqual(i, agsm.index_ln(agsm.l[i], agsm.n[i]))
            self.assertEqual(i, agsm.index_nl(agsm.n[i], agsm.l[i]))

    def test_load_modelS_amde(self):
        css1, eigs1, x1 = adipls.load_amde('data/modelS_nfmode1.amde', return_object=False)
        css2, eigs2, x2 = adipls.load_amde('data/modelS_nfmode2.amde', return_object=False, nfmode=2)
        css3, eigs3, x3 = adipls.load_amde('data/modelS_nfmode3.amde', return_object=False, nfmode=3)

        for cs1, cs2, cs3 in zip(css1, css2, css3):
            np.testing.assert_equal(cs1, cs2)
            np.testing.assert_equal(cs1, cs3)
            self.assertAlmostEqual(cs1['M'], 1.989e33)
            self.assertAlmostEqual(cs1['R'], 6.959894677e10)

        for eig1, eig2, eig3 in zip(eigs1, eigs2, eigs3):
            np.testing.assert_equal(eig1[:,1:3], eig2)
            np.testing.assert_equal(eig1[:,5:], eig3)
            np.testing.assert_equal(eig1[:,0], x1)
            np.testing.assert_equal(eig1[:,0], x2)
            np.testing.assert_equal(eig1[:,0], x3)

        amde1 = adipls.load_amde('data/modelS_nfmode1.amde', return_object=True)
        amde2 = adipls.load_amde('data/modelS_nfmode2.amde', return_object=True, nfmode=2)
        amde3 = adipls.load_amde('data/modelS_nfmode3.amde', return_object=True, nfmode=3)

        np.testing.assert_equal(amde1.eigs[:,:,1:3], amde2.eigs)
        np.testing.assert_equal(amde1.x, amde2.x)
        np.testing.assert_equal(amde1.x, amde3.x)
        np.testing.assert_equal(amde1.css, amde2.css)
        np.testing.assert_equal(amde1.css, amde3.css)

    def test_load_amde_nfmode_value_error(self):
        self.assertRaises(ValueError, adipls.load_amde, 'data/modelS_nfmode1.amde', nfmode=4)

    def test_load_modelS_amdl(self):
        D, A, nmod = adipls.load_amdl('data/modelS.amdl', return_nmod=True, return_object=False)
        self.assertEqual(nmod, 1)
        self.assertEqual(len(A), 2482)
        self.assertAlmostEqual(D[0], 1.989e33)
        self.assertAlmostEqual(D[1], 6.959894677e10)

    def test_load_modelS_rkr(self):
        css, rkrs = adipls.load_rkr('data/modelS.rkr', return_object=False)

        for cs in css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 6.959894677e10)

        rkr = adipls.load_rkr('data/modelS.rkr', return_object=True)

        for cs in rkr.css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 6.959894677e10)

        np.testing.assert_equal(rkr.K[0], rkr.K_ln(rkr.l[0], rkr.n[0]))
        np.testing.assert_equal(rkr.K[0], rkr.K_nl(rkr.n[0], rkr.l[0]))

    def test_save_mesa_amdl(self):
        D1, A1, nmod1 = adipls.load_amdl('data/mesa.amdl', return_nmod=True, return_object=False)
        adipls.save_amdl(tmpfile, D1, A1, nmod=nmod1)
        D2, A2, nmod2 = adipls.load_amdl(tmpfile, return_nmod=True, return_object=False)
        np.testing.assert_equal(nmod1, nmod2)
        np.testing.assert_equal(len(A1), len(A2))
        np.testing.assert_equal(D1, D2)
        np.testing.assert_equal(A1, A2)

        a1 = adipls.load_amdl('data/mesa.amdl', return_object=True)
        a1.to_file(tmpfile)
        a2 = adipls.load_amdl(tmpfile, return_object=True)
        np.testing.assert_equal(a1.nmod, a2.nmod)
        np.testing.assert_equal(a1.nn, a2.nn)
        np.testing.assert_equal(a1.D, a2.D)
        np.testing.assert_equal(a1.A, a2.A)

    def test_save_modelS_amdl(self):
        D1, A1, nmod1 = adipls.load_amdl('data/modelS.amdl', return_nmod=True, return_object=False)
        adipls.save_amdl(tmpfile, D1, A1, nmod=nmod1)
        D2, A2, nmod2 = adipls.load_amdl(tmpfile, return_nmod=True, return_object=False)
        np.testing.assert_equal(nmod1, nmod2)
        np.testing.assert_equal(len(A1), len(A2))
        np.testing.assert_equal(D1, D2)
        np.testing.assert_equal(A1, A2)

        a1 = adipls.load_amdl('data/modelS.amdl', return_object=True)
        a1.to_file(tmpfile)
        a2 = adipls.load_amdl(tmpfile, return_object=True)
        np.testing.assert_equal(a1.nmod, a2.nmod)
        np.testing.assert_equal(a1.nn, a2.nn)
        np.testing.assert_equal(a1.D, a1.D)
        np.testing.assert_equal(a1.A, a1.A)

    def test_fgong_to_amdl_modelS(self):
        D1, A1 = adipls.load_amdl('data/modelS.amdl', return_object=False)
        glob, var = fgong.load_fgong('data/modelS.fgong', return_object=False)
        D2, A2 = adipls.fgong_to_amdl(glob, var, G=6.67232e-8)

        np.testing.assert_allclose(D1, D2)
        np.testing.assert_allclose(A1, A2)

        a1 = adipls.load_amdl('data/modelS.amdl', return_object=True,
                              G=6.67232e-8)
        m2 = fgong.load_fgong('data/modelS.fgong', return_object=True,
                              G=6.67232e-8)
        a2 = m2.to_amdl()

        np.testing.assert_allclose(a1.D, a2.D)
        np.testing.assert_allclose(a1.A, a2.A)

    def test_fgong_to_amdl_mesa(self):
        D1, A1 = adipls.load_amdl('data/mesa.amdl', return_object=False)
        glob, var = fgong.load_fgong('data/mesa.fgong', return_object=False)
        D2, A2 = adipls.fgong_to_amdl(glob, var, G=6.67428e-8)

        np.testing.assert_allclose(D1, D2)
        np.testing.assert_allclose(A1, A2)

        a1 = adipls.load_amdl('data/mesa.amdl', return_object=True,
                              G=6.67428e-8)
        f = fgong.load_fgong('data/mesa.fgong', return_object=True,
                             G=6.67428e-8)
        a2 = f.to_amdl()

        np.testing.assert_allclose(a1.D, a2.D)
        np.testing.assert_allclose(a1.A, a2.A)

    def test_amdl_to_fgong_modelS(self):
        I = [0,3,4,9,14] # only these columns can be restored from AMDL format

        D, A = adipls.load_amdl('data/modelS.amdl', return_object=False)
        glob1, var1 = fgong.load_fgong('data/modelS.fgong', return_object=False)
        glob2, var2 = adipls.amdl_to_fgong(D, A, G=6.67232e-8)

        np.testing.assert_allclose(glob1[:2], glob2[:2])
        np.testing.assert_allclose(var1[:-1,I], var2[:-1,I])

        m1 = fgong.load_fgong('data/modelS.fgong', return_object=True,
                              G=6.67232e-8)
        a2 = adipls.load_amdl('data/modelS.amdl', return_object=True,
                              G=6.67232e-8)
        m2 = a2.to_fgong()

        np.testing.assert_allclose(m1.glob[:2], m2.glob[:2])
        np.testing.assert_allclose(m1.var[:-1,I], m2.var[:-1,I])

    def test_amdl_to_fgong_mesa(self):
        I = [0,3,4,9,14] # only these columns can be restored from AMDL format

        D, A = adipls.load_amdl('data/mesa.amdl', return_object=False)
        glob1, var1 = fgong.load_fgong('data/mesa.fgong', return_object=False)
        glob2, var2 = adipls.amdl_to_fgong(D, A, G=6.67428e-8)

        np.testing.assert_allclose(glob1[:2], glob2[:2])
        np.testing.assert_allclose(var1[:,I], var2[:,I])

        m1 = fgong.load_fgong('data/mesa.fgong', return_object=True,
                              G=6.67428e-8)
        a2 = adipls.load_amdl('data/mesa.amdl', return_object=True,
                              G=6.67428e-8)
        m2 = a2.to_fgong()

        np.testing.assert_allclose(m1.glob[:2], m2.glob[:2])
        np.testing.assert_allclose(m1.var[:,I], m2.var[:,I])

    def test_cross_check_css(self):
        css_agsm = adipls.load_agsm('data/modelS.agsm', return_object=False)
        css_amde = adipls.load_amde('data/modelS_nfmode1.amde', return_object=False)[0]
        css_rkr = adipls.load_rkr('data/modelS.rkr', return_object=False)[0]
        np.testing.assert_equal(css_agsm, css_amde)
        np.testing.assert_equal(css_agsm, css_rkr)

        agsm = adipls.load_agsm('data/modelS.agsm', return_object=True)
        amde = adipls.load_amde('data/modelS_nfmode1.amde', return_object=True)
        rkr = adipls.load_rkr('data/modelS.rkr', return_object=True)
        np.testing.assert_equal(agsm.css, amde.css)
        np.testing.assert_equal(agsm.css, rkr.css)

if __name__ == '__main__':
    unittest.main()
