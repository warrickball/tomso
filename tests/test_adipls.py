from tomso import adipls, fgong
import numpy as np
import unittest

tmpfile = 'data/tmpfile'
EPS = np.finfo(float).eps

class TestADIPLSFunctions(unittest.TestCase):

    def test_load_mesa_amdl(self):
        D, A, nmod = adipls.load_amdl('data/mesa.amdl', return_nmod=True)
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
            D, A = adipls.load_amdl(tmpfile)

        with self.assertRaises(IOError):
            a = adipls.load_amdl(tmpfile, return_object=True)

    def test_amdl_get(self):
        D, A = adipls.load_amdl('data/mesa.amdl')
        M, R, x, G1 = adipls.amdl_get(['M','R','x','G1'], D, A)
        self.assertEqual(M, D[0])
        self.assertEqual(R, D[1])
        self.assertTrue(np.all(x==A[:,0]))
        self.assertTrue(np.all(G1==A[:,3]))

        cs, = adipls.amdl_get(['cs'], D, A)
        cs2 = adipls.amdl_get('cs2', D, A)
        self.assertTrue(np.allclose(cs**2, cs2))

        m = adipls.load_amdl('data/mesa.amdl', return_object=True)
        self.assertEqual(m.M, m.D[0])
        self.assertEqual(m.R, m.D[1])
        self.assertTrue(np.all(m.x==m.A[:,0]))
        self.assertTrue(np.all(m.Gamma_1==m.A[:,3]))
        self.assertTrue(np.all(m.G1==m.A[:,3]))
        self.assertTrue(np.all(m.G1==m.Gamma_1))
        self.assertTrue(np.allclose(m.cs**2, m.cs2))

    def test_amdl_get_cross_check(self):
        D, A = adipls.load_amdl('data/mesa.amdl')
        M, R, P_c, rho_c, r, x, m, q, g, rho, P, Hp, G1, cs2, cs, tau \
            = adipls.amdl_get(['M','R', 'P_c', 'rho_c', 'r', 'x', 'm',
                                'q', 'g', 'rho', 'P', 'Hp', 'G1', 'cs2',
                                'cs', 'tau'], D, A)
        self.assertTrue(np.allclose(q, m/M, rtol=4*EPS))
        self.assertTrue(np.allclose(x, r/R, rtol=4*EPS))
        self.assertTrue(np.allclose(Hp, P/(rho*g), rtol=4*EPS, equal_nan=True))
        self.assertTrue(np.allclose(cs2, G1*P/rho, rtol=4*EPS))

        m = adipls.load_amdl('data/mesa.amdl', return_object=True)
        self.assertTrue(np.allclose(m.q, m.m/m.M, rtol=4*EPS))
        self.assertTrue(np.allclose(m.x, m.r/m.R, rtol=4*EPS))
        self.assertTrue(np.allclose(m.Hp, m.P/(m.rho*m.g), rtol=4*EPS, equal_nan=True))
        self.assertTrue(np.allclose(m.cs2, m.Gamma_1*m.P/m.rho, rtol=4*EPS))

    def test_load_modelS_agsm(self):
        css = adipls.load_agsm('data/modelS.agsm')
        for cs in css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 69599062580.0)

        agsm = adipls.load_agsm('data/modelS.agsm', return_object=True)
        for cs in agsm.css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 69599062580.0)

    def test_load_modelS_amde(self):
        css1, eigs1 = adipls.load_amde('data/modelS_nfmode1.amde')
        css2, eigs2, x2 = adipls.load_amde('data/modelS_nfmode2.amde', nfmode=2)
        css3, eigs3, x3 = adipls.load_amde('data/modelS_nfmode3.amde', nfmode=3)

        for cs1, cs2, cs3 in zip(css1, css2, css3):
            self.assertTrue(np.all(cs1==cs2))
            self.assertTrue(np.all(cs1==cs3))
            self.assertAlmostEqual(cs1['M'], 1.989e33)
            self.assertAlmostEqual(cs1['R'], 69599062580.0)

        for eig1, eig2, eig3 in zip(eigs1, eigs2, eigs3):
            self.assertTrue(np.all(eig1[:,1:3] == eig2))
            self.assertTrue(np.all(eig1[:,5:] == eig3))
            self.assertTrue(np.all(eig1[:,0] == x2))
            self.assertTrue(np.all(eig1[:,0] == x3))

        # repeat test using objects once implemented

    def test_load_amde_nfmode_value_error(self):
        self.assertRaises(ValueError, adipls.load_amde, 'data/modelS_nfmode1.amde', nfmode=4)

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

        rkr = adipls.load_rkr('data/modelS.rkr', return_object=True)

        for cs in rkr.css:
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

        a1 = adipls.load_amdl('data/mesa.amdl', return_object=True)
        a1.to_file(tmpfile)
        a2 = adipls.load_amdl(tmpfile, return_object=True)
        self.assertTrue(np.all(a1.nmod == a2.nmod))
        self.assertTrue(np.all(a1.nn == a2.nn))
        self.assertTrue(np.all(a1.D == a2.D))
        self.assertTrue(np.all(a1.A == a2.A))

    def test_save_modelS_amdl(self):
        D1, A1, nmod1 = adipls.load_amdl('data/modelS.amdl', return_nmod=True)
        adipls.save_amdl(tmpfile, D1, A1, nmod=nmod1)
        D2, A2, nmod2 = adipls.load_amdl(tmpfile, return_nmod=True)
        self.assertTrue(np.all(nmod1 == nmod2))
        self.assertTrue(np.all(len(A1) == len(A2)))
        self.assertTrue(np.all(D1 == D2))
        self.assertTrue(np.all(A1 == A2))

        a1 = adipls.load_amdl('data/modelS.amdl', return_object=True)
        a1.to_file(tmpfile)
        a2 = adipls.load_amdl(tmpfile, return_object=True)
        self.assertTrue(np.all(a1.nmod == a2.nmod))
        self.assertTrue(np.all(a1.nn == a2.nn))
        self.assertTrue(np.all(a1.D == a1.D))
        self.assertTrue(np.all(a1.A == a1.A))

    def test_fgong_to_amdl_modelS(self):
        D1, A1 = adipls.load_amdl('data/modelS.amdl')
        glob, var = fgong.load_fgong('data/modelS.fgong')
        D2, A2 = adipls.fgong_to_amdl(glob, var, G=6.67232e-8)

        for (x,y) in zip(D1, D2):
            self.assertAlmostEqual(x,y)

        for i in range(len(A1)):
            for (x,y) in zip(A1[i], A2[i]):
                self.assertAlmostEqual(x,y)

        a1 = adipls.load_amdl('data/modelS.amdl', return_object=True,
                              G=6.67232e-8)
        m2 = fgong.load_fgong('data/modelS.fgong', return_object=True,
                              G=6.67232e-8)
        a2 = m2.to_amdl()

        for (x,y) in zip(a1.D, a2.D):
            self.assertAlmostEqual(x,y)

        for i in range(a1.nn):
            for (x,y) in zip(a1.A[i], a2.A[i]):
                self.assertAlmostEqual(x,y)

    def test_fgong_to_amdl_mesa(self):
        D1, A1 = adipls.load_amdl('data/mesa.amdl')
        glob, var = fgong.load_fgong('data/mesa.fgong')
        D2, A2 = adipls.fgong_to_amdl(glob, var, G=6.67428e-8)

        for (x,y) in zip(D1, D2):
            self.assertAlmostEqual(x,y)

        for i in range(len(A1)):
            for (x,y) in zip(A1[i], A2[i]):
                self.assertAlmostEqual(x,y)

        a1 = adipls.load_amdl('data/mesa.amdl', return_object=True,
                              G=6.67428e-8)
        f = fgong.load_fgong('data/mesa.fgong', return_object=True,
                             G=6.67428e-8)
        a2 = f.to_amdl()

        for (x,y) in zip(a1.D, a2.D):
            self.assertAlmostEqual(x,y)

        for i in range(len(a1.A)):
            for (x,y) in zip(a1.A[i], a2.A[i]):
                self.assertAlmostEqual(x,y)

    def test_amdl_to_fgong_modelS(self):
        D, A = adipls.load_amdl('data/modelS.amdl')
        glob1, var1 = fgong.load_fgong('data/modelS.fgong')
        glob2, var2 = adipls.amdl_to_fgong(D, A, G=6.67232e-8)

        for i in [0, 1]:
            self.assertAlmostEqual(glob1[i], glob2[i])

        for i in range(len(var1)-1):
            self.assertAlmostEqual(np.exp(var2[i,1]-var1[i,1]), 1.)
            for j in [0,3,4,9,14]:
                self.assertAlmostEqual(var2[i,j]/var1[i,j], 1.)

        m1 = fgong.load_fgong('data/modelS.fgong', return_object=True,
                              G=6.67232e-8)
        a2 = adipls.load_amdl('data/modelS.amdl', return_object=True,
                              G=6.67232e-8)
        m2 = a2.to_fgong()

        for i in [0, 1]:
            self.assertAlmostEqual(m1.glob[i], m2.glob[i])

        for i in range(len(var1)-1):
            self.assertAlmostEqual(np.exp(m2.var[i,1]-m1.var[i,1]), 1.)
            for j in [0,3,4,9,14]:
                self.assertAlmostEqual(m2.var[i,j]/m1.var[i,j], 1.)

    def test_amdl_to_fgong_mesa(self):
        D, A = adipls.load_amdl('data/mesa.amdl')
        glob1, var1 = fgong.load_fgong('data/mesa.fgong')
        glob2, var2 = adipls.amdl_to_fgong(D, A, G=6.67428e-8)

        for i in [0, 1]:
            self.assertAlmostEqual(glob1[i], glob2[i])

        for i in range(len(var1)-1):
            self.assertAlmostEqual(np.exp(var2[i,1]-var1[i,1]), 1.)
            for j in [0,3,4,9,14]:
                self.assertAlmostEqual(var2[i,j]/var1[i,j], 1.)

        m1 = fgong.load_fgong('data/mesa.fgong', return_object=True,
                              G=6.67428-8)
        a2 = adipls.load_amdl('data/mesa.amdl', return_object=True,
                              G=6.67428e-8)
        m2 = a2.to_fgong()

        for i in [0, 1]:
            self.assertAlmostEqual(m1.glob[i], m2.glob[i])

        for i in range(len(var1)-1):
            self.assertAlmostEqual(np.exp(m2.var[i,1]-m1.var[i,1]), 1.)
            for j in [0,3,4,9,14]:
                self.assertAlmostEqual(m2.var[i,j]/m1.var[i,j], 1.)

    def test_cross_check_css(self):
        css_agsm = adipls.load_agsm('data/modelS.agsm')
        css_amde = adipls.load_amde('data/modelS_nfmode1.amde')[0]
        css_rkr = adipls.load_rkr('data/modelS.rkr')[0]
        self.assertTrue(np.all(css_agsm == css_amde))
        self.assertTrue(np.all(css_agsm == css_rkr))

        agsm = adipls.load_agsm('data/modelS.agsm', return_object=True)
        # amde = adipls.load_amde('data/modelS_nfmode1.amde', return_object=True)
        rkr = adipls.load_rkr('data/modelS.rkr', return_object=True)
        # self.assertTrue(np.all(css_agsm == css_amde))
        self.assertTrue(np.all(agsm.css == rkr.css))

if __name__ == '__main__':
    unittest.main()
