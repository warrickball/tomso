from tomso import fgong
import numpy as np
import unittest

tmpfile = 'data/tmpfile'
EPS = np.finfo(float).eps

class TestFGONGFunctions(unittest.TestCase):

    def test_load_fgong(self):
        glob, var, comment = fgong.load_fgong('data/modelS.fgong', return_comment=True)
        self.assertEqual(comment[0][:6], 'L5BI.D')
        self.assertEqual(len(glob), 15)
        self.assertEqual(len(var), 2482)
        self.assertEqual(len(var[0]), 30)

        self.assertAlmostEqual(glob[0], 1.989e33)
        self.assertAlmostEqual(glob[1], 6.959906258e10)
        self.assertAlmostEqual(glob[2], 3.845999350e33)

        m = fgong.load_fgong('data/modelS.fgong', return_object=True)
        self.assertEqual(glob[0], m.M)
        self.assertEqual(glob[1], m.R)
        self.assertEqual(glob[2], m.L)
        self.assertTrue(np.all(m.m == np.exp(var[:,1])*glob[0]))
        self.assertTrue(np.all(m.x == var[:,0]/glob[1]))

    def test_save_fgong(self):
        glob1, var1, comment1 = fgong.load_fgong('data/modelS.fgong', return_comment=True)
        fgong.save_fgong(tmpfile, glob1, var1, comment=comment1, fmt='%16.9E')
        glob2, var2, comment2 = fgong.load_fgong(tmpfile, return_comment=True)
        for i in range(len(glob1)):
            self.assertAlmostEqual(glob1[i], glob2[i])

        for line1, line2 in zip(comment1, comment2):
            self.assertEqual(line1, line2)

        for i in range(len(var1)):
            for j in range(len(var1[i])):
                self.assertAlmostEqual(var1[i,j], var2[i,j])

        m1 = fgong.load_fgong('data/modelS.fgong', return_object=True)
        m1.to_file(tmpfile, fmt='%16.9E')
        m2 = fgong.load_fgong(tmpfile, return_object=True)

        for i in range(m1.iconst):
            self.assertAlmostEqual(m1.glob[i], m2.glob[i])

        for line1, line2 in zip(m1.description, m2.description):
            self.assertEqual(line1, line2)

        for i in range(m1.nn):
            for j in range(m1.ivar):
                self.assertAlmostEqual(m1.var[i,j], m2.var[i,j])

    def test_fgong_get(self):
        glob, var = fgong.load_fgong('data/modelS.fgong')
        x, r, R, M = fgong.fgong_get(['x','r','R','M'], glob, var)
        self.assertTrue(np.all(x==r/R))
        self.assertAlmostEqual(M, 1.989e33)
        self.assertAlmostEqual(R, 6.959906258e10)

        cs, = fgong.fgong_get(['cs'], glob, var)
        cs2 = fgong.fgong_get('cs2', glob, var)
        self.assertTrue(np.allclose(cs**2, cs2))

        m = fgong.load_fgong('data/modelS.fgong', return_object=True)
        self.assertTrue(np.all(m.x==m.r/m.R))
        self.assertAlmostEqual(m.M, 1.989e33)
        self.assertAlmostEqual(m.R, 6.959906258e10)
        self.assertTrue(np.allclose(m.cs**2, m.cs2))

    def test_fgong_get_reverse(self):
        glob, var = fgong.load_fgong('data/modelS.fgong')
        x_fwd, = fgong.fgong_get(['x'], glob, var)
        x_rev = fgong.fgong_get('x', glob, var, reverse=True)
        self.assertTrue(np.all(x_fwd[::-1] == x_rev))

    def test_fgong_get_cross_check(self):
        glob, var = fgong.load_fgong('data/modelS.fgong')
        M, R, L, r, x, m, q, g, rho, P, Hp, G1, T, X, L_r, kappa, epsilon, cs2, cs, tau \
            = fgong.fgong_get(['M','R', 'L', 'r', 'x', 'm', 'q', 'g',
                               'rho', 'P', 'Hp', 'G1', 'T', 'X', 'L_r',
                               'kappa', 'epsilon', 'cs2', 'cs', 'tau'],
                           glob, var)
        self.assertTrue(np.allclose(q, m/M, rtol=4*EPS))
        self.assertTrue(np.allclose(x, r/R, rtol=4*EPS))
        self.assertTrue(np.allclose(Hp, P/(rho*g), rtol=4*EPS, equal_nan=True))
        self.assertTrue(np.allclose(cs2, G1*P/rho, rtol=4*EPS))

        m = fgong.load_fgong('data/modelS.fgong', return_object=True)
        self.assertTrue(np.allclose(m.q, m.m/m.M, rtol=4*EPS))
        self.assertTrue(np.allclose(m.x, m.r/m.R, rtol=4*EPS))
        self.assertTrue(np.allclose(m.Hp, m.P/(m.rho*m.g), rtol=4*EPS, equal_nan=True))
        self.assertTrue(np.allclose(m.cs2, m.Gamma_1*m.P/m.rho, rtol=4*EPS))

if __name__ == '__main__':
    unittest.main()
