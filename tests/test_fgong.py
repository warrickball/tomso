from tomso import fgong
import numpy as np
import unittest

tmpfile = 'data/tmpfile'
remote_url = 'https://raw.githubusercontent.com/warrickball/tomso/master/tests/'
EPS = np.finfo(float).eps

class TestFGONGFunctions(unittest.TestCase):

    def test_load_fgong(self):
        glob, var, comment = fgong.load_fgong('data/modelS.fgong', return_comment=True, return_object=False)
        self.assertEqual(comment[0][:6], 'L5BI.D')
        self.assertEqual(len(glob), 15)
        self.assertEqual(len(var), 2482)
        self.assertEqual(len(var[0]), 30)

        self.assertAlmostEqual(glob[0], 1.989e33)
        self.assertAlmostEqual(glob[1], 6.959906258e10)
        self.assertAlmostEqual(glob[2], 3.845999350e33)

        m = fgong.load_fgong('data/modelS.fgong', return_object=True)
        self.assertEqual(len(m), 2482)
        self.assertEqual(glob[0], m.M)
        self.assertEqual(glob[1], m.R)
        self.assertEqual(glob[2], m.L)
        np.testing.assert_equal(m.m, np.exp(var[:,1])*glob[0])
        np.testing.assert_equal(m.x, var[:,0]/glob[1])
        np.testing.assert_equal(m.G1, m.Gamma_1)

        r = fgong.load_fgong(remote_url + 'data/modelS.fgong', return_object=True)

        for line1, line2 in zip(m.description, r.description):
            self.assertEqual(line1, line2)

        np.testing.assert_equal(m.glob, r.glob)
        np.testing.assert_equal(m.var, r.var)

    def test_save_fgong(self):
        glob1, var1, comment1 = fgong.load_fgong('data/modelS.fgong', return_comment=True, return_object=False)
        fgong.save_fgong(tmpfile, glob1, var1, comment=comment1, float_formatter='ivers')
        glob2, var2, comment2 = fgong.load_fgong(tmpfile, return_comment=True, return_object=False)

        for line1, line2 in zip(comment1, comment2):
            self.assertEqual(line1, line2)

        np.testing.assert_allclose(glob1, glob2)
        np.testing.assert_allclose(var1, var2)

        m1 = fgong.load_fgong('data/modelS.fgong', return_object=True)
        m1.to_file(tmpfile, float_formatter='ivers')
        m2 = fgong.load_fgong(tmpfile, return_object=True)

        for line1, line2 in zip(m1.description, m2.description):
            self.assertEqual(line1, line2)

        np.testing.assert_allclose(m1.glob, m2.glob)
        np.testing.assert_allclose(m1.var, m2.var)

    def test_fgong_get(self):
        glob, var = fgong.load_fgong('data/modelS.fgong', return_object=False)
        x, r, R, M = fgong.fgong_get(['x','r','R','M'], glob, var)
        np.testing.assert_equal(x, r/R)
        self.assertAlmostEqual(M, 1.989e33)
        self.assertAlmostEqual(R, 6.959906258e10)

        cs, = fgong.fgong_get(['cs'], glob, var)
        cs2 = fgong.fgong_get('cs2', glob, var)
        np.testing.assert_allclose(cs**2, cs2)

        m = fgong.load_fgong('data/modelS.fgong', return_object=True)
        np.testing.assert_equal(m.x, m.r/m.R)
        self.assertAlmostEqual(m.M, 1.989e33)
        self.assertAlmostEqual(m.R, 6.959906258e10)
        np.testing.assert_allclose(m.cs**2, m.cs2)

    def test_fgong_get_reverse(self):
        glob, var = fgong.load_fgong('data/modelS.fgong', return_object=False)
        x_fwd, = fgong.fgong_get(['x'], glob, var)
        x_rev = fgong.fgong_get('x', glob, var, reverse=True)
        np.testing.assert_equal(x_fwd[::-1], x_rev)

        m = fgong.load_fgong('data/modelS.fgong', return_object=True)
        m_rev = fgong.load_fgong('data/modelS.fgong', return_object=True)
        m_rev.var = m_rev.var[::-1]
        for attr in ['x', 'rho', 'P', 'tau', 'cs']:
            np.testing.assert_allclose(getattr(m, attr), getattr(m_rev, attr)[::-1])

    def test_fgong_get_cross_check(self):
        glob, var = fgong.load_fgong('data/modelS.fgong', return_object=False)
        M, R, L, r, x, m, q, g, rho, P, Hp, G1, T, X, L_r, kappa, epsilon, cs2, cs, tau \
            = fgong.fgong_get(['M','R', 'L', 'r', 'x', 'm', 'q', 'g',
                               'rho', 'P', 'Hp', 'G1', 'T', 'X', 'L_r',
                               'kappa', 'epsilon', 'cs2', 'cs', 'tau'],
                           glob, var)
        np.testing.assert_allclose(q, m/M, rtol=4*EPS)
        np.testing.assert_allclose(x, r/R, rtol=4*EPS)
        np.testing.assert_allclose(Hp[:-1], (P/rho/g)[:-1], rtol=4*EPS)
        np.testing.assert_allclose(cs2, G1*P/rho, rtol=4*EPS)

        m = fgong.load_fgong('data/modelS.fgong', return_object=True)
        np.testing.assert_allclose(m.q, m.m/m.M, rtol=4*EPS)
        np.testing.assert_allclose(m.x, m.r/m.R, rtol=4*EPS)
        np.testing.assert_allclose(m.Hp[:-1], (m.P/m.rho/m.g)[:-1], rtol=4*EPS)
        np.testing.assert_allclose(m.cs2, m.Gamma_1*m.P/m.rho, rtol=4*EPS)

if __name__ == '__main__':
    unittest.main()
