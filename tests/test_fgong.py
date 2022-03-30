from tomso import fgong
import numpy as np
import unittest

tmpfile = 'data/tmpfile'
remote_url = 'https://raw.githubusercontent.com/warrickball/tomso/master/tests/'
EPS = np.finfo(float).eps

class TestFGONGFunctions(unittest.TestCase):

    def test_load_fgong(self):
        m = fgong.load_fgong('data/modelS.fgong')
        self.assertEqual(len(m), 2482)
        self.assertEqual(m.glob[0], m.M)
        self.assertEqual(m.glob[1], m.R)
        self.assertEqual(m.glob[2], m.L)
        np.testing.assert_equal(m.m, np.exp(m.var[:,1])*m.glob[0])
        np.testing.assert_equal(m.x, m.var[:,0]/m.glob[1])
        np.testing.assert_equal(m.G1, m.Gamma_1)

        np.testing.assert_allclose(m.Gamma_1/(m.Gamma_3-1),
                                   m.Gamma_2/(m.Gamma_2-1),
                                   rtol=1e-14, atol=1e-14)

        s = '%s' % m
        s = '%r' % m

        r = fgong.load_fgong(remote_url + 'data/modelS.fgong')

        for line1, line2 in zip(m.description, r.description):
            self.assertEqual(line1, line2)

        np.testing.assert_equal(m.glob, r.glob)
        np.testing.assert_equal(m.var, r.var)

    def test_save_fgong(self):
        m1 = fgong.load_fgong('data/modelS.fgong')
        m1.to_file(tmpfile, float_formatter='ivers')
        m2 = fgong.load_fgong(tmpfile)

        for line1, line2 in zip(m1.description, m2.description):
            self.assertEqual(line1, line2)

        np.testing.assert_allclose(m1.glob, m2.glob)
        np.testing.assert_allclose(m1.var, m2.var)

    def test_fgong_get(self):
        m = fgong.load_fgong('data/modelS.fgong')
        np.testing.assert_equal(m.x, m.r/m.R)
        self.assertAlmostEqual(m.M, 1.989e33)
        self.assertAlmostEqual(m.R, 6.959894677e10)
        np.testing.assert_allclose(m.cs**2, m.cs2)

        self.assertRaises(ValueError, fgong.fgong_get, ['asdf'], m.glob, m.var)

    def test_fgong_get_reverse(self):
        m = fgong.load_fgong('data/modelS.fgong')
        m_rev = fgong.load_fgong('data/modelS.fgong')
        m_rev.var = m_rev.var[::-1]
        for attr in ['x', 'rho', 'P', 'tau', 'cs']:
            np.testing.assert_allclose(getattr(m, attr), getattr(m_rev, attr)[::-1])

    def test_fgong_get_cross_check(self):
        m = fgong.load_fgong('data/modelS.fgong')
        M, R, L, r, x, m, q, g, rho, P, Hp, G1, T, X, L_r, kappa, epsilon, cs2, cs, tau \
            = fgong.fgong_get(['M','R', 'L', 'r', 'x', 'm', 'q', 'g',
                               'rho', 'P', 'Hp', 'G1', 'T', 'X', 'L_r',
                               'kappa', 'epsilon', 'cs2', 'cs', 'tau'],
                           m.glob, m.var)
        np.testing.assert_allclose(q, m/M, rtol=4*EPS)
        np.testing.assert_allclose(x, r/R, rtol=4*EPS)
        np.testing.assert_allclose(Hp[:-1], (P/rho/g)[:-1], rtol=4*EPS)
        np.testing.assert_allclose(cs2, G1*P/rho, rtol=4*EPS)

        m = fgong.load_fgong('data/modelS.fgong')
        np.testing.assert_allclose(m.q, m.m/m.M, rtol=4*EPS)
        np.testing.assert_allclose(m.x, m.r/m.R, rtol=4*EPS)
        np.testing.assert_allclose(m.Hp[:-1], (m.P/m.rho/m.g)[:-1], rtol=4*EPS)
        np.testing.assert_allclose(m.cs2, m.Gamma_1*m.P/m.rho, rtol=4*EPS)

if __name__ == '__main__':
    unittest.main()
