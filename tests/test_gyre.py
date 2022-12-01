from tomso import gyre
import numpy as np
import unittest

tmpfile = 'data/tmpfile'
remote_url = 'https://raw.githubusercontent.com/warrickball/tomso/master/tests/'

class TestGYREFunctions(unittest.TestCase):

    def test_load_summary(self):
        s = gyre.load_summary('data/gyre.summ')
        self.assertAlmostEqual(s['M_star'], 1.989e33)
        self.assertAlmostEqual(s['R_star'], 6.959906258e10)
        self.assertEqual(len(s), 3)
        for i, row in enumerate(s):
            self.assertEqual(row['l'], 1)
            self.assertEqual(row['n_pg'], i+19)

        self.assertRaises(KeyError, s.__getitem__, 'asdf')

        x = '%s' % s
        x = '%r' % s

        r = gyre.load_summary(remote_url + 'data/gyre.summ')
        np.testing.assert_equal(s.header, r.header)
        np.testing.assert_equal(s.data, r.data)

        s = gyre.load_summary('data/gyre_noheader.summ')
        for i, row in enumerate(s):
            self.assertEqual(row['l'], 0)
            self.assertEqual(row['n_pg'], i+8)

        r = gyre.load_summary(remote_url + 'data/gyre_noheader.summ')
        np.testing.assert_equal(s.header, r.header)
        np.testing.assert_equal(s.data, r.data)

    def test_load_mode(self):
        for i in range(3):
            m = gyre.load_mode('data/gyre.mode_%i' % (i+1))
            self.assertEqual(m['n_pg'], i+19)
            self.assertEqual(m['l'], 1)
            self.assertEqual(m['Imomega'], 0.0)
            self.assertEqual(m['Imfreq'], 0.0)

            np.testing.assert_equal(m['Imxi_r'], 0.0)
            np.testing.assert_equal(m['Imxi_h'], 0.0)

    def test_load_gyre(self):
        m = gyre.load_gyre('data/mesa.gyre')
        self.assertEqual(len(m.data), 601)
        self.assertAlmostEqual(m.M, 1.9882053999999999E+33)
        self.assertAlmostEqual(m.R, 6.2045507132959908E+10)
        self.assertAlmostEqual(m.L, 3.3408563666602257E+33)
        self.assertEqual(m.version, 101)

        np.testing.assert_allclose(m.Omega, 0)
        np.testing.assert_allclose(m.Gamma_1/(m.Gamma_3-1),
                                   m.Gamma_2/(m.Gamma_2-1),
                                   rtol=1e-14, atol=1e-14)

        s = '%r' % m

        r = gyre.load_gyre(remote_url + 'data/mesa.gyre')
        np.testing.assert_equal(m.header, r.header)
        np.testing.assert_equal(m.data, r.data)

    def test_load_spb_mesa_versions(self):
        filenames = ['data/spb.mesa.78677cc', 'data/spb.mesa.813eed2',
                     'data/spb.mesa.adc6989']
        for filename in filenames:
            m1 = gyre.load_gyre('data/mesa.gyre')
            m1.to_plain(tmpfile)
            m2 = gyre.load_gyre(tmpfile)

            np.testing.assert_equal(m1.header, m2.header)
            np.testing.assert_equal(m1.data, m2.data)

            np.testing.assert_allclose(m1.r, m2.x*m2.R)
            np.testing.assert_allclose(m1.cs2, m2.Gamma_1*m2.P/m2.rho)
            np.testing.assert_allclose(m1.AA[1:], m2.N2[1:]*m2.r[1:]/m2.g[1:])

    def test_save_gyre(self):
        m1 = gyre.load_gyre('data/mesa.gyre')
        m1.to_plain(tmpfile)
        m2 = gyre.load_gyre(tmpfile)

        np.testing.assert_equal(m1.header, m2.header)
        np.testing.assert_equal(m1.data, m2.data)

        np.testing.assert_allclose(m1.r, m2.x*m2.R)
        np.testing.assert_allclose(m1.cs2, m2.Gamma_1*m2.P/m2.rho)
        np.testing.assert_allclose(m1.AA[1:], m2.N2[1:]*m2.r[1:]/m2.g[1:])


if __name__ == '__main__':
    unittest.main()
