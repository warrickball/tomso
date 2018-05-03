from tomso import gyre
import numpy as np
import unittest

tmpfile = 'data/tmpfile'


class TestGYREFunctions(unittest.TestCase):

    def test_load_summary(self):
        header, data = gyre.load_summary('data/gyre.summ')
        self.assertAlmostEqual(header['M_star'], 1.989e33)
        self.assertAlmostEqual(header['R_star'], 6.959906258e10)
        for i, row in enumerate(data):
            self.assertEqual(row['l'], 1)
            self.assertEqual(row['n_pg'], i+19)

        header, data = gyre.load_summary('data/gyre_noheader.summ')
        for i, row in enumerate(data):
            self.assertEqual(row['l'], 0)
            self.assertEqual(row['n_pg'], i+8)

    def test_load_mode(self):
        for i in range(3):
            header, data = gyre.load_mode('data/gyre.mode_%i' % (i+1))
            self.assertEqual(header['n_pg'], i+19)
            self.assertEqual(header['l'], 1)
            self.assertEqual(header['Imomega'], 0.0)
            self.assertEqual(header['Imfreq'], 0.0)
            for row in data:
                self.assertEqual(row['Imxi_r'], 0.0)
                self.assertEqual(row['Imxi_h'], 0.0)

    def test_load_gyre(self):
        header, data = gyre.load_gyre('data/mesa.gyre')
        self.assertEqual(header['n'], 601)
        self.assertAlmostEqual(header['M'], 1.9882053999999999E+33)
        self.assertAlmostEqual(header['R'], 6.2045507132959908E+10)
        self.assertAlmostEqual(header['L'], 3.3408563666602257E+33)
        self.assertEqual(header['version'], 101)

    def test_save_gyre(self):
        header1, data1 = gyre.load_gyre('data/mesa.gyre')
        gyre.save_gyre(tmpfile, header1, data1)
        header2, data2 = gyre.load_gyre(tmpfile)
        self.assertEqual(header1, header2)
        for row1, row2 in zip(data1, data2):
            self.assertEqual(row1, row2)


if __name__ == '__main__':
    unittest.main()
