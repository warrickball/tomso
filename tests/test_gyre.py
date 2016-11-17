from tomso import gyre
import numpy as np
import unittest

tmpfile = 'data/tmpfile'

class TestGYREFunctions(unittest.TestCase):

    def test_load_summary(self):
        header, data = gyre.load_summary('data/gyre.summ')
        self.assertAlmostEqual(header['M_star'], 1.989e33)
        self.assertAlmostEqual(header['R_star'], 6.959894677e10)
        for i, row in enumerate(data):
            self.assertEqual(row['l'], 1000)
            self.assertEqual(row['n_pg'], i)

    def test_load_mode(self):
        for i in range(3):
            header, data = gyre.load_mode('data/gyre.mode%i' % (i+1))
            self.assertEqual(header['n_pg'], i)
            for k in ['x','Rexi_r','Imxi_r','Rexi_h','Imxi_h']:
                self.assertAlmostEqual(data[k][0], 0.0)


if __name__ == '__main__':
    unittest.main()

