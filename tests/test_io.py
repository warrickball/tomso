from tomso import io
import numpy as np
import unittest

tmpfile = 'data/tmpfile'

class TestIOFunctions(unittest.TestCase):

    def test_load_fgong(self):
        fgong = io.load_fgong('data/modelS.fgong')
        self.assertEqual(fgong['header'][0][:6], 'L5BI.D')
        self.assertEqual(len(fgong['var']), 2482)
        self.assertEqual(fgong['nn'], 2482)
        self.assertEqual(fgong['iconst'], 15)
        self.assertEqual(fgong['ivar'], 30)
        self.assertEqual(fgong['ivers'], 250)
        # test M, R, L
        self.assertAlmostEqual(fgong['glob'][0], 1.989e33)
        self.assertAlmostEqual(fgong['glob'][1], 6.959906258e10)
        self.assertAlmostEqual(fgong['glob'][2], 3.845999350e33)

    
    def test_save_fgong(self):
        fgong1 = io.load_fgong('data/mesa.fgong')
        io.save_fgong(tmpfile, fgong1)
        fgong2 = io.load_fgong(tmpfile)
        for k in ['nn','iconst','ivar','ivers']:
            self.assertEqual(fgong1[k], fgong2[k])

        for line1, line2 in zip(fgong1['header'], fgong2['header']):
            self.assertEqual(line1, line2)

        I = np.where(np.isfinite(fgong1['var']))
        self.assertTrue(np.allclose(fgong1['var'][I], fgong2['var'][I]))
        I = np.where(np.isfinite(fgong2['var']))
        self.assertTrue(np.allclose(fgong1['var'][I], fgong2['var'][I]))


    def test_load_mesa(self):
        header, data = io.load_mesa('data/mesa.gyre')
        self.assertEqual(header['n'], 600)
        self.assertAlmostEqual(header['M'], 1.9892e33)
        self.assertAlmostEqual(header['R'], 6.19021624544E+10)
        self.assertAlmostEqual(header['L'], 3.29445756181E+33)
        self.assertEqual(header['n_col'], 19)

    
    def test_save_mesa(self):
        header1, data1 = io.load_mesa('data/mesa.gyre')
        io.save_mesa(tmpfile, header1, data1)
        header2, data2 = io.load_mesa(tmpfile)
        self.assertEqual(header1, header2)
        for row1, row2 in zip(data1, data2):
            self.assertEqual(row1, row2)


if __name__ == '__main__':
    unittest.main()

