from tomso import io
import numpy as np
import unittest

tmpfile = 'data/tmpfile'


class TestIOFunctions(unittest.TestCase):

    def test_load_fgong(self):
        glob, var, comment = io.load_fgong('data/modelS.fgong', return_comment=True)
        self.assertEqual(comment[0][:6], 'L5BI.D')
        self.assertEqual(len(glob), 15)
        self.assertEqual(len(var), 2482)
        self.assertEqual(len(var[0]), 30)

        self.assertAlmostEqual(glob[0], 1.989e33)
        self.assertAlmostEqual(glob[1], 6.959906258e10)
        self.assertAlmostEqual(glob[2], 3.845999350e33)

    def test_save_fgong(self):
        glob1, var1, comment1 = io.load_fgong('data/modelS.fgong', return_comment=True)
        io.save_fgong(tmpfile, glob1, var1, comment=comment1, fmt='%16.9E')
        glob2, var2, comment2 = io.load_fgong(tmpfile, return_comment=True)
        for i in range(len(glob1)):
            self.assertAlmostEqual(glob1[i], glob2[i])
            
        for line1, line2 in zip(comment1, comment2):
            self.assertEqual(line1, line2)

        for i in range(len(var1)):
            for j in range(len(var1[i])):
                self.assertAlmostEqual(var1[i,j], var2[i,j])

    def test_load_gyre(self):
        header, data = io.load_gyre('data/mesa.gyre')
        self.assertEqual(header['n'], 601)
        self.assertAlmostEqual(header['M'], 1.9882053999999999E+33)
        self.assertAlmostEqual(header['R'], 6.2045507132959908E+10)
        self.assertAlmostEqual(header['L'], 3.3408563666602257E+33)
        self.assertEqual(header['version'], 101)

    def test_save_gyre(self):
        header1, data1 = io.load_gyre('data/mesa.gyre')
        io.save_gyre(tmpfile, header1, data1)
        header2, data2 = io.load_gyre(tmpfile)
        self.assertEqual(header1, header2)
        for row1, row2 in zip(data1, data2):
            self.assertEqual(row1, row2)


if __name__ == '__main__':
    unittest.main()
