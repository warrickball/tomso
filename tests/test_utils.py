from tomso import utils
import numpy as np
import unittest

EPS = np.finfo(float).eps

class TestUtilsFunctions(unittest.TestCase):

    def test_integrate_constant(self):
        x = np.linspace(0., 4., 101)
        y = np.ones_like(x)

        np.testing.assert_equal(utils.integrate(y, x), x)

    def test_integrate_line(self):
        x = np.linspace(0., 4., 101)
        y = x

        np.testing.assert_allclose(utils.integrate(y, x),
                                    x**2/2., atol=4.*EPS)

    def test_complement_constant(self):
        x = np.linspace(0., 4., 101)
        y = np.ones_like(x)

        np.testing.assert_equal(utils.complement(y, x), x[-1]-x)

    def test_complement_line(self):
        x = np.linspace(0., 4., 101)
        y = x

        np.testing.assert_allclose(utils.complement(y, x),
                                    (x[-1]**2-x**2)/2., atol=4.*EPS)

    def test_tomso_open(self):
        with utils.tomso_open('data/test.txt', 'rb') as f:
            lines = f.read().decode('utf-8').split('\n')

        with utils.tomso_open('data/test.txt.gz', 'rb') as f:
            lines_gz = f.read().decode('utf-8').split('\n')

        for line, line_gz in zip(lines, lines_gz):
            self.assertEqual(line, line_gz)


    def test_load_mesa_gyre_error(self):
        self.assertRaises(ValueError, utils.load_mesa_gyre, 'data/mesa.history', 'asdf')
            

if __name__ == '__main__':
    unittest.main()
