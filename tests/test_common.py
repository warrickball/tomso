from tomso import common
import numpy as np
import unittest

EPS = np.finfo(float).eps

class TestCommonFunctions(unittest.TestCase):

    def test_integrate_constant(self):
        x = np.linspace(0., 4., 101)
        y = np.ones_like(x)

        np.testing.assert_equal(common.integrate(y, x), x)

    def test_integrate_line(self):
        x = np.linspace(0., 4., 101)
        y = x

        np.testing.assert_allclose(common.integrate(y, x),
                                    x**2/2., atol=4.*EPS)

    def test_complement_constant(self):
        x = np.linspace(0., 4., 101)
        y = np.ones_like(x)

        np.testing.assert_equal(common.complement(y, x), x[-1]-x)

    def test_complement_line(self):
        x = np.linspace(0., 4., 101)
        y = x

        np.testing.assert_allclose(common.complement(y, x),
                                    (x[-1]**2-x**2)/2., atol=4.*EPS)

    def test_tomso_open(self):
        with common.tomso_open('data/test.txt', 'rb') as f:
            lines = f.read().decode('utf-8').split('\n')

        with common.tomso_open('data/test.txt.gz', 'rb') as f:
            lines_gz = f.read().decode('utf-8').split('\n')

        for line, line_gz in zip(lines, lines_gz):
            self.assertEqual(line, line_gz)
            

if __name__ == '__main__':
    unittest.main()
