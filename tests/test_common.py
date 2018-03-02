from tomso import common
import numpy as np
import unittest

EPS = np.finfo(float).eps

class TestCommonFunctions(unittest.TestCase):

    def test_integrate_constant(self):
        x = np.linspace(0., 4., 101)
        y = np.ones_like(x)

        self.assertTrue(np.all(common.integrate(y, x) == x))

    def test_integrate_line(self):
        x = np.linspace(0., 4., 101)
        y = x

        self.assertTrue(np.allclose(common.integrate(y, x), x**2/2., atol=4.*EPS))


if __name__ == '__main__':
    unittest.main()
