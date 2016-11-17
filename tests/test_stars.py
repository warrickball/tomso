from tomso import stars
import unittest

class TestStarsFunctions(unittest.TestCase):

    def test_load_model(self):
        header, data, ddata = stars.load_model('data/stars.modout')
        self.assertEqual(header[0]['K'], 199)
        self.assertAlmostEqual(header[0]['dt'], 3.86009e6)
        self.assertAlmostEqual(header[0]['t'], 1.077518993e10)


    def test_load_plot(self):
        data = stars.load_plot('data/stars.plot')
        self.assertEqual(data['n'][0], 0)


if __name__ == '__main__':
    unittest.main()

