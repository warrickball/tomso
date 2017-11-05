from tomso import stars
import numpy as np
import unittest


class TestStarsFunctions(unittest.TestCase):

    def test_load_out(self):
        summaries, profiles = stars.load_out('data/stars.out')

        self.assertTrue(np.all(np.abs(profiles[0]['H1']-0.7) < 1e-9))
        self.assertTrue(np.all(np.abs(profiles[0]['He4']-0.28) < 1e-9))

        for profile in profiles:
            self.assertTrue(np.all(profile['k']), np.arange(1,200)[::-1])

        self.assertAlmostEqual(summaries[0]['dt'], 6308.266)

        for i, summary in enumerate(summaries[:10]):
            self.assertEqual(summary['n'], i+1)
            self.assertAlmostEqual(summary['H1_cntr'], 0.7)
            self.assertAlmostEqual(summary['He4_cntr'], 0.28)

    def test_load_plot(self):
        data = stars.load_plot('data/stars.plot')
        self.assertEqual(data['n'][0], 1)


if __name__ == '__main__':
    unittest.main()
